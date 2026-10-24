#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <opencv2/opencv.hpp>

#include "io/camera.hpp"
#include "io/gimbal/gimbal.hpp"
#include "tasks/auto_aim/solver.hpp"
#include "tasks/auto_aim/yolo.hpp"
#include "tools/exiter.hpp"
#include "tools/img_tools.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"
#include "tools/plotter.hpp"
#include "tools/yaml.hpp"

const std::string keys =
  "{help h usage ? | | 输出命令行参数说明}"
  "{@config-path   | | yaml配置文件路径 }";

using namespace std::chrono_literals;

int main(int argc, char * argv[])
{
  cv::CommandLineParser cli(argc, argv, keys);
  auto config_path = cli.get<std::string>("@config-path");
  if (cli.has("help") || !cli.has("@config-path")) {
    cli.printMessage();
    return 0;
  }

  // YAML 的偏置单位为度，后续计算和串口命令统一使用弧度。
  const auto yaml = tools::load(config_path);
  const auto enemy_color_name = tools::read<std::string>(yaml, "enemy_color");
  if (enemy_color_name != "red" && enemy_color_name != "blue") {
    tools::logger()->error("[Task1] enemy_color must be red or blue.");
    return 1;
  }
  const auto enemy_color = enemy_color_name == "red" ? auto_aim::red : auto_aim::blue;
  const double yaw_offset = tools::read_or<double>(yaml, "yaw_offset", 0.0) * CV_PI / 180.0;
  const double pitch_offset = tools::read_or<double>(yaml, "pitch_offset", 0.0) * CV_PI / 180.0;
  if (!std::isfinite(yaw_offset) || !std::isfinite(pitch_offset)) {
    tools::logger()->error("[Task1] Invalid yaw/pitch offset.");
    return 1;
  }

  // 初始化工具类
  tools::Exiter exiter;
  tools::Plotter plotter;  // 默认发送到 127.0.0.1:9870

  // 初始化io类
  io::Camera camera(config_path);
  io::Gimbal gimbal(config_path);

  // 初始化auto_aim类
  auto_aim::YOLO yolo(config_path, true);
  auto_aim::Solver solver(config_path);

  cv::Mat img;
  Eigen::Quaterniond q;
  std::chrono::steady_clock::time_point t;

  // 以下为跟随调试的起始参数；pitch 软件限幅需按实机允许范围核对。
  constexpr double pitch_limit = 20.0 * CV_PI / 180.0;
  constexpr double max_yaw_speed = 120.0 * CV_PI / 180.0;   // rad/s
  constexpr double max_pitch_speed = 90.0 * CV_PI / 180.0;  // rad/s
  constexpr double association_gate = 0.2;  // 图像归一化距离，防止跳到另一块板
  constexpr auto lost_timeout = 300ms;

  // 这些状态放在循环外，使目标选择和命令限速能够跨帧连续。
  int frame_count = 0;
  bool has_target = false;
  bool was_controlling = false;
  auto_aim::ArmorName last_name = auto_aim::not_armor;
  Eigen::Vector3d last_position = Eigen::Vector3d::Zero();
  const auto start_time = std::chrono::steady_clock::now();
  auto last_seen = start_time;
  auto last_command_time = start_time;
  float last_yaw_cmd = 0.0f;
  float last_pitch_cmd = 0.0f;

  // Plotter 与 send 使用同一组最终值；任务一的 fire 始终为 false。
  const auto send_command = [&](bool control, float yaw, float pitch, bool tracking_valid) {
    gimbal.send(control, false, yaw, pitch);
    plotter.plot({
      {"time_s", tools::delta_time(std::chrono::steady_clock::now(), start_time)},
      {"yaw_cmd_rad", yaw},
      {"pitch_cmd_rad", pitch},
      {"control_cmd", control},
      {"fire_cmd", false},
      {"tracking_valid", tracking_valid},
    });
  };

  while (!exiter.exit()) {
    // Your code start

    const auto state = gimbal.state();
    const bool state_valid = std::isfinite(state.yaw) && std::isfinite(state.pitch);
    float yaw_cmd = state_valid ? state.yaw : 0.0f;
    float pitch_cmd = state_valid ? state.pitch : 0.0f;
    bool tracking_valid = false;
    bool quit = false;

    if (
      gimbal.mode() != io::GimbalMode::AUTO_AIM ||
      std::chrono::steady_clock::now() - last_seen > lost_timeout) {
      has_target = false;
    }

    try {
      // 使用超时读取；失败时不再处理 img 中可能残留的上一帧。
      const bool got_frame = camera.try_read_for(img, t, 100ms);
      const auto frame_age = std::chrono::steady_clock::now() - t;
      if (got_frame && !img.empty() && frame_age >= 0ms && frame_age <= lost_timeout) {
        // 每帧只取一次对应时刻的姿态，再进行 PnP 和坐标变换。
        q = gimbal.q(t);
        if (q.coeffs().allFinite() && q.norm() > 1e-6) {
          solver.set_R_gimbal2world(q.normalized());
          auto armors = yolo.detect(img, frame_count++);

          // 把上一帧目标投影到当前画面，减小云台转动对目标关联的影响。
          cv::Point2f reference(img.cols * 0.5f, img.rows * 0.5f);
          bool reference_valid = true;
          if (has_target) {
            const auto projected = solver.world2pixel(last_position);
            reference_valid = projected && std::isfinite(projected->x) &&
                              std::isfinite(projected->y);
            if (reference_valid) reference = *projected;
          }

          auto_aim::Armor * selected = nullptr;
          double best_score = std::numeric_limits<double>::infinity();
          for (auto & armor : armors) {
            if (
              armor.color != enemy_color || armor.name < auto_aim::one ||
              armor.name >= auto_aim::not_armor || armor.points.size() != 4 ||
              !std::isfinite(armor.confidence) || !reference_valid) {
              continue;
            }

            // 排除越界、重复、共线或自交的关键点，避免退化的 PnP 输入。
            const bool points_valid = std::all_of(
              armor.points.begin(), armor.points.end(), [&](const cv::Point2f & point) {
                return std::isfinite(point.x) && std::isfinite(point.y) && point.x >= 0 &&
                       point.x < img.cols && point.y >= 0 && point.y < img.rows;
              });
            if (
              !points_valid || !cv::isContourConvex(armor.points) ||
              cv::contourArea(armor.points) < 4.0) {
              continue;
            }

            const double dx = (armor.center.x - reference.x) / img.cols;
            const double dy = (armor.center.y - reference.y) / img.rows;
            const double score = dx * dx + dy * dy;
            if (
              !std::isfinite(score) || score >= best_score ||
              (has_target &&
               (armor.name != last_name || score > association_gate * association_gate))) {
              continue;
            }

            // YOLOv5 未区分特殊板尺寸，在解算前补齐其类型。
            if (armor.name == auto_aim::outpost || armor.name == auto_aim::base) {
              armor.type = auto_aim::special;
            }
            armor.priority = auto_aim::first;
            try {
              solver.solve(armor);
            } catch (const cv::Exception & e) {
              tools::logger()->warn("[Task1] PnP failed: {}", e.what());
              continue;
            }
            if (
              !armor.xyz_in_world.allFinite() || !armor.xyz_in_gimbal.allFinite() ||
              armor.xyz_in_gimbal.x() <= 0 ||
              std::hypot(armor.xyz_in_world.x(), armor.xyz_in_world.y()) < 1e-6) {
              continue;
            }
            selected = &armor;
            best_score = score;
          }

          if (selected != nullptr) {
            const auto now = std::chrono::steady_clock::now();
            last_position = selected->xyz_in_world;
            last_name = selected->name;
            last_seen = now;
            has_target = true;

            // 瞄准目标的位置角，而不是装甲板自身的朝向角 ypr。
            const double target_yaw =
              std::atan2(last_position.y(), last_position.x()) + yaw_offset;
            const double target_pitch = std::clamp(
              -std::atan2(last_position.z(), std::hypot(last_position.x(), last_position.y())) +
                pitch_offset,
              -pitch_limit, pitch_limit);

            if (state_valid) {
              // 假设电控接收世界系绝对角（弧度、抬头为正），yaw 支持连续角。
              // 初次锁定从反馈角出发；后续使用周期角差，跨 ±π 时走最短方向。
              const double base_yaw = was_controlling ? last_yaw_cmd : state.yaw;
              const double base_pitch = was_controlling ? last_pitch_cmd : state.pitch;
              const double dt = std::clamp(tools::delta_time(now, last_command_time), 0.0, 0.1);
              yaw_cmd = base_yaw + std::clamp(
                tools::limit_rad(target_yaw - base_yaw), -max_yaw_speed * dt, max_yaw_speed * dt);
              pitch_cmd = std::clamp(
                base_pitch + std::clamp(
                  target_pitch - base_pitch, -max_pitch_speed * dt, max_pitch_speed * dt),
                -pitch_limit, pitch_limit);
              tracking_valid = std::isfinite(yaw_cmd) && std::isfinite(pitch_cmd);
            }

            tools::draw_points(img, selected->points, {0, 255, 255}, 2);
            tools::draw_point(img, selected->center, {0, 255, 255});
          }

          cv::Mat preview;
          cv::resize(img, preview, {}, 0.5, 0.5);
          cv::imshow("task_1", preview);
        }
      }
      const int key = cv::waitKey(1) & 0xff;
      quit = key == 27 || key == 'q';  // Esc / q 或 Ctrl+C 退出
    } catch (const std::exception & e) {
      tracking_valid = false;
      tools::logger()->warn("[Task1] Frame processing failed: {}", e.what());
    }

    // 丢帧/失锁立即释放控制；短时只保留关联信息，不继续发送旧的跟随指令。
    // 发送前再次检查模式，处理推理过程中退出自瞄的情况。
    if (std::chrono::steady_clock::now() - t > lost_timeout) tracking_valid = false;
    const bool auto_aim = gimbal.mode() == io::GimbalMode::AUTO_AIM;
    const bool control = auto_aim && tracking_valid && !quit && !exiter.exit();
    if (!auto_aim) has_target = false;
    if (!control) {
      yaw_cmd = state_valid ? state.yaw : 0.0f;
      pitch_cmd = state_valid ? state.pitch : 0.0f;
    }
    send_command(control, yaw_cmd, pitch_cmd, tracking_valid);
    was_controlling = control;
    last_yaw_cmd = yaw_cmd;
    last_pitch_cmd = pitch_cmd;
    last_command_time = std::chrono::steady_clock::now();
    if (quit) break;

    // Your code end
  }

  // 正常退出和 Ctrl+C 均显式取消视觉控制，并记录最后一条命令。
  send_command(false, 0.0f, 0.0f, false);
  cv::destroyAllWindows();

  return 0;
}

