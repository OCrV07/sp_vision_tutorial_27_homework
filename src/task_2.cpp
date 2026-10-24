#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <limits>
#include <opencv2/opencv.hpp>
#include <optional>

#include "io/camera.hpp"
#include "io/gimbal/gimbal.hpp"
#include "tasks/auto_aim/solver.hpp"
#include "tasks/auto_aim/target.hpp"
#include "tasks/auto_aim/yolo.hpp"
#include "tools/exiter.hpp"
#include "tools/img_tools.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"
#include "tools/plotter.hpp"
#include "tools/trajectory.hpp"
#include "tools/yaml.hpp"

namespace
{
using Clock = std::chrono::steady_clock;
using TimePoint = Clock::time_point;
constexpr double DEG = CV_PI / 180.0;
constexpr double PITCH_LIMIT = 20.0 * DEG;  // 软件保护范围，上车时核对机械限位

struct Parameters
{
  auto_aim::Color enemy_color;
  double yaw_offset;
  double pitch_offset;
  double execution_delay;  // s，仅包含尚未计入图像处理耗时的通信/执行延迟
};

// 阶段由人工确认，不用估计转速或 AUTO_AIM 猜测摩擦轮状态。
// 窗口获得焦点后，1/2 选档；确认该档已允许射击再按 a，空格立即暂停。
struct ShotController
{
  int stage = 0;  // 0：等待，1：低速，2：中速
  std::array<int, 2> requests{0, 0};
  bool permitted = false;
  std::optional<TimePoint> last_shot;

  void handle_key(int key)
  {
    if (key == '1' && stage <= 1) {
      stage = 1;
      permitted = false;
    } else if (key == '2' && stage >= 1) {
      stage = 2;
      permitted = false;
    } else if (key == 'a' && stage > 0) {
      permitted = requests[stage - 1] < 10;
    } else if (key == ' ') {
      permitted = false;
    }
    // 重复选择档位不清零预算，进入中速后也不能返回低速补充预算。
  }

  bool ready(TimePoint now, bool aim_ready) const
  {
    return stage > 0 && permitted && aim_ready && requests[stage - 1] < 10 &&
           (!last_shot || now - *last_shot >= std::chrono::milliseconds(200));
  }

  void record_request(TimePoint now)
  {
    ++requests[stage - 1];
    last_shot = now;
    if (requests[stage - 1] >= 10) permitted = false;
  }
};

struct AimSolution
{
  bool valid = false;
  bool fireable = false;
  int id = -1;
  double yaw = 0;
  double pitch = 0;
  double fly_time = 0;
  double horizon = 0;
  double yaw_tolerance = 0;
  double pitch_tolerance = 0;
};

bool healthy(const auto_aim::Target & target)
{
  return !target.diverged() && target.ekf().P.diagonal().minCoeff() >= 0 &&
         std::abs(target.ekf_x()[10]) < 0.4;
}

bool valid_detection(const auto_aim::Armor & armor, const Parameters & params, cv::Size size)
{
  // 本任务使用普通四板车辆模型；前哨站、基地和二板平衡车不送入该模型。
  const bool balance = armor.type == auto_aim::big && armor.name >= auto_aim::three &&
                       armor.name <= auto_aim::five;
  if (
    armor.color != params.enemy_color || armor.name < auto_aim::one ||
    armor.name >= auto_aim::outpost || balance || armor.points.size() != 4 ||
    !std::isfinite(armor.confidence)) {
    return false;
  }
  const bool finite_points = std::all_of(
    armor.points.begin(), armor.points.end(), [&](const cv::Point2f & point) {
      return std::isfinite(point.x) && std::isfinite(point.y) && point.x >= 0 &&
             point.x < size.width && point.y >= 0 && point.y < size.height;
    });
  return finite_points && cv::isContourConvex(armor.points) &&
         cv::contourArea(armor.points) >= 4.0;
}

// 先关联同一机器人，再由 Target::update 选择其内部板号。
double association_score(const auto_aim::Armor & armor, const auto_aim::Target & target)
{
  if (armor.name != target.name || armor.type != target.armor_type) {
    return std::numeric_limits<double>::infinity();
  }
  double best = std::numeric_limits<double>::infinity();
  for (const auto & xyza : target.armor_xyza_list()) {
    const double distance = (armor.xyz_in_world - xyza.head<3>()).norm();
    const double angle = std::abs(tools::limit_rad(armor.ypr_in_world[0] - xyza[3]));
    if (distance < 0.6 && angle < 60.0 * DEG) {
      best = std::min(best, distance / 0.6 + angle / (60.0 * DEG));
    }
  }
  return best;
}

AimSolution predict_aim(
  const auto_aim::Target & target, TimePoint model_time, TimePoint now, double bullet_speed,
  int previous_id, const Parameters & params)
{
  AimSolution result;
  const auto state = target.ekf_x();
  const double age = tools::delta_time(now, model_time);
  if (!healthy(target) || age < 0 || age > 0.5) return result;

  // 弹速无效时仍可几何跟随，但不会把这种解用于射击。
  const bool speed_valid = std::isfinite(bullet_speed) && bullet_speed >= 5 && bullet_speed <= 40;
  double flight = speed_valid ? std::hypot(state[0], state[2]) / bullet_speed : 0;
  for (int iteration = 0; iteration < 3; ++iteration) {
    result = {};  // 每轮重新验证，不能在本轮无解时沿用上一轮的开火结论。
    const double horizon = age + params.execution_delay + flight;
    if (!std::isfinite(horizon) || horizon > 1.0) return {};
    auto future = target;  // 在副本上预测，保持主滤波器停留在当前图像时刻。
    future.predict(horizon);

    int chosen = -1;
    double best = std::numeric_limits<double>::infinity();
    double facing = 0;
    const auto candidates = future.armor_xyza_list();
    for (int id = 0; id < static_cast<int>(candidates.size()); ++id) {
      const auto & point = candidates[id];
      if (!point.allFinite()) continue;
      const double bearing = std::atan2(point[1], point[0]);
      const double error = std::abs(tools::limit_rad(point[3] - bearing));
      // 切板惩罚形成滞回，防止两块板正对程度相近时来回跳。
      const double score = error + (id == previous_id ? 0 : 8.0 * DEG);
      if (error < 75.0 * DEG && score < best) {
        chosen = id;
        best = score;
        facing = error;
      }
    }
    if (chosen < 0) return {};

    const auto & point = candidates[chosen];
    const double distance = std::hypot(point[0], point[1]);
    if (distance < 0.05) return {};
    result.valid = true;
    result.id = chosen;
    result.yaw = std::atan2(point[1], point[0]) + params.yaw_offset;
    result.pitch = -std::atan2(point[2], distance) + params.pitch_offset;
    result.horizon = horizon;
    result.yaw_tolerance = std::atan2(
      (target.armor_type == auto_aim::big ? 0.230 : 0.135) / 4.0, distance);
    result.pitch_tolerance = std::atan2(0.056 / 4.0, distance);
    if (!speed_valid) return result;

    // 先采用无阻力低弹道模型；位置须相对于已校准的发射参考原点。
    const tools::Trajectory trajectory(bullet_speed, distance, point[2]);
    if (
      trajectory.unsolvable || !std::isfinite(trajectory.pitch) ||
      !std::isfinite(trajectory.fly_time) || trajectory.fly_time <= 0 ||
      trajectory.fly_time > 1.0) {
      return result;
    }
    result.pitch = -trajectory.pitch + params.pitch_offset;
    result.fly_time = trajectory.fly_time;
    // 飞行时间迭代、正对程度和俯仰范围都通过，才允许进入射击判断。
    result.fireable = std::abs(trajectory.fly_time - flight) < 0.002 &&
                      facing < 45.0 * DEG && std::abs(result.pitch) <= PITCH_LIMIT;
    flight = trajectory.fly_time;
    previous_id = chosen;
  }
  return result;
}
}  // namespace

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

  const auto yaml = tools::load(config_path);
  const auto enemy_color = tools::read<std::string>(yaml, "enemy_color");
  const Parameters params{
    enemy_color == "red" ? auto_aim::red : auto_aim::blue,
    tools::read_or<double>(yaml, "yaw_offset", 0.0) * DEG,
    tools::read_or<double>(yaml, "pitch_offset", 0.0) * DEG,
    tools::read_or<double>(yaml, "task2_execution_delay_ms", 15.0) / 1000.0};
  if (
    (enemy_color != "red" && enemy_color != "blue") || !std::isfinite(params.yaw_offset) ||
    !std::isfinite(params.pitch_offset) || !std::isfinite(params.execution_delay) ||
    params.execution_delay < 0 || params.execution_delay > 0.2) {
    tools::logger()->error("[Task2] Invalid color, angle offset or execution delay.");
    return 1;
  }

  // 初始化工具类
  tools::Exiter exiter;
  tools::Plotter plotter;  // 注意plotter工具的使用

  // 初始化io类
  io::Camera camera(config_path);
  io::Gimbal gimbal(config_path);

  // 初始化auto_aim类
  auto_aim::YOLO yolo(config_path, true);
  auto_aim::Solver solver(config_path);

  cv::Mat img;
  Eigen::Quaterniond q;
  std::chrono::steady_clock::time_point t;

  std::optional<auto_aim::Target> target;
  ShotController shots;
  int frame_count = 0;
  int consecutive_hits = 0;
  int observation_count = 0;
  int aim_id = -1;
  const auto start_time = Clock::now();
  auto model_time = start_time;
  auto last_seen = start_time;
  auto track_start = start_time;
  auto last_command_time = start_time;
  bool was_controlling = false;
  float last_yaw = 0, last_pitch = 0;

  const auto reset_tracking = [&] {
    target.reset();
    consecutive_hits = observation_count = 0;
    aim_id = -1;
    shots.permitted = false;  // 丢失整个目标需要重新授权，但不能重置已用预算。
  };

  tools::logger()->info("[Task2] 1: low, 2: medium, a: authorize, SPACE: pause, Esc/q: exit.");

  while (!exiter.exit()) {
    // Your code start

    bool observed = false;
    bool quit = false;
    if (gimbal.mode() != io::GimbalMode::AUTO_AIM || Clock::now() - last_seen > 500ms) {
      reset_tracking();
    }

    try {
      const bool got_frame = camera.try_read_for(img, t, 100ms);
      const auto frame_age = Clock::now() - t;
      if (got_frame && !img.empty() && frame_age >= 0ms && frame_age <= 250ms) {
        q = gimbal.q(t);  // 一帧只取一次姿态，与该帧时间戳对齐。
        if (q.coeffs().allFinite() && q.norm() > 1e-6) {
          solver.set_R_gimbal2world(q.normalized());
          auto detections = yolo.detect(img, frame_count++);
          std::vector<auto_aim::Armor *> armors;
          for (auto & armor : detections) {
            if (!valid_detection(armor, params, img.size())) continue;
            armor.priority = auto_aim::first;
            try {
              solver.solve(armor);
            } catch (const cv::Exception &) {
              continue;
            }
            if (
              armor.xyz_in_world.allFinite() && armor.xyz_in_gimbal.allFinite() &&
              armor.ypr_in_world.allFinite() && armor.ypd_in_world.allFinite() &&
              armor.xyz_in_gimbal.x() > 0 && armor.ypd_in_world[2] > 0.05) {
              armors.push_back(&armor);  // 只保存观测地址，不复制 Armor 中尚未使用的成员。
            }
          }

          // 每个图像时间戳只预测一次，更新时只使用同一靶机的一块最佳观测。
          const bool forward_frame = !target || t > model_time;
          if (target && forward_frame) {
            target->predict(t);
            model_time = t;
            if (!healthy(*target)) reset_tracking();
          }
          if (forward_frame && gimbal.mode() == io::GimbalMode::AUTO_AIM) {
            auto_aim::Armor * selected = nullptr;
            double best = std::numeric_limits<double>::infinity();
            for (const auto armor : armors) {
              const double dx = (armor->center.x - img.cols * 0.5) / img.cols;
              const double dy = (armor->center.y - img.rows * 0.5) / img.rows;
              const double score = target ? association_score(*armor, *target) : dx * dx + dy * dy;
              if (score < best) {
                best = score;
                selected = armor;
              }
            }
            if (selected != nullptr) {
              if (!target) {
                target.emplace(*selected, t, 0.2, 4);
                model_time = t;
                track_start = Clock::now();
              } else {
                target->update(*selected);
              }
              observed = healthy(*target);
              if (observed) {
                last_seen = t;
                ++consecutive_hits;
                ++observation_count;
                tools::draw_points(img, selected->points, {0, 255, 255});
              } else {
                reset_tracking();
              }
            }
          }

          tools::draw_text(
            img, "1:LOW 2:MID a:ARM SPACE:PAUSE q:EXIT", {10, 30}, {0, 255, 255}, 0.7);
          tools::draw_text(
            img, "stage=" + std::to_string(shots.stage) + " low=" +
                   std::to_string(shots.requests[0]) + "/10 mid=" +
                   std::to_string(shots.requests[1]) + "/10 " +
                   (shots.permitted ? "AUTHORIZED" : "PAUSED"),
            {10, 60}, {0, 255, 255}, 0.7);
          cv::Mat preview;
          cv::resize(img, preview, {}, 0.5, 0.5);
          cv::imshow("task_2", preview);
        }
      }
      const int key = cv::waitKey(1) & 0xff;
      quit = key == 27 || key == 'q';
      shots.handle_key(key);
    } catch (const std::exception & e) {
      observed = false;
      shots.permitted = false;
      tools::logger()->warn("[Task2] Frame processing failed: {}", e.what());
    }

    const auto now = Clock::now();
    const bool auto_aim = gimbal.mode() == io::GimbalMode::AUTO_AIM;
    if (!auto_aim || now - last_seen > 500ms) reset_tracking();
    if (now - t > 250ms) observed = false;
    if (!observed) consecutive_hits = 0;
    const auto state = gimbal.state();  // 发送前取最新反馈，用于限速和瞄准误差判断。
    const bool state_valid = std::isfinite(state.yaw) && std::isfinite(state.pitch);
    AimSolution aim;
    if (target && healthy(*target) && now - last_seen <= 150ms) {
      aim = predict_aim(*target, model_time, now, state.bullet_speed, aim_id, params);
    }
    const bool control = auto_aim && state_valid && aim.valid && !quit && !exiter.exit();
    float yaw_cmd = state_valid ? state.yaw : 0;
    float pitch_cmd = state_valid ? state.pitch : 0;
    if (control) {
      // 与任务一相同：绝对角、弧度、抬头为正、yaw 支持连续角；现场核对协议。
      const double dt = std::clamp(tools::delta_time(now, last_command_time), 0.0, 0.1);
      const double base_yaw = was_controlling ? last_yaw : state.yaw;
      const double base_pitch = was_controlling ? last_pitch : state.pitch;
      yaw_cmd = base_yaw + std::clamp(
        tools::limit_rad(aim.yaw - base_yaw), -120.0 * DEG * dt, 120.0 * DEG * dt);
      pitch_cmd = std::clamp(
        base_pitch + std::clamp(aim.pitch - base_pitch, -90.0 * DEG * dt, 90.0 * DEG * dt),
        -PITCH_LIMIT, PITCH_LIMIT);
      aim_id = aim.id;
    }

    // 以下为射击调试条件，不是题面额外规定的考核阈值。
    const bool model_ready = target && target->jumped && observation_count >= 30 &&
                             consecutive_hits >= 10 && now - track_start >= 700ms &&
                             target->ekf().P(7, 7) < 1.0;
    const double omega = target ? target->ekf_x()[7] : 0;
    const bool speed_matches_stage =
      (shots.stage == 1 && std::abs(omega) >= 2.4 && std::abs(omega) <= 5.6) ||
      (shots.stage == 2 && std::abs(omega) >= 4.0 && std::abs(omega) <= 10.0);
    const bool aligned = state_valid &&
                         std::abs(tools::limit_rad(aim.yaw - state.yaw)) < aim.yaw_tolerance &&
                         std::abs(aim.pitch - state.pitch) < aim.pitch_tolerance;
    // 用期望瞄准角判断是否对准，不能用限速后仍接近当前反馈的命令冒充对准。
    const bool fire = shots.ready(
      now, control && observed && model_ready && aim.fireable && aligned && speed_matches_stage);
    gimbal.send(control, fire, yaw_cmd, pitch_cmd);
    if (fire) shots.record_request(Clock::now());  // 计请求，不把它冒充实际出弹反馈。

    nlohmann::json data{
      {"time_s", tools::delta_time(now, start_time)},
      {"yaw_cmd_rad", yaw_cmd}, {"pitch_cmd_rad", pitch_cmd},
      {"control_cmd", control}, {"fire_cmd", fire},
      {"tracking_valid", observed}, {"model_ready", model_ready},
      {"stage", shots.stage}, {"fire_permitted", shots.permitted},
      {"low_shot_requests", shots.requests[0]}, {"medium_shot_requests", shots.requests[1]},
      {"bullet_count_raw", state.bullet_count}, {"aim_armor_id", aim.id},
      {"aim_fireable", aim.fireable}};
    if (target && healthy(*target)) {
      data["omega_ekf_rad_s"] = omega;
      data["omega_abs_rad_s"] = std::abs(omega);
      data["observed_armor_id"] = target->last_id;
    }
    if (aim.valid) {
      data["flight_time_s"] = aim.fly_time;
      data["prediction_horizon_s"] = aim.horizon;
    }
    plotter.plot(data);  // 与本次 send 的最终浮点实参完全一致。
    was_controlling = control;
    last_yaw = yaw_cmd;
    last_pitch = pitch_cmd;
    last_command_time = now;
    if (quit) break;

    // Your code end
  }

  gimbal.send(false, false, 0, 0);
  plotter.plot({
    {"time_s", tools::delta_time(Clock::now(), start_time)},
    {"yaw_cmd_rad", 0}, {"pitch_cmd_rad", 0}, {"control_cmd", false}, {"fire_cmd", false},
    {"tracking_valid", false}, {"stage", shots.stage},
    {"low_shot_requests", shots.requests[0]}, {"medium_shot_requests", shots.requests[1]}});
  cv::destroyAllWindows();

  return 0;
}
