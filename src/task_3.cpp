#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <limits>
#include <opencv2/opencv.hpp>
#include <optional>

#include "io/camera.hpp"
#include "io/gimbal/gimbal.hpp"
#include "tasks/auto_aim/outpost_aim.hpp"
#include "tasks/auto_aim/solver.hpp"
#include "tasks/auto_aim/yolo.hpp"
#include "tools/exiter.hpp"
#include "tools/img_tools.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"
#include "tools/plotter.hpp"
#include "tools/yaml.hpp"

namespace
{
namespace outpost = auto_aim::outpost_aim;
using outpost::Clock;
using outpost::DEG;

bool valid_detection(const auto_aim::Armor & armor, auto_aim::Color color, cv::Size size)
{
  if (
    armor.color != color || armor.name != auto_aim::outpost || armor.points.size() != 4 ||
    !std::isfinite(armor.confidence)) return false;
  const bool finite_points = std::all_of(
    armor.points.begin(), armor.points.end(), [&](const cv::Point2f & point) {
      return std::isfinite(point.x) && std::isfinite(point.y) && point.x >= 0 &&
             point.x < size.width && point.y >= 0 && point.y < size.height;
    });
  return finite_points && cv::isContourConvex(armor.points) && cv::contourArea(armor.points) >= 4;
}

void draw_model(cv::Mat & img, const auto_aim::Target & target, const auto_aim::Solver & solver)
{
  const auto points = target.armor_xyza_list();
  for (int id = 0; id < 3; ++id) {
    const Eigen::Vector3d xyz = points[id].head<3>();
    const auto pixel = solver.world2pixel(xyz);
    if (!pixel || !std::isfinite(pixel->x) || !std::isfinite(pixel->y) || pixel->x < 0 ||
        pixel->x >= img.cols || pixel->y < 0 || pixel->y >= img.rows) continue;
    const cv::Scalar color = id == target.last_id ? cv::Scalar(0, 255, 0) : cv::Scalar(255, 180, 0);
    const auto corners = solver.reproject_armor(
      xyz, points[id][3], auto_aim::special, auto_aim::outpost);
    if (std::all_of(corners.begin(), corners.end(), [&](const cv::Point2f & p) {
          return std::isfinite(p.x) && std::isfinite(p.y) && p.x >= 0 && p.x < img.cols &&
                 p.y >= 0 && p.y < img.rows;
        })) tools::draw_points(img, corners, color);
    tools::draw_text(img, std::to_string(id), *pixel, color, 0.7);
  }
}
}  // namespace

const std::string keys =
  "{help h usage ? | | 输出命令行参数说明}"
  "{@config-path   | | yaml配置文件路径 }";

using namespace std::chrono_literals;

int main(int argc, char * argv[])
{
  cv::CommandLineParser cli(argc, argv, keys);
  if (cli.has("help") || !cli.has("@config-path")) {
    cli.printMessage();
    return 0;
  }
  const auto config_path = cli.get<std::string>("@config-path");
  const auto yaml = tools::load(config_path);
  const auto color_name = tools::read<std::string>(yaml, "enemy_color");
  const auto color = color_name == "red" ? auto_aim::red : auto_aim::blue;
  const outpost::Parameters params{
    tools::read_or<double>(yaml, "yaw_offset", 0.0) * DEG,
    tools::read_or<double>(yaml, "pitch_offset", 0.0) * DEG,
    tools::read_or<double>(yaml, "task3_execution_delay_ms", 15.0) / 1000.0,
    tools::read_or<double>(yaml, "task3_pitch_limit_deg", 20.0) * DEG};
  const auto fire_mode = tools::read_or<std::string>(yaml, "task3_fire_mode", "disabled");
  outpost::FireController shots;
  shots.external_gate_confirmed =
    tools::read_or<bool>(yaml, "task3_external_fire_gate_confirmed", false);
  if (fire_mode == "continuous") shots.mode = outpost::FireMode::CONTINUOUS;
  else if (fire_mode == "single") shots.mode = outpost::FireMode::SINGLE;
  else if (fire_mode != "disabled") {
    tools::logger()->error("[Task3] task3_fire_mode must be disabled, continuous or single.");
    return 1;
  }
  if (
    (color_name != "red" && color_name != "blue") || !std::isfinite(params.yaw_offset) ||
    !std::isfinite(params.pitch_offset) || !std::isfinite(params.execution_delay) ||
    params.execution_delay < 0 || params.execution_delay > 0.2 ||
    !std::isfinite(params.pitch_limit) || params.pitch_limit <= 0 || params.pitch_limit > 45 * DEG ||
    (shots.mode != outpost::FireMode::DISABLED && !shots.external_gate_confirmed)) {
    tools::logger()->error("[Task3] Invalid parameters or external fire gate is not confirmed.");
    return 1;
  }

  tools::Exiter exiter;
  tools::Plotter plotter;
  io::Camera camera(config_path);
  io::Gimbal gimbal(config_path);
  auto_aim::YOLO yolo(config_path, true);
  auto_aim::Solver solver(config_path);

  cv::Mat img;
  auto t = Clock::now();
  std::optional<auto_aim::Target> target;
  std::array<int, 3> board_observations{};
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
    board_observations.fill(0);
    consecutive_hits = observation_count = 0;
    aim_id = -1;
  };
  tools::logger()->info(
    "[Task3] fire={}, external_gate={}. SPACE: pause, r: resume, Esc/q: exit. "
    "The external switch controls the official 30 s firing interval.",
    fire_mode, shots.external_gate_confirmed);

  while (!exiter.exit()) {
    bool observed = false;
    bool quit = false;
    if (gimbal.mode() != io::GimbalMode::AUTO_AIM || Clock::now() - last_seen > 500ms) {
      reset_tracking();
    }
    try {
      const bool got_frame = camera.try_read_for(img, t, 100ms);
      const auto frame_age = Clock::now() - t;
      if (got_frame && !img.empty() && frame_age >= 0ms && frame_age <= 250ms) {
        const auto q = gimbal.q(t);
        if (q.coeffs().allFinite() && q.norm() > 1e-6) {
          solver.set_R_gimbal2world(q.normalized());
          auto detections = yolo.detect(img, frame_count++);
          std::vector<auto_aim::Armor *> armors;
          for (auto & armor : detections) {
            if (!valid_detection(armor, color, img.size())) continue;
            // YOLOv5 默认将 outpost 标成 small；仅此任务修正为 Solver 的 129 mm 特殊板。
            armor.type = auto_aim::special;
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
              armors.push_back(&armor);
            }
          }

          const bool forward_frame = !target || t > model_time;
          if (target && forward_frame) {
            target->predict(t);
            model_time = t;
            if (!outpost::healthy(*target)) reset_tracking();
          }
          if (forward_frame && gimbal.mode() == io::GimbalMode::AUTO_AIM) {
            auto_aim::Armor * selected = nullptr;
            double best = std::numeric_limits<double>::infinity();
            for (auto armor : armors) {
              const double dx = (armor->center.x - img.cols * 0.5) / img.cols;
              const double dy = (armor->center.y - img.rows * 0.5) / img.rows;
              const double score = target ? outpost::association_score(*armor, *target) :
                                            dx * dx + dy * dy;
              if (score < best) {
                best = score;
                selected = armor;
              }
            }
            if (selected) {
              if (!target) {
                target.emplace(*selected, t, 0.2765, 3);
                model_time = t;
                track_start = Clock::now();
              } else {
                target->update(*selected);
              }
              observed = outpost::healthy(*target);
              if (observed) {
                last_seen = t;
                ++consecutive_hits;
                ++observation_count;
                ++board_observations[target->last_id];
                tools::draw_points(img, selected->points, {0, 255, 255});
              } else {
                reset_tracking();
              }
            }
          }
          if (target && outpost::healthy(*target)) draw_model(img, *target, solver);
          tools::draw_text(
            img, "OUTPOST fire=" + fire_mode + (shots.paused ? " PAUSED" : "") +
                   " SPACE:pause r:resume q:exit", {10, 30}, {0, 255, 255}, 0.7);
          tools::draw_text(
            img, "boards=" + std::to_string(board_observations[0]) + "/" +
                   std::to_string(board_observations[1]) + "/" +
                   std::to_string(board_observations[2]) + " layers=" +
                   (target && outpost::layers_ready(*target) ? "READY" : "LEARNING"),
            {10, 60}, {0, 255, 255}, 0.7);
          cv::Mat preview;
          cv::resize(img, preview, {}, 0.5, 0.5);
          cv::imshow("task_3", preview);
        }
      }
      const int key = cv::waitKey(1) & 0xff;
      quit = key == 27 || key == 'q';
      if (key == ' ') shots.paused = true;
      if (key == 'r') shots.paused = false;
    } catch (const std::exception & e) {
      observed = false;
      shots.paused = true;
      tools::logger()->warn("[Task3] Frame processing failed: {}", e.what());
    }

    const auto now = Clock::now();
    const bool auto_aim = gimbal.mode() == io::GimbalMode::AUTO_AIM;
    if (!auto_aim || now - last_seen > 500ms) reset_tracking();
    if (now - t > 250ms) observed = false;
    if (!observed) consecutive_hits = 0;
    const auto state = gimbal.state();
    const bool state_valid = std::isfinite(state.yaw) && std::isfinite(state.pitch);
    outpost::AimSolution aim;
    if (target && outpost::healthy(*target) && now - last_seen <= 150ms) {
      aim = outpost::predict_aim(*target, model_time, now, state.bullet_speed, aim_id, params);
    }
    const bool control = auto_aim && state_valid && aim.valid && !quit && !exiter.exit();
    float yaw_cmd = state_valid ? state.yaw : 0;
    float pitch_cmd = state_valid ? state.pitch : 0;
    if (control) {
      // 延续前两任务的绝对角/弧度约定，yaw 用周期差限速，pitch 按哨兵配置限位。
      const double dt = std::clamp(tools::delta_time(now, last_command_time), 0.0, 0.1);
      const double base_yaw = was_controlling ? last_yaw : state.yaw;
      const double base_pitch = was_controlling ? last_pitch : state.pitch;
      yaw_cmd = base_yaw + std::clamp(
        tools::limit_rad(aim.yaw - base_yaw), -120.0 * DEG * dt, 120.0 * DEG * dt);
      pitch_cmd = std::clamp(
        base_pitch + std::clamp(aim.pitch - base_pitch, -90.0 * DEG * dt, 90.0 * DEG * dt),
        -params.pitch_limit, params.pitch_limit);
      aim_id = aim.id;
    }

    // 以下为调试起点，非题面新增要求。三块板都要被实际观测过，且层高、转速稳定。
    const bool model_ready = target && outpost::layers_ready(*target) &&
      *std::min_element(board_observations.begin(), board_observations.end()) >= 5 &&
      observation_count >= 60 && consecutive_hits >= 10 && now - track_start >= 1500ms &&
      target->ekf().P(7, 7) < 0.16 && target->ekf().last_nis < 9.49;
    const bool aligned = state_valid &&
      std::abs(tools::limit_rad(aim.yaw - state.yaw)) < aim.yaw_tolerance &&
      std::abs(aim.pitch - state.pitch) < aim.pitch_tolerance;
    const bool fire = shots.ready(
      now, control && observed && model_ready && aim.fireable && aligned);
    gimbal.send(control, fire, yaw_cmd, pitch_cmd);
    if (fire) shots.record_request(Clock::now());

    nlohmann::json data{
      {"time_s", tools::delta_time(now, start_time)},
      {"yaw_cmd_rad", yaw_cmd}, {"pitch_cmd_rad", pitch_cmd},
      {"control_cmd", control}, {"fire_cmd", fire},
      {"tracking_valid", observed}, {"model_ready", model_ready},
      {"fire_mode", fire_mode}, {"fire_paused", shots.paused},
      {"external_fire_gate_confirmed", shots.external_gate_confirmed},
      {"fire_request_packets", shots.request_packets}, {"bullet_count_raw", state.bullet_count},
      {"aim_armor_id", aim.id}, {"aim_fireable", aim.fireable},
      {"board0_observations", board_observations[0]},
      {"board1_observations", board_observations[1]},
      {"board2_observations", board_observations[2]}};
    if (target && outpost::healthy(*target)) {
      const auto x = target->ekf_x();
      data["omega_ekf_rad_s"] = x[7];
      data["omega_abs_rad_s"] = std::abs(x[7]);
      data["omega_variance"] = target->ekf().P(7, 7);
      data["observed_armor_id"] = target->last_id;
      data["radius_m"] = x[8];
      data["height0_m"] = x[4];
      data["height1_m"] = x[4] + x[9];
      data["height2_m"] = x[4] + x[10];
      data["nis"] = observation_count > 1 ? target->ekf().last_nis : 0;
    }
    if (aim.valid) {
      data["flight_time_s"] = aim.fly_time;
      data["prediction_horizon_s"] = aim.horizon;
      data["aim_height_m"] = aim.height;
    }
    plotter.plot(data);  // 输出的控制命令与本次 send 的最终浮点实参一致。
    was_controlling = control;
    last_yaw = yaw_cmd;
    last_pitch = pitch_cmd;
    last_command_time = now;
    if (quit) break;
  }

  gimbal.send(false, false, 0, 0);
  plotter.plot({
    {"time_s", tools::delta_time(Clock::now(), start_time)},
    {"yaw_cmd_rad", 0}, {"pitch_cmd_rad", 0}, {"control_cmd", false}, {"fire_cmd", false},
    {"tracking_valid", false}, {"model_ready", false},
    {"fire_request_packets", shots.request_packets}});
  cv::destroyAllWindows();
  return 0;
}
