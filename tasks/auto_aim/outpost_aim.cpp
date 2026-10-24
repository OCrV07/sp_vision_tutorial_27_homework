#include "outpost_aim.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

#include "tools/math_tools.hpp"
#include "tools/trajectory.hpp"

namespace auto_aim::outpost_aim
{
bool healthy(const Target & target)
{
  const auto x = target.ekf_x();
  return target.name == ArmorName::outpost && target.armor_xyza_list().size() == 3 &&
         !target.diverged() && std::abs(x[7]) < 15 &&
         Eigen::Vector3d(x[1], x[3], x[5]).norm() < 5;
}

bool layers_ready(const Target & target)
{
  if (!healthy(target)) return false;
  const auto x = target.ekf_x();
  std::array<double, 3> heights{0.0, x[9], x[10]};
  std::sort(heights.begin(), heights.end());
  // 不预设初始板是低/中/高中的哪一块，也不预设板序或转向。
  return std::abs(heights[1] - heights[0] - 0.100) < 0.035 &&
         std::abs(heights[2] - heights[1] - 0.100) < 0.035 &&
         target.ekf().P(9, 9) < 0.03 * 0.03 && target.ekf().P(10, 10) < 0.03 * 0.03;
}

double association_score(const Armor & armor, const Target & target)
{
  double best = std::numeric_limits<double>::infinity();
  if (armor.name != ArmorName::outpost || armor.type != target.armor_type) return best;
  const auto candidates = target.armor_xyza_list();
  for (int id = 0; id < 3; ++id) {
    const auto & point = candidates[id];
    const double horizontal = (armor.xyz_in_world.head<2>() - point.head<2>()).norm();
    const double angle = std::abs(tools::limit_rad(armor.ypr_in_world[0] - point[3]));
    double z_variance = target.ekf().P(4, 4);
    if (id > 0) {
      const int height_index = id == 1 ? 9 : 10;
      z_variance += target.ekf().P(height_index, height_index) +
                    2 * target.ekf().P(4, height_index);
    }
    // 未知层高初期保留 0.2 m 的学习空间；收敛后收紧，拒绝不符模型的顶部板。
    const double z_gate = std::clamp(3 * std::sqrt(std::max(0.0, z_variance)), 0.08, 0.30);
    const double z_error = std::abs(armor.xyz_in_world[2] - point[2]);
    if (horizontal < 0.45 && angle < 60.0 * DEG && z_error < z_gate) {
      best = std::min(best, horizontal / 0.45 + angle / (60.0 * DEG) + z_error / z_gate);
    }
  }
  return best;
}

AimSolution predict_aim(
  const Target & target, TimePoint model_time, TimePoint now, double bullet_speed,
  int previous_id, const Parameters & params)
{
  AimSolution result;
  const auto state = target.ekf_x();
  const double age = tools::delta_time(now, model_time);
  if (!healthy(target) || age < 0 || age > 0.5) return result;

  const bool speed_valid = std::isfinite(bullet_speed) && bullet_speed >= 5 && bullet_speed <= 40;
  double flight = speed_valid ? std::hypot(state[0], state[2]) / bullet_speed : 0;
  for (int iteration = 0; iteration < 4; ++iteration) {
    result = {};
    const double horizon = age + params.execution_delay + flight;
    if (!std::isfinite(horizon) || horizon < 0 || horizon > 1) return {};
    auto future = target;
    future.predict(horizon);
    if (!healthy(future)) return {};
    const auto candidates = future.armor_xyza_list();

    int chosen = -1;
    double best = std::numeric_limits<double>::infinity();
    double facing = 0;
    for (int id = 0; id < 3; ++id) {
      const auto & point = candidates[id];
      if (!point.allFinite()) continue;
      const double bearing = std::atan2(point[1], point[0]);
      const double error = std::abs(tools::limit_rad(point[3] - bearing));
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
    result.height = point[2];
    result.yaw = std::atan2(point[1], point[0]) + params.yaw_offset;
    result.pitch = std::atan2(point[2], distance) + params.pitch_offset;
    result.horizon = horizon;
    result.yaw_tolerance = std::atan2(0.129 * std::cos(facing) / 4.0, distance);
    result.pitch_tolerance = std::atan2(0.056 / 4.0, distance);
    if (!speed_valid) return result;

    const tools::Trajectory trajectory(bullet_speed, distance, point[2]);
    if (
      trajectory.unsolvable || !std::isfinite(trajectory.pitch) ||
      !std::isfinite(trajectory.fly_time) || trajectory.fly_time <= 0 || trajectory.fly_time > 1) {
      return result;
    }
    result.pitch = trajectory.pitch + params.pitch_offset;
    result.fly_time = trajectory.fly_time;
    result.fireable = std::abs(trajectory.fly_time - flight) < 0.002 && facing < 45.0 * DEG &&
                      std::abs(result.pitch) <= params.pitch_limit;
    flight = trajectory.fly_time;
    previous_id = chosen;
  }
  return result;
}

bool FireController::ready(TimePoint now, bool aim_ready) const
{
  if (mode == FireMode::DISABLED || !external_gate_confirmed || paused || !aim_ready) return false;
  return mode == FireMode::CONTINUOUS || !last_request ||
         std::chrono::duration<double>(now - *last_request).count() >= 1.0 / 12.0;
}

void FireController::record_request(TimePoint now)
{
  ++request_packets;  // 这是 fire=true 的命令包数量，不能冒充实际出弹数。
  last_request = now;
}
}  // namespace auto_aim::outpost_aim
