#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>

#include "tasks/auto_aim/outpost_aim.hpp"
#include "tools/math_tools.hpp"

namespace auto_aim
{
// 数值微分直接检查生产代码的完整观测雅可比，而不是更新时人为解耦的车模矩阵。
struct TargetModelTestAccess
{
  static Eigen::VectorXd & state(Target & target) { return target.ekf_.x; }
  static Eigen::MatrixXd & covariance(Target & target) { return target.ekf_.P; }
  static Eigen::MatrixXd jacobian(const Target & target, const Eigen::VectorXd & x, int id)
  {
    return target.h_jacobian(x, id);
  }
  static Eigen::Vector4d measurement(const Target & target, const Eigen::VectorXd & x, int id)
  {
    const Eigen::Vector3d ypd = tools::xyz2ypd(target.h_armor_xyz(x, id));
    return {ypd[0], ypd[1], ypd[2], tools::limit_rad(x[6] + id * 2 * CV_PI / target.armor_num_)};
  }
};
}  // namespace auto_aim

namespace
{
namespace aim = auto_aim::outpost_aim;
using Access = auto_aim::TargetModelTestAccess;
using namespace std::chrono_literals;

void require(bool condition, const std::string & message)
{
  if (!condition) throw std::runtime_error(message);
}

auto_aim::Armor observation(const Eigen::Vector3d & xyz, double angle)
{
  auto_aim::Armor armor(0, 6, 0.99, cv::Rect(0, 0, 100, 50),
    {{0, 0}, {100, 0}, {100, 50}, {0, 50}});
  armor.type = auto_aim::special;
  armor.priority = auto_aim::first;
  armor.xyz_in_world = xyz;
  armor.xyz_in_gimbal = xyz;
  armor.ypd_in_world = tools::xyz2ypd(xyz);
  armor.ypr_in_world = {angle, 0, 0};
  return armor;
}

void geometry_and_jacobian()
{
  const auto t = aim::Clock::now();
  auto armor = observation({2.7235, 0, 0.2}, 0);
  auto_aim::Target target(armor, t);
  require(target.armor_xyza_list().size() == 3, "outpost must default to three plates");
  require(std::abs(target.ekf_x()[8] - 0.2765) < 1e-12, "outpost default radius");
  require(!target.diverged() && !aim::layers_ready(target), "zero layer offsets are a valid initial state");

  auto & x = Access::state(target);
  x[9] = -0.1;
  x[10] = 0.1;
  const auto points = target.armor_xyza_list();
  for (int id = 0; id < 3; ++id) {
    require(std::abs(std::hypot(points[id][0] - 3, points[id][1]) - 0.2765) < 1e-10,
      "all three plates must share a radius");
    const std::array<double, 3> expected_z{0.2, 0.1, 0.3};
    require(std::abs(points[id][2] - expected_z[id]) < 1e-12, "plate-specific signed height");
    require(std::abs(tools::limit_rad(points[(id + 1) % 3][3] - points[id][3]) - 2 * CV_PI / 3) < 1e-10,
      "120 degree spacing");
  }
  require(!target.diverged(), "negative height difference must not be treated as a second radius");

  double worst = 0;
  for (const bool vehicle : {false, true}) {
    armor.name = vehicle ? auto_aim::three : auto_aim::outpost;
    auto_aim::Target model(armor, t);
    auto & state = Access::state(model);
    state[0] = 3.1;
    state[2] = -0.6;
    state[4] = 0.4;
    state[9] = vehicle ? 0.06 : -0.1;
    state[10] = vehicle ? 0.08 : 0.1;
    for (const double phase : {-3.1415925, -0.8, 0.4, 3.1415925}) {
      state[6] = phase;
      for (int id = 0; id < (vehicle ? 4 : 3); ++id) {
        const auto analytic = Access::jacobian(model, state, id);
        Eigen::MatrixXd numerical(4, 11);
        for (int column = 0; column < 11; ++column) {
          Eigen::VectorXd plus = state, minus = state;
          constexpr double epsilon = 1e-6;
          plus[column] += epsilon;
          minus[column] -= epsilon;
          Eigen::Vector4d difference = Access::measurement(model, plus, id) -
                                       Access::measurement(model, minus, id);
          difference[0] = tools::limit_rad(difference[0]);
          difference[3] = tools::limit_rad(difference[3]);
          numerical.col(column) = difference / (2 * epsilon);
        }
        worst = std::max(worst, (analytic - numerical).cwiseAbs().maxCoeff());
      }
    }
  }
  require(worst < 1e-6, "analytic Jacobian differs from numerical derivative");
  x[9] = 0.4;
  require(target.diverged(), "impossible layer separation must diverge");
  x[9] = 0;
  x[7] = std::numeric_limits<double>::quiet_NaN();
  require(target.diverged(), "non-finite state must diverge");
  std::cout << "geometry/Jacobian: max error=" << worst << '\n';
}

// 独立生成实际装甲板观测：只观察最正对相机的一块，加入位置/角度噪声和掉帧。
void rotating_streams()
{
  double worst_speed_error = 0;
  double worst_height_error = 0;
  int cases = 0;
  std::array<double, 3> heights{0.2, 0.3, 0.4};
  do {
    for (double omega : {-2.5, 2.5}) {
      for (const int fps : {30, 80}) {
        std::mt19937 random(42 + cases);
        std::normal_distribution<double> position_noise(0, 0.003);
        std::normal_distribution<double> angle_noise(0, 0.025);
        std::uniform_real_distribution<double> missing(0, 1);
        const auto start = aim::Clock::now();
        const auto make_observation = [&](double seconds, int & observed_id) {
          double facing = 10;
          Eigen::Vector3d xyz;
          double angle = 0;
          for (int id = 0; id < 3; ++id) {
            const double theta = tools::limit_rad(0.1 + omega * seconds + id * 2 * CV_PI / 3);
            const Eigen::Vector3d point(3.2 - 0.2765 * std::cos(theta),
              0.15 - 0.2765 * std::sin(theta), heights[id]);
            const double error = std::abs(tools::limit_rad(theta - std::atan2(point[1], point[0])));
            if (error < facing) {
              facing = error;
              xyz = point;
              angle = theta;
              observed_id = id;
            }
          }
          for (int axis = 0; axis < 3; ++axis) xyz[axis] += position_noise(random);
          return observation(xyz, tools::limit_rad(angle + angle_noise(random)));
        };
        int physical_id = -1;
        auto first = make_observation(0, physical_id);
        require(physical_id == 0, "synthetic initial plate");
        auto_aim::Target target(first, start);
        std::array<int, 3> counts{1, 0, 0};
        int accepted = 0;
        int available = 0;
        int ready_frames = 0;
        for (int frame = 1; frame <= fps * 12; ++frame) {
          const double seconds = static_cast<double>(frame) / fps;
          const auto t = start + std::chrono::duration_cast<aim::Clock::duration>(
            std::chrono::duration<double>(seconds));
          target.predict(t);
          require(aim::healthy(target), "stream prediction diverged");
          if (missing(random) < 0.08 || (seconds > 5 && seconds < 5.15)) continue;
          auto armor = make_observation(seconds, physical_id);
          ++available;
          if (!std::isfinite(aim::association_score(armor, target))) continue;
          target.update(armor);
          ++accepted;
          ++counts[target.last_id];
          require(aim::healthy(target), "stream update diverged");
          require(target.last_id == physical_id, "board identity changed during rotation");
          if (seconds >= 3) {
            worst_speed_error = std::max(worst_speed_error, std::abs(target.ekf_x()[7] - omega));
            if (aim::layers_ready(target) && target.ekf().P(7, 7) < 0.16) ++ready_frames;
          }
        }
        const auto x = target.ekf_x();
        std::cout << "stream " << cases << ": w=" << x[7] << " heights=" << x[4] << ','
          << x[4] + x[9] << ',' << x[4] + x[10] << " accepted=" << accepted << '/'
          << available << " ready=" << ready_frames << '\n';
        require(accepted > available * 0.95, "association rejects valid rotating plates");
        require(*std::min_element(counts.begin(), counts.end()) > 5, "all plates must be observed");
        require(ready_frames > fps * 5, "layer/speed uncertainty did not converge");
        for (int id = 0; id < 3; ++id) {
          const double predicted_z = x[4] + (id == 0 ? 0 : x[id == 1 ? 9 : 10]);
          worst_height_error = std::max(worst_height_error, std::abs(predicted_z - heights[id]));
        }
        require(aim::layers_ready(target), "final layers not ready");
        auto top = observation({3.2, 0.15, 1.2}, 0);
        require(!std::isfinite(aim::association_score(top, target)), "unrelated top plate must be rejected");
        ++cases;
      }
    }
  } while (std::next_permutation(heights.begin(), heights.end()));
  require(worst_speed_error < 0.4, "synthetic speed error exceeds 0.4 rad/s");
  require(worst_height_error < 0.025, "synthetic height error exceeds 25 mm");
  std::cout << cases << " streams: max speed error=" << worst_speed_error
            << " rad/s, final height error=" << worst_height_error << " m\n";
}

void prediction_and_fire()
{
  const auto t = aim::Clock::now();
  auto armor = observation({2.7235, 0, 0.2}, 0);
  auto_aim::Target target(armor, t);
  auto & x = Access::state(target);
  x[7] = 2.5;
  x[9] = 0.1;
  x[10] = -0.1;
  aim::Parameters params;
  const Eigen::VectorXd before = x;
  const Eigen::MatrixXd covariance_before = target.ekf().P;
  const auto solution = aim::predict_aim(target, t, t + 20ms, 25, -1, params);
  require(solution.valid && solution.fireable, "normal ballistic prediction must work");
  require((target.ekf_x() - before).norm() == 0 &&
    (target.ekf().P - covariance_before).norm() == 0, "future prediction must not mutate tracking state");
  auto future = target;
  future.predict(solution.horizon);
  const auto point = future.armor_xyza_list()[solution.id];
  require(std::abs(solution.height - point[2]) < 1e-12, "aim must use selected plate height");
  const double distance = std::hypot(point[0], point[1]);
  require(std::abs(25 * std::cos(solution.pitch) * solution.fly_time - distance) < 1e-6,
    "ballistic horizontal displacement");
  require(std::abs(25 * std::sin(solution.pitch) * solution.fly_time -
    0.5 * 9.794 * solution.fly_time * solution.fly_time - point[2]) < 1e-6,
    "ballistic vertical displacement");
  for (const double speed : {0.0, -1.0, 100.0, std::numeric_limits<double>::quiet_NaN()}) {
    const auto invalid_speed = aim::predict_aim(target, t, t + 20ms, speed, -1, params);
    require(invalid_speed.valid && !invalid_speed.fireable, "invalid bullet speed allows only geometric tracking");
  }
  require(!aim::predict_aim(target, t, t + 501ms, 25, -1, params).valid, "stale image prediction");
  require(!aim::predict_aim(target, t, t - 1ms, 25, -1, params).valid, "future image timestamp");
  params.pitch_limit = 0.01;
  require(!aim::predict_aim(target, t, t, 25, -1, params).fireable, "pitch limit gates fire");

  aim::FireController fire;
  require(!fire.ready(t, true), "fire disabled by default");
  fire.mode = aim::FireMode::CONTINUOUS;
  require(!fire.ready(t, true), "unconfirmed external switch must gate fire");
  fire.external_gate_confirmed = true;
  require(fire.ready(t, true) && !fire.ready(t, false), "continuous mode follows aim permission");
  fire.paused = true;
  require(!fire.ready(t, true), "pause gates fire");
  fire.paused = false;
  fire.mode = aim::FireMode::SINGLE;
  fire.record_request(t);
  require(!fire.ready(t + 83ms, true) && fire.ready(t + 84ms, true), "single mode never exceeds 12 Hz");
  fire.record_request(t + 10s);
  require(!fire.ready(t + 10001ms, true), "no catch-up burst after a stall");
  for (int shot = 0; shot < 20; ++shot) {
    const auto now = t + 11s + 100ms * shot;
    require(fire.ready(now, true), "outpost must not inherit task2 ten-shot budget");
    fire.record_request(now);
  }
  std::cout << "prediction/fire: passed\n";
}
}  // namespace

int main()
{
  try {
    geometry_and_jacobian();
    rotating_streams();
    prediction_and_fire();
    return 0;
  } catch (const std::exception & e) {
    std::cerr << "FAIL: " << e.what() << '\n';
    return 1;
  }
}
