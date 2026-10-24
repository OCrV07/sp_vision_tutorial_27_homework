#ifndef AUTO_AIM__OUTPOST_AIM_HPP
#define AUTO_AIM__OUTPOST_AIM_HPP

#include <array>
#include <chrono>
#include <cstdint>
#include <optional>

#include "target.hpp"

namespace auto_aim::outpost_aim
{
using Clock = std::chrono::steady_clock;
using TimePoint = Clock::time_point;
constexpr double DEG = 3.14159265358979323846 / 180.0;

struct Parameters
{
  double yaw_offset = 0;
  double pitch_offset = 0;
  double execution_delay = 0.015;  // s，不重复计算已经过去的图像处理耗时
  double pitch_limit = 20.0 * DEG;  // 上车时按哨兵实测限位配置
};

struct AimSolution
{
  bool valid = false;
  bool fireable = false;
  int id = -1;
  double yaw = 0;
  double pitch = 0;
  double height = 0;
  double fly_time = 0;
  double horizon = 0;
  double yaw_tolerance = 0;
  double pitch_tolerance = 0;
};

bool healthy(const Target & target);
bool layers_ready(const Target & target);
double association_score(const Armor & armor, const Target & target);
AimSolution predict_aim(
  const Target & target, TimePoint model_time, TimePoint now, double bullet_speed,
  int previous_id, const Parameters & params);

enum class FireMode { DISABLED, CONTINUOUS, SINGLE };

// AUTO_AIM 只表示跟随。必须确认电控会用独立开火挡位门控，才可发送 fire 请求。
// CONTINUOUS 由电控维持 12 Hz；SINGLE 限制请求间隔，不补发、不设任务二的十发预算。
struct FireController
{
  FireMode mode = FireMode::DISABLED;
  bool external_gate_confirmed = false;
  bool paused = false;
  uint64_t request_packets = 0;
  std::optional<TimePoint> last_request;

  bool ready(TimePoint now, bool aim_ready) const;
  void record_request(TimePoint now);
};
}  // namespace auto_aim::outpost_aim

#endif  // AUTO_AIM__OUTPOST_AIM_HPP
