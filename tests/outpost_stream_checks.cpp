#include <Eigen/Geometry>
#include <opencv2/opencv.hpp>
#include <nlohmann/json.hpp>
#include <chrono>
#include <cmath>
#include <iostream>
#include <list>
#include <stdexcept>
#include <thread>
#include <vector>
#include <filesystem>
#include <fstream>
#include <yaml-cpp/yaml.h>

void require(bool condition, const std::string &message) {
  if (!condition) throw std::runtime_error(message);
}
#define CHECK(condition) require(static_cast<bool>(condition), #condition)
#include "tasks/auto_aim/solver.hpp"
#include "tasks/auto_aim/target.hpp"

namespace validation {
constexpr double rad = CV_PI / 180.0;
struct Target {
  double yaw = 0, pitch = 0;
  auto_aim::Color color = auto_aim::red;
  auto_aim::ArmorName name = auto_aim::outpost;
  int invalid_points = 0;
};
struct Frame {
  bool spinning = false;
  double omega = 2.5;
  int interval_ms = 12;
  std::vector<Target> targets{{}};
  bool camera_ok = true, empty = false, invalid_q = false;
  bool throw_inference = false, exit_mode_in_inference = false;
  int age_ms = 0, inference_ms = 0, key = -1;
};
struct Packet {
  bool control, fire;
  float yaw, pitch;
  std::chrono::steady_clock::time_point time;
};
std::vector<Frame> frames;
std::vector<Packet> packets;
std::vector<nlohmann::json> plots;
std::size_t cursor = 0, current = 0;
int q_calls = 0, detect_calls = 0;
bool follow_commands = false;
int feedback_count = 0;
double bullet_speed = 25, rotor_phase = 0.1;
std::chrono::steady_clock::time_point previous_frame;
bool idle = false;
double feedback_yaw = 0, feedback_pitch = 0, look_yaw = 0, look_pitch = 0;
Eigen::Quaterniond orientation() {
  return Eigen::Quaterniond(Eigen::AngleAxisd(look_yaw * rad, Eigen::Vector3d::UnitZ()) *
    Eigen::AngleAxisd(-look_pitch * rad, Eigen::Vector3d::UnitY()));
}
}

// Substitute hardware and GUI only; the actual task_3 main, Armor, Solver and PnP run below.
#define IO__CAMERA_HPP
#define IO__GIMBAL_HPP
#define AUTO_AIM__YOLO_HPP
#define TOOLS__EXITER_HPP
#define TOOLS__PLOTTER_HPP
namespace io {
enum class GimbalMode { IDLE, AUTO_AIM };
struct GimbalState { float yaw, pitch, bullet_speed; uint16_t bullet_count; };
class Camera {
 public:
  explicit Camera(const std::string &) {}
  bool try_read_for(cv::Mat &img, std::chrono::steady_clock::time_point &t,
                    std::chrono::milliseconds) {
    CHECK(validation::cursor < validation::frames.size());
    validation::current = validation::cursor++;
    const auto &frame = validation::frames[validation::current];
    std::this_thread::sleep_for(std::chrono::milliseconds(frame.interval_ms));
    if (!frame.camera_ok) return false;  // Intentionally leave old img/t intact.
    t = std::chrono::steady_clock::now() - std::chrono::milliseconds(frame.age_ms);
    if (frame.spinning) {
      validation::rotor_phase += frame.omega *
        std::chrono::duration<double>(t - validation::previous_frame).count();
      validation::previous_frame = t;
    }
    img = frame.empty ? cv::Mat() : cv::Mat::zeros(1080, 1440, CV_8UC3);
    return true;
  }
};
class Gimbal {
 public:
  explicit Gimbal(const std::string &) {}
  GimbalState state() const {
    return {static_cast<float>(validation::feedback_yaw * validation::rad),
      static_cast<float>(validation::feedback_pitch * validation::rad),
      static_cast<float>(validation::bullet_speed), static_cast<uint16_t>(validation::feedback_count)};
  }
  GimbalMode mode() const { return validation::idle ? GimbalMode::IDLE : GimbalMode::AUTO_AIM; }
  Eigen::Quaterniond q(std::chrono::steady_clock::time_point) {
    ++validation::q_calls;
    if (validation::frames[validation::current].invalid_q)
      return Eigen::Quaterniond(0, 0, 0, 0);
    return validation::orientation();
  }
  void send(bool control, bool fire, float yaw, float pitch) {
    validation::packets.push_back({control, fire, yaw, pitch, std::chrono::steady_clock::now()});
    if (fire) ++validation::feedback_count;
    if (control && validation::follow_commands) {
      validation::feedback_yaw = validation::look_yaw = yaw / validation::rad;
      validation::feedback_pitch = validation::look_pitch = pitch / validation::rad;
    }
  }
};
}
namespace tools {
class Exiter {
 public:
  bool exit() const {
    return validation::cursor == validation::frames.size() &&
      validation::packets.size() >= validation::frames.size();
  }
};
class Plotter {
 public:
  void plot(const nlohmann::json &json) { validation::plots.push_back(json); }
};
}
namespace auto_aim {
class YOLO {
  Solver projector_;
 public:
  YOLO(const std::string &path, bool) : projector_(path) {}
  std::list<Armor> detect(const cv::Mat &, int frame_count) {
    CHECK(frame_count == validation::detect_calls++);
    const auto &frame = validation::frames[validation::current];
    if (frame.throw_inference) throw std::runtime_error("injected detector failure");
    std::this_thread::sleep_for(std::chrono::milliseconds(frame.inference_ms));
    projector_.set_R_gimbal2world(validation::orientation());
    std::list<Armor> result;
    for (const auto &target : frame.targets) {
      double yaw = target.yaw * validation::rad, pitch = target.pitch * validation::rad;
      Eigen::Vector3d position(3 * std::cos(pitch) * std::cos(yaw),
        3 * std::cos(pitch) * std::sin(yaw), 3 * std::sin(pitch));
      if (frame.spinning) {
        int selected = 0;
        double best = 10;
        for (int id=0; id<3; ++id) {
          const double angle = std::atan2(std::sin(validation::rotor_phase+id*2*CV_PI/3),
            std::cos(validation::rotor_phase+id*2*CV_PI/3));
          if (std::abs(angle)<best) { selected=id; best=std::abs(angle); }
        }
        yaw = validation::rotor_phase + selected * 2*CV_PI/3;
        const double radius = 0.2765;
        position = Eigen::Vector3d(3-radius*std::cos(yaw), -radius*std::sin(yaw),
          0.1 + selected * 0.1);
      }
      const auto points = projector_.reproject_armor(position, yaw, special, target.name);
      result.emplace_back(1, 6, 0.99f, cv::Rect(), points);
      auto &armor = result.back();
      armor.color = target.color;
      armor.name = target.name;
      switch (target.invalid_points) {
        case 1: armor.points.assign(4, cv::Point2f(0, 0)); break;
        case 2: armor.points[0].x = std::numeric_limits<float>::quiet_NaN(); break;
        case 3: std::swap(armor.points[0], armor.points[1]); break;
        case 4: armor.points[0].x = -2; break;
        case 5: armor.points.resize(3); break;
      }
    }
    if (frame.exit_mode_in_inference) validation::idle = true;
    return result;
  }
};
}
namespace cv {
void test_imshow(const String &, const Mat &) {}
int test_waitKey(int) { return validation::frames[validation::current].key; }
void test_destroyAllWindows() {}
}
#define imshow test_imshow
#define waitKey test_waitKey
#define destroyAllWindows test_destroyAllWindows
#define main task3_entry
#include "src/task_3.cpp"
#undef main
#undef imshow
#undef waitKey
#undef destroyAllWindows

namespace validation {
void run(const std::string &name, std::vector<Frame> input, const std::string &path,
         double yaw = 0, double pitch = 0, double camera_pitch = 0, bool manual = false) {
  frames = std::move(input);
  packets.clear(); plots.clear(); cursor = current = 0; q_calls = detect_calls = 0;
  idle = manual;
  rotor_phase = 0.1;
  previous_frame = std::chrono::steady_clock::now();
  feedback_count = 0;
  feedback_yaw = look_yaw = yaw;
  feedback_pitch = pitch; look_pitch = camera_pitch;
  std::string executable = "task3_validation", config = path;
  char *argv[] = {executable.data(), config.data(), nullptr};
  CHECK(task3_entry(2, argv) == 0);
  CHECK(packets.size() == frames.size() + 1);
  CHECK(plots.size() == packets.size());
  for (std::size_t i = 0; i < packets.size(); ++i) {
    const auto &packet = packets[i];
    CHECK((!packet.fire || packet.control) && std::isfinite(packet.yaw) && std::isfinite(packet.pitch));
    CHECK(plots[i]["yaw_cmd_rad"].get<float>() == packet.yaw);
    CHECK(plots[i]["pitch_cmd_rad"].get<float>() == packet.pitch);
    CHECK(plots[i]["control_cmd"].get<bool>() == packet.control);
    CHECK(plots[i]["fire_cmd"].get<bool>() == packet.fire);
  }
  CHECK(!packets.back().control);
  std::cout << "PASS: " << name << '\n';
}
}


int main(int argc, char **argv)
{
  using namespace validation;
  try {
    CHECK(argc == 2);
    const std::string config = argv[1];
    run("default is tracking only and actual commands match Plotter", std::vector<Frame>(4), config);
    CHECK(packets[0].control && plots[0].contains("omega_ekf_rad_s"));
    for (const auto &packet : packets) CHECK(!packet.fire);
    CHECK(std::abs(plots[0]["radius_m"].get<double>() - 0.2765) < 1e-12);

    std::vector<Frame> loss(5);
    loss[1].targets.clear(); loss[2].camera_ok = false; loss[3].empty = true;
    run("no observation, camera timeout, empty image", loss, config);
    for (int frame : {1, 2, 3}) {
      CHECK(!plots[frame]["tracking_valid"].get<bool>() && !packets[frame].fire);
    }
    CHECK(plots[4]["tracking_valid"].get<bool>());
    Frame frame; frame.age_ms = 400;
    run("stale image bypasses detector and pose", {frame}, config);
    CHECK(q_calls == 0 && detect_calls == 0 && !packets[0].control);
    frame = Frame(); frame.inference_ms = 300;
    run("slow inference disables control and fire", {frame}, config);
    CHECK(!packets[0].control && !packets[0].fire);
    frame = Frame(); frame.invalid_q = true;
    run("invalid quaternion", {frame}, config);
    CHECK(!packets[0].control && detect_calls == 0);
    frame = Frame(); frame.throw_inference = true;
    run("detector exception pauses firing", {frame}, config);
    CHECK(!packets[0].control && plots[0]["fire_paused"].get<bool>());
    for (const auto name : {auto_aim::three, auto_aim::base}) {
      frame = Frame(); frame.targets[0].name = name;
      run("reject non-outpost armor", {frame}, config);
      CHECK(!plots[0].contains("omega_ekf_rad_s"));
    }
    frame = Frame(); frame.targets[0].color = auto_aim::blue;
    run("reject friendly outpost", {frame}, config);
    CHECK(!plots[0].contains("omega_ekf_rad_s"));
    for (int kind = 1; kind <= 5; ++kind) {
      frame = Frame(); frame.targets[0].invalid_points = kind;
      run("reject malformed keypoints " + std::to_string(kind), {frame}, config);
      CHECK(!packets[0].control);
    }
    frame = Frame(); frame.exit_mode_in_inference = true;
    run("mode change during inference", {frame}, config);
    CHECK(!packets[0].control && !packets[0].fire);
    run("manual mode", {Frame()}, config, 0, 0, 0, true);
    CHECK(!packets[0].control);
    frame = Frame(); frame.key = 27;
    run("Esc releases before send", {frame}, config);
    CHECK(!packets[0].control);
    bullet_speed = 0;
    run("invalid bullet speed keeps geometric tracking", {Frame()}, config);
    CHECK(packets[0].control && !packets[0].fire);
    bullet_speed = 25;

    const auto temporary = std::filesystem::temp_directory_path() /
      ("outpost-stream-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".yaml");
    struct Cleanup {
      std::filesystem::path path;
      ~Cleanup() { std::error_code error; std::filesystem::remove(path, error); }
    } cleanup{temporary};
    auto yaml = YAML::LoadFile(config);
    yaml["task3_fire_mode"] = "single";
    yaml["task3_external_fire_gate_confirmed"] = false;
    { std::ofstream output(temporary); output << yaml; }
    std::string executable = "invalid_gate", path = temporary.string();
    char *arguments[] = {executable.data(), path.data(), nullptr};
    CHECK(task3_entry(2, arguments) == 1);
    yaml["task3_external_fire_gate_confirmed"] = true;
    { std::ofstream output(temporary); output << yaml; }
    for (const double omega : {-2.5, 2.5}) {
      std::vector<Frame> spinning(650);
      for (auto &f : spinning) { f.spinning = true; f.omega = omega; }
      spinning[350].key = ' ';
      spinning[400].key = 'r';
      // 模型已建立后注入短暂丢帧，必须抑制该帧的 fire。
      spinning[480].camera_ok = false;
      follow_commands = true;
      run("actual PnP/EKF/prediction loop, omega=" + std::to_string(omega), spinning, path);
      int fires = 0, model_ready = 0;
      std::optional<std::chrono::steady_clock::time_point> last;
      for (std::size_t i = 0; i + 1 < packets.size(); ++i) {
        if (plots[i]["model_ready"].get<bool>()) ++model_ready;
        if (i >= 350 && i < 400) CHECK(!packets[i].fire);
        if (packets[i].fire) {
          ++fires;
          CHECK(plots[i]["tracking_valid"].get<bool>());
          CHECK(plots[i]["model_ready"].get<bool>() && plots[i]["aim_fireable"].get<bool>());
          if (last) CHECK(std::chrono::duration<double>(packets[i].time - *last).count() >= 1.0 / 12.0);
          last = packets[i].time;
        }
      }
      CHECK(!packets[480].fire);
      std::cout << "Live loop: fire requests=" << fires << " ready frames=" << model_ready
        << " final speed=" << plots[645]["omega_ekf_rad_s"] << '\n';
      CHECK(fires > 10 && model_ready > 100);
      CHECK(std::abs(plots[645]["omega_ekf_rad_s"].get<double>() - omega) < 0.4);
    }
    std::cout << "PASS: actual task3 loop, both signs, learned heights, pause, loss, >10 requests and <=12 Hz\n";
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "FAIL: " << e.what() << '\n';
    return 1;
  }
}
