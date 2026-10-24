cmake_minimum_required(VERSION 3.16.3)

project(final_project_aim)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)
message(STATUS "--------------------CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}--------------------")

find_package(OpenCV REQUIRED)
find_package(fmt REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(spdlog REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(nlohmann_json REQUIRED)
set(OpenVINO_DIR "/opt/intel/openvino_2024.6.0/runtime/cmake/")
find_package(OpenVINO REQUIRED)

include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR})


add_subdirectory(tools)
add_subdirectory(io)
add_subdirectory(tasks/auto_aim)

####################src###################
add_executable(task_1 src/task_1.cpp)
add_executable(task_2 src/task_2.cpp)
add_executable(task_3 src/task_3.cpp)

target_link_libraries(task_1 ${OpenCV_LIBS} auto_aim fmt::fmt yaml-cpp tools io)
target_link_libraries(task_2 ${OpenCV_LIBS} auto_aim fmt::fmt yaml-cpp tools io)
target_link_libraries(task_3 ${OpenCV_LIBS} auto_aim fmt::fmt yaml-cpp tools io)

include(CTest)
if(BUILD_TESTING)
  add_executable(outpost_checks tests/outpost_checks.cpp)
  target_link_libraries(outpost_checks ${OpenCV_LIBS} auto_aim fmt::fmt yaml-cpp tools io)
  add_test(NAME outpost_checks COMMAND outpost_checks)
  # 只替换相机、串口、检测器和 GUI；运行实际 task_3 主循环、PnP、EKF 与弹道。
  add_executable(outpost_stream_checks
    tests/outpost_stream_checks.cpp
    tasks/auto_aim/armor.cpp tasks/auto_aim/solver.cpp
    tasks/auto_aim/target.cpp tasks/auto_aim/outpost_aim.cpp
    tools/extended_kalman_filter.cpp tools/math_tools.cpp
    tools/img_tools.cpp tools/logger.cpp tools/trajectory.cpp)
  target_link_libraries(outpost_stream_checks ${OpenCV_LIBS} fmt::fmt yaml-cpp spdlog::spdlog pthread)
  add_test(NAME outpost_stream_checks
    COMMAND outpost_stream_checks ${PROJECT_SOURCE_DIR}/tests/task3_stream.yaml)
  set_tests_properties(outpost_stream_checks PROPERTIES TIMEOUT 90)
endif()
