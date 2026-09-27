#pragma once

#include <opencv2/opencv.hpp>

#include "hikrobot/include/MvCameraControl.h"

class Camera
{
public:
  Camera();
  ~Camera();

  cv::Mat read();

private:
  Camera(const Camera &) = delete;
  Camera & operator=(const Camera &) = delete;

  void * handle_;
  bool opened_;
  bool grabbing_;

  cv::Mat transfer(const MV_FRAME_OUT & raw);
  void release();
  static void check(int ret, const char * message);
};