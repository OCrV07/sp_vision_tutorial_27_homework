#include "camera.hpp"

#include <iostream>
#include <stdexcept>

Camera::Camera() : handle_(nullptr), opened_(false), grabbing_(false)
{
  MV_CC_DEVICE_INFO_LIST device_list = {};

  // 1. 枚举 USB 相机
  int ret = MV_CC_EnumDevices(MV_USB_DEVICE, &device_list);
  check(ret, "MV_CC_EnumDevices failed");

  if (device_list.nDeviceNum == 0) {
    throw std::runtime_error("No camera found");
  }

  // 2. 创建相机句柄
  ret = MV_CC_CreateHandle(&handle_, device_list.pDeviceInfo[0]);

  check(ret, "MV_CC_CreateHandle failed");

  try {
    // 3. 打开相机
    ret = MV_CC_OpenDevice(handle_);
    check(ret, "MV_CC_OpenDevice failed");

    opened_ = true;

    // 4. 设置相机参数
    ret = MV_CC_SetEnumValue(handle_, "BalanceWhiteAuto", MV_BALANCEWHITE_AUTO_CONTINUOUS);
    check(ret, "Set BalanceWhiteAuto failed");

    ret = MV_CC_SetEnumValue(handle_, "ExposureAuto", MV_EXPOSURE_AUTO_MODE_OFF);
    check(ret, "Set ExposureAuto failed");

    ret = MV_CC_SetEnumValue(handle_, "GainAuto", MV_GAIN_MODE_OFF);
    check(ret, "Set GainAuto failed");

    ret = MV_CC_SetFloatValue(handle_, "ExposureTime", 10000);
    check(ret, "Set ExposureTime failed");

    ret = MV_CC_SetFloatValue(handle_, "Gain", 0);
    check(ret, "Set Gain failed");

    ret = MV_CC_SetFrameRate(handle_, 60);
    check(ret, "Set FrameRate failed");

    // 5. 开始采集
    ret = MV_CC_StartGrabbing(handle_);
    check(ret, "MV_CC_StartGrabbing failed");

    grabbing_ = true;
  } catch (...) {
    // 构造过程中如果任何一步失败，
    // 立即释放已经成功申请的资源
    release();

    throw;
  }
}

Camera::~Camera() { release(); }

// 读取图像
cv::Mat Camera::read()
{
  MV_FRAME_OUT raw = {};

  const unsigned int timeout = 100;

  // 从相机中取得一帧
  int ret = MV_CC_GetImageBuffer(handle_, &raw, timeout);

  if (ret != MV_OK) {
    std::cerr << "MV_CC_GetImageBuffer failed, error code = " << ret << std::endl;

    return cv::Mat();
  }

  cv::Mat img;

  try {
    // 将海康相机原始图像转换成 OpenCV BGR 图像
    img = transfer(raw);
  } catch (...) {
    // 即使转换失败，也必须把相机 SDK 的缓冲区还回去
    MV_CC_FreeImageBuffer(handle_, &raw);

    throw;
  }

  // 用完 SDK 图像缓冲区后必须归还
  ret = MV_CC_FreeImageBuffer(handle_, &raw);

  check(ret, "MV_CC_FreeImageBuffer failed");

  return img;
}

// 格式转换
cv::Mat Camera::transfer(const MV_FRAME_OUT & raw)
{
  const int width = static_cast<int>(raw.stFrameInfo.nWidth);

  const int height = static_cast<int>(raw.stFrameInfo.nHeight);

  // OpenCV 默认彩色图像：
  // 8 bit + 3 channels + BGR
  cv::Mat img(height, width, CV_8UC3);

  MV_CC_PIXEL_CONVERT_PARAM cvt_param = {};

  // 原始图像信息
  cvt_param.nWidth = raw.stFrameInfo.nWidth;

  cvt_param.nHeight = raw.stFrameInfo.nHeight;

  cvt_param.pSrcData = raw.pBufAddr;

  cvt_param.nSrcDataLen = raw.stFrameInfo.nFrameLen;

  cvt_param.enSrcPixelType = raw.stFrameInfo.enPixelType;

  // 转换后的目标图像
  cvt_param.pDstBuffer = img.data;

  cvt_param.nDstBufferSize = static_cast<unsigned int>(img.total() * img.elemSize());

  // OpenCV imshow 默认使用 BGR
  cvt_param.enDstPixelType = PixelType_Gvsp_BGR8_Packed;

  int ret = MV_CC_ConvertPixelType(handle_, &cvt_param);

  check(ret, "MV_CC_ConvertPixelType failed");

  return img;
}

void Camera::release()
{
  if (handle_ == nullptr) return;

  // 顺序和打开过程相反：

  // 1. 停止采集
  if (grabbing_) {
    MV_CC_StopGrabbing(handle_);
    grabbing_ = false;
  }

  // 2. 关闭设备
  if (opened_) {
    MV_CC_CloseDevice(handle_);
    opened_ = false;
  }

  // 3. 销毁句柄
  MV_CC_DestroyHandle(handle_);
  handle_ = nullptr;
}

// 错误检查
void Camera::check(int ret, const char * message)
{
  if (ret != MV_OK) {
    std::cerr << message << ", error code = " << ret << std::endl;

    throw std::runtime_error(message);
  }
}