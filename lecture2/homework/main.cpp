#include <iostream>
#include <opencv2/opencv.hpp>

#include "io/camera.hpp"
#include "tasks/yolo.hpp"
#include "tools/img_tools.hpp"

int main()
{
  try {
    Camera camera;
    auto_aim::YOLO yolo("./configs/yolo.yaml", false);
    cv::namedWindow("img", cv::WINDOW_AUTOSIZE);

    while (true) {
      cv::Mat img = camera.read();

      if (img.empty()) {
        if (cv::waitKey(1) == 'q') break;
        continue;
      }

      const auto armors = yolo.detect(img);
      for (const auto & armor : armors) {
        tools::draw_points(img, armor.points, cv::Scalar(0, 255, 0));
      }

      cv::resize(img, img, cv::Size(640, 480));
      cv::imshow("img", img);
      if (cv::waitKey(1) == 'q') break;
    }
  } catch (const std::exception & e) {
    std::cerr << "运行失败: " << e.what() << std::endl;
    return 1;
  }

  return 0;
}
