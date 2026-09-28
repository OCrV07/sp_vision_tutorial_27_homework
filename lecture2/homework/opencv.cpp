#include <iostream>
#include <opencv2/opencv.hpp>

#include "io/camera.hpp"
#include "tasks/apriltag_detector.hpp"
#include "tools/img_tools.hpp"

int main()
{
  try {
    Camera camera;
    auto_charge::AprilTagDetector detector("./configs/yolo.yaml");
    cv::namedWindow("img", cv::WINDOW_AUTOSIZE);

    while (true) {
      cv::Mat img = camera.read();

      if (img.empty()) {
        if (cv::waitKey(1) == 'q') break;
        continue;
      }

      const auto detections = detector.detect(img);
      for (const auto & detection : detections) {
        tools::draw_points(img, detection.corners, cv::Scalar(0, 255, 0), 2);

        const std::string text = "ID: " + std::to_string(detection.id);
        cv::Point text_point(
          static_cast<int>(detection.corners[0].x), static_cast<int>(detection.corners[0].y) - 8);

        int baseline = 0;
        const auto text_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.7, 2, &baseline);
        if (text_point.y - text_size.height < 0) {
          text_point.y = static_cast<int>(detection.corners[2].y) + text_size.height + 8;
        }
        if (text_point.x + text_size.width >= img.cols) {
          text_point.x = std::max(0, img.cols - text_size.width - 1);
        }

        tools::draw_text(img, text, text_point, cv::Scalar(0, 255, 0), 2, 2);
      }

      cv::resize(img, img, cv::Size(1280, 960));
      cv::imshow("img", img);
      if (cv::waitKey(1) == 'q') break;
    }
  } catch (const std::exception & e) {
    std::cerr << "运行失败: " << e.what() << std::endl;
    return 1;
  }

  return 0;
}
