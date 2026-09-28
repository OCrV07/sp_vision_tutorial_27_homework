#include <fmt/format.h>

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

        // 在装甲板框线旁显示颜色和编号。
        const auto info =
          fmt::format("{} {}", auto_aim::COLORS[armor.color], auto_aim::ARMOR_NAMES[armor.name]);

        // 优先显示在框线左上方；若空间不足则放到框线下方。
        cv::Point text_point(armor.box.x, armor.box.y - 8);
        int baseline = 0;
        const auto text_size = cv::getTextSize(info, cv::FONT_HERSHEY_SIMPLEX, 0.55, 1, &baseline);
        if (text_point.y - text_size.height < 0) {
          text_point.y = armor.box.y + armor.box.height + text_size.height + 8;
        }
        if (text_point.x + text_size.width >= img.cols) {
          text_point.x = std::max(0, img.cols - text_size.width - 1);
        }

        // 显示文字
        tools::draw_text(img, info, text_point, cv::Scalar(0, 255, 0), 2, 1);
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
