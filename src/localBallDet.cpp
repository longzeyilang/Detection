/*
 * @Author: gzy
 * @Date: 2022-03-27 09:07:46
 * @Description: file content
 */
#include "localBallDet.hpp"

#include "algorithm.hpp"
void LocalBallDet::resize_uniform(cv::Mat &src, cv::Mat &dst, cv::Size dst_size,
                              object_rect &effect_area) {}

void LocalBallDet::preprocess(cv::Mat &img, cv::Mat &roiImage,
                          object_rect &effect_area, float ball_center_x,
                          float ball_center_y) {
  // raw image size
  int width = img.cols;
  int height = img.rows;

  const int center_x_offset = inputParam.xOffset;
  const int center_y_offset = inputParam.yOffset;
  const int roi_width = inputParam.xWidth;
  const int roi_height = inputParam.yWidth;

  int xmax = 0, ymax = 0;
  // 右手
  if (st == SHOOTRIGHT) 
  {
    xmax = ball_center_x - center_x_offset + roi_width;
    ymax = ball_center_y + center_y_offset;

    xmax = std::max(roi_width, xmax);
    xmax = std::min(width, xmax);

    ymax = std::max(roi_height, ymax);
    ymax = std::min(height, ymax);

    effect_area.x = xmax - roi_width;
    effect_area.y = ymax - roi_height;
  } 
  else if (st == SHOOTLEFT) 
  {
    xmax = ball_center_x + center_x_offset;
    ymax = ball_center_y + center_y_offset;

    xmax = std::max(roi_width, xmax);
    xmax = std::min(width, xmax);

    ymax = std::max(roi_height, ymax);
    ymax = std::min(height, ymax);

    effect_area.x = xmax - roi_width;
    effect_area.y = ymax - roi_height;
  }
  cv::Rect rec =cv::Rect(cv::Point(effect_area.x, effect_area.y), cv::Point(xmax, ymax));
  // cv::rectangle(img,rec,cv::Scalar(255,0,0));
  cv::Mat rctImage = img(rec);
#ifdef USE_NEON
  roiImage = cv::Mat(cv::Size(inputParam.inputWidth, inputParam.inputHeight),
                     CV_8UC1, cv::Scalar(0));
  resize_bilinear_c1(rctImage.clone().data, roi_width, roi_height, roi_width,
                     roiImage.data, inputParam.inputWidth,
                     inputParam.inputHeight, inputParam.inputWidth);
#else
  cv::resize(rctImage, roiImage,
             cv::Size(inputParam.inputWidth, inputParam.inputHeight));
#endif
}

void LocalBallDet::postprocess(const cv::Mat &bgr,
                           const std::vector<BoxInfo> &bboxes,
                           std::vector<BoxInfo> &outbboxes,
                           object_rect effect_roi, std::string save_path) {
  int src_w = bgr.cols;
  int src_h = bgr.rows;

  const int roi_width = inputParam.xWidth;
  const int roi_height = inputParam.yWidth;

  float width_ratio = (float)roi_width / (float)inputParam.inputWidth;
  float height_ratio = (float)roi_height / (float)inputParam.inputHeight;

  // get bounding box in raw image
  for (size_t i = 0; i < bboxes.size(); i++) {
    const BoxInfo bbox = bboxes[i];
    if (inputParam.filterOpen) {
      int label = bbox.label;
      if (bbox.score < inputParam.filterScore[label]) continue;
    }
    float x1 = (bbox.x1 * width_ratio + effect_roi.x);
    float y1 = (bbox.y1 * height_ratio + effect_roi.y);
    float x2 = (bbox.x2 * width_ratio + effect_roi.x);
    float y2 = (bbox.y2 * height_ratio + effect_roi.y);
    outbboxes.emplace_back(BoxInfo{x1, y1, x2, y2, bbox.score, bbox.label});
  }

  // according to result_style, if 0, return, else get drawed boxes
  if (!inputParam.resultStyle) return;

  cv::Mat image;
  cv::cvtColor(bgr, image, cv::COLOR_GRAY2BGR);
  for (size_t i = 0; i < outbboxes.size(); i++) {
    const BoxInfo bbox = outbboxes[i];
    cv::Scalar color = cv::Scalar(inputParam.colorLists[bbox.label][0],
                                  inputParam.colorLists[bbox.label][1],
                                  inputParam.colorLists[bbox.label][2]);
    cv::rectangle(
        image,
        cv::Rect(cv::Point(bbox.x1, bbox.y1), cv::Point(bbox.x2, bbox.y2)),
        color);

    char text[256];
    sprintf(text, "%s %.1f%%", inputParam.classNames[bbox.label].c_str(),
            bbox.score * 100);

    int baseLine = 0;
    cv::Size label_size =
        cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.4, 1, &baseLine);

    int x = bbox.x1;
    int y = bbox.y1 - label_size.height - baseLine;
    y = (y < 0) ? 0 : y;
    x = (x + label_size.width > image.cols) ? image.cols - label_size.width : x;
    cv::putText(image, text, cv::Point(x, y + label_size.height),
                cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 255));
  }

  if (save_path != "None") {
    cv::imwrite(save_path, image);
    // std::cout << save_path << std::endl;
  }
}