/*
 * @Author: gzy
 * @Date: 2022-03-27 09:05:02
 * @Description: file content
 */
#include "gloalDet.hpp"

#include "algorithm.hpp"
void GloalDet::resize_uniform(cv::Mat &src, cv::Mat &dst, cv::Size dst_size,
                              object_rect &effect_area) {
  int w = src.cols;
  int h = src.rows;
  int dst_w = dst_size.width;
  int dst_h = dst_size.height;

  dst = cv::Mat(cv::Size(dst_w, dst_h), CV_8UC1, cv::Scalar(0));

  float ratio_src = w * 1.0 / h;
  float ratio_dst = dst_w * 1.0 / dst_h;

  int tmp_w = 0;
  int tmp_h = 0;
  if (ratio_src > ratio_dst) {
    tmp_w = dst_w;
    tmp_h = floor((dst_w * 1.0 / w) * h);
  } else if (ratio_src < ratio_dst) {
    tmp_h = dst_h;
    tmp_w = floor((dst_h * 1.0 / h) * w);
  } else {
    cv::resize(src, dst, dst_size);
    effect_area.x = 0;
    effect_area.y = 0;
    effect_area.width = dst_w;
    effect_area.height = dst_h;
    return;
  }

  uint8_t *tmp = (uint8_t *)malloc(tmp_w * tmp_h);  // tmp_w=256 tmp_h=192
#ifdef USE_NEON
  resize_bilinear_c1(src.data, w, h, w, tmp, tmp_w, tmp_h, tmp_w);
#else
  cv::Mat tmpMat(tmp_h, tmp_w, CV_8UC1, tmp);
  cv::resize(src, tmpMat, cv::Size(tmp_w, tmp_h));  // tmp_w=256 tmp_h=192
#endif

  if (tmp_w != dst_w) {
    int index_w = floor((dst_w - tmp_w) / 2.0);
    for (int i = 0; i < dst_h; i++)
      memcpy(dst.data + i * dst_w + index_w, tmp + i * tmp_w, tmp_w);
    effect_area.x = index_w;
    effect_area.y = 0;
    effect_area.width = tmp_w;
    effect_area.height = tmp_h;
  } else if (tmp_h != dst_h) {
    int index_h = floor((dst_h - tmp_h) / 2.0);
    memcpy(dst.data + index_h * dst_w, tmp, tmp_w * tmp_h);
    effect_area.x = 0;
    effect_area.y = index_h;
    effect_area.width = tmp_w;
    effect_area.height = tmp_h;
  } else {
    printf("error\n");
  }

  free(tmp);
}

void GloalDet::preprocess(cv::Mat &img, cv::Mat &roiImage,
                          object_rect &effect_area, float ball_center_x,
                          float ball_center_y) {
  resize_uniform(img, roiImage,
                 cv::Size(inputParam.inputWidth, inputParam.inputHeight),
                 effect_area);
}

void GloalDet::postprocess(const cv::Mat &bgr,
                           const std::vector<BoxInfo> &bboxes,
                           std::vector<BoxInfo> &outbboxes,
                           object_rect effect_roi, std::string save_path) {
  int src_w = bgr.cols;
  int src_h = bgr.rows;
  int dst_w = effect_roi.width;
  int dst_h = effect_roi.height;
  float width_ratio = (float)src_w / (float)dst_w;
  float height_ratio = (float)src_h / (float)dst_h;
  int num = 0;

  // get bounding box in raw image
  for (size_t i = 0; i < bboxes.size(); i++) {
    const BoxInfo bbox = bboxes[i];
    if (inputParam.filterOpen) {
      int label = bbox.label;
      if (bbox.score < inputParam.filterScore[label]) continue;
    }

    float x1 = (bbox.x1 - effect_roi.x) * width_ratio;
    float y1 = (bbox.y1 - effect_roi.y) * height_ratio;
    float x2 = (bbox.x2 - effect_roi.x) * width_ratio;
    float y2 = (bbox.y2 - effect_roi.y) * height_ratio;

    outbboxes.emplace_back(BoxInfo{x1, y1, x2, y2, bbox.score, bbox.label});
  }

  // according to result_style, if fasle, return, else get drawed boxes
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

  if (save_path != "None" /*&& num != 1*/) {
    cv::imwrite(save_path, image);
  }

  // if (outbboxes.size() > 2)
  // {
  // cv::imshow("1", image);
  // cv::waitKey(0);
  // }
}