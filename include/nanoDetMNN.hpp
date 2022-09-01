/*
 * @Author: gzy
 * @Date: 2022-03-20 20:22:46
 * @Description: file content
 */
#ifndef __NanoDetMNN_H__
#define __NanoDetMNN_H__

#pragma once

#include <algorithm>
#include <chrono>
#include <iostream>
#include <memory>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "MNN/ImageProcess.hpp"
#include "MNN/Interpreter.hpp"
#include "MNN/MNNDefine.h"
#include "MNN/Tensor.hpp"
#include "golfAbstract.hpp"

#define EPS 1e-6

typedef struct HeadInfo_ {
  std::string cls_layer;
  std::string dis_layer;
  int stride;
} HeadInfo;

typedef struct CenterPrior_ {
  int x;
  int y;
  int stride;
} CenterPrior;

class NanoDetMNN : public GolfAbstract {
 public:
  NanoDetMNN(){};
  ~NanoDetMNN();
  bool init(Config config);
  int detect(cv::Mat &img, std::vector<BoxInfo> &result_list);
  virtual void resize_uniform(cv::Mat &src, cv::Mat &dst, cv::Size dst_size,
                              object_rect &effect_area){};
  virtual void preprocess(cv::Mat &img, cv::Mat &roiImage,
                          object_rect &effect_area, float ball_center_x = -1,
                          float ball_center_y = -1){};
  virtual void postprocess(const cv::Mat &bgr,
                           const std::vector<BoxInfo> &bboxes,
                           std::vector<BoxInfo> &outbboxes,
                           object_rect effect_roi,
                           std::string save_path = "None"){};

  // modify these parameters to the same with your config if you want to use
  // your own model
  int reg_max = 7;  // `reg_max` set in the training config. Default: 7.
  std::vector<int> strides = {0};  // strides of the multi-level feature.  ,32
  std::string input_name = "data";
  std::string output_name = "output";
  int image_w;
  int image_h;

 private:
  void decode_infer(MNN::Tensor *pred, std::vector<CenterPrior> center_priors,
                    float threshold,
                    std::vector<std::vector<BoxInfo>> &results);
  BoxInfo disPred2Bbox(const float *&dfl_det, int label, float score, int x,
                       int y, int stride);
  void nms(std::vector<BoxInfo> &input_boxes, float NMS_THRESH);

 private:
  std::shared_ptr<MNN::Interpreter> NanoDetMNN_interpreter= nullptr;
  std::shared_ptr<MNN::CV::ImageProcess> pretreat_data= nullptr;
  MNN::Session *NanoDetMNN_session = nullptr;
  MNN::Tensor *input_tensor = nullptr;
  std::vector<CenterPrior> center_priors;
};
#endif  // __NanoDetMNN_H__
