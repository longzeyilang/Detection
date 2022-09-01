/*
 * @Author: gzy
 * @Date: 2022-03-20 20:22:46
 * @Description: file content
 */
#ifndef __GOLFABSTRACT_H__
#define __GOLFABSTRACT_H__

#pragma once
#include <algorithm>
#include <chrono>
#include <iostream>
#include <memory>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#define EPS 1e-6

struct BoxInfo {
  float x1;
  float y1;
  float x2;
  float y2;
  float score;
  int label;
};

// object rect
struct object_rect {
  int x;
  int y;
  int width;
  int height;
};

// all config
struct Config {
  std::string MNNmodelPath;                  // MNNmodelPath
  std::string NCNNmodelPath;                 // NCNNmodelPath
  std::string NCNNmodelParam;                // NCNNmodelParam
  std::vector<float> meanVals;               // mean vals
  std::vector<float> normVals;               // norm vals
  int inputWidth;                            // model input width
  int inputHeight;                           // model input height
  std::vector<int> strides;                  // stride of image
  int regMax;                                // regmax
  int numThreads;                            // threads of model
  float scoreThreshold;                      // score
  float nmsThreshold;                        // nms score threshold
  int numClass;                              // num class label
  std::vector<std::string> classNames;       // label name
  std::vector<std::vector<int>> colorLists;  // color list
  bool resultStyle;  // result_style, if false, return object bounding box
                     //               if true, return object bounding box in
                     //               image

  // filter
  bool filterOpen;
  std::vector<float> filterScore;  // the class filtered score

  // save
  std::string saveDir;

  // roi detection
  int xOffset;
  int yOffset;
  int xWidth;
  int yWidth;
};

// shot type
enum ShootType {
  SHOOTLEFT,
  SHOOTRIGHT,
};

class GolfAbstract {
 public:
  GolfAbstract(){};
  ~GolfAbstract(){};
  virtual bool init(Config config) = 0;
  virtual int detect(cv::Mat &img, std::vector<BoxInfo> &result_list) = 0;
  virtual void resize_uniform(cv::Mat &src, cv::Mat &dst, cv::Size dst_size,
                              object_rect &effect_area) = 0;
  virtual void preprocess(cv::Mat &img, cv::Mat &roiImage,
                          object_rect &effect_area, float ball_center_x = -1,
                          float ball_center_y = -1) = 0;
  virtual void postprocess(const cv::Mat &bgr,
                           const std::vector<BoxInfo> &bboxes,
                           std::vector<BoxInfo> &outbboxes,
                           object_rect effect_roi,
                           std::string save_path = "None") = 0;

 public:
  Config inputParam;
  ShootType st;
};
#endif  // __GOLFABSTRACT_H__
