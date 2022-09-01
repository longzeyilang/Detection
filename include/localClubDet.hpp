/*
 * @Author: gzy
 * @Date: 2022-03-20 20:22:46
 * @Description: file content
 */
#ifndef __Local_CLUB_H__
#define __Local_CLUB_H__
#pragma once
#ifdef USE_MNN
#include "nanoDetMNN.hpp"
class LocalClubDet : public NanoDetMNN {
#else
#include "nanoDetNCNN.hpp"
class LocalClubDet : public NanoDetNCNN {
#endif
 public:
  LocalClubDet(){};
  ~LocalClubDet(){};
  virtual void resize_uniform(cv::Mat &src, cv::Mat &dst, cv::Size dst_size,
                              object_rect &effect_area);
  virtual void preprocess(cv::Mat &img, cv::Mat &roiImage,
                          object_rect &effect_area, float ball_center_x = -1,
                          float ball_center_y = -1);
  virtual void postprocess(const cv::Mat &bgr,
                           const std::vector<BoxInfo> &bboxes,
                           std::vector<BoxInfo> &outbboxes,
                           object_rect effect_roi,
                           std::string save_path = "None");
};
#endif  // __Local_CLUB_H__
