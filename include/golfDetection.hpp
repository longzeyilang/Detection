/*
 * @Author: gzy
 * @Date: 2022-03-20 20:22:46
 * @Description: file content
 */

#ifndef GOLFDETECTION_H
#define GOLFDETECTION_H
#pragma once
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <unistd.h>

#include <chrono>
#include <memory>
#include <opencv2/opencv.hpp>
#include <string>

#include "golfAbstract.hpp"

namespace golf {
class GolfDetection {
 public:
  int init(const std::string confPath, const std::string logPath);
  std::vector<BoxInfo> run(cv::Mat src, float ball_center_x = 0,
                           float ball_center_y = 0,
                           ShootType shoottype = SHOOTRIGHT,
                           std::string savedPath = "None");
 private:
  int m_width = 0;
  int m_height = 0;
  std::string m_saveDir;
  std::shared_ptr<GolfAbstract> ndPtr = nullptr;
  std::vector<std::string> algorType{"global", "localBall", "localClub"};

  int algorIndex = -1;
  std::string m_algorType;
};

}  // namespace golf
#endif