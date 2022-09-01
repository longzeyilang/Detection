/*
 * @Author: gzy
 * @Date: 2022-03-21 08:49:37
 * @Description: file content
 */
#include "golfDetection.hpp"

#include "Log.h"
#include "gloalDet.hpp"
#include "localBallDet.hpp"
#include "localClubDet.hpp"
#include "yaml-cpp/yaml.h"
namespace golf {
int GolfDetection::init(const std::string confPath, const std::string logPath) {
  Log::Initialise(logPath);
  std::stringstream sstream;
  sstream << "confPath: " << confPath << "\n";
  sstream << "logPath: " << logPath << "\n";
  Log::Info(sstream.str());
  sstream.clear();

  /* Node conf. */
  YAML::Node conf = YAML::LoadFile(confPath);
  Config config;
#ifdef USE_MNN
  config.MNNmodelPath = conf["model"]["MNNmodelPath"].as<std::string>();
#else
  config.NCNNmodelPath = conf["model"]["NCNNmodelPath"].as<std::string>();
  config.NCNNmodelParam = conf["model"]["NCNNmodelParam"].as<std::string>();
#endif
  config.meanVals = conf["model"]["meanVals"].as<std::vector<float>>();
  config.normVals = conf["model"]["normVals"].as<std::vector<float>>();
  config.inputWidth = conf["model"]["inputWidth"].as<int>();
  config.inputHeight = conf["model"]["inputHeight"].as<int>();
  config.strides = conf["model"]["strides"].as<std::vector<int>>();
  config.regMax = conf["model"]["regMax"].as<int>();
  config.numThreads = conf["model"]["numThreads"].as<int>();
  config.scoreThreshold = conf["model"]["scoreThreshold"].as<float>();
  config.nmsThreshold = conf["model"]["nmsThreshold"].as<float>();
  config.numClass = conf["model"]["numClass"].as<int>();

  config.classNames =
      conf["model"]["classNames"].as<std::vector<std::string>>();
  config.colorLists = {{255, 0, 0}, {0, 255, 0}};  // todo
  config.resultStyle = conf["model"]["resultStyle"].as<bool>();
  m_algorType = conf["model"]["type"].as<std::string>();

  // roi
  config.xOffset = conf["roi"]["xOffset"].as<int>();
  config.yOffset = conf["roi"]["yOffset"].as<int>();
  config.xWidth = conf["roi"]["xWidth"].as<int>();
  config.yWidth = conf["roi"]["yWidth"].as<int>();

  // filter
  config.filterOpen = conf["filter"]["open"].as<bool>();
  config.filterScore = conf["filter"]["score"].as<std::vector<float>>();

  // save dir
  config.saveDir = conf["save"]["dir"].as<std::string>();
  std::string::size_type index = config.saveDir.rfind("/");
  if (index != (config.saveDir.size() - 1)) config.saveDir += "/";

  // algorithm init
  std::vector<std::string>::iterator it =
      std::find(algorType.begin(), algorType.end(), m_algorType);
  if (it == algorType.end()) {
    sstream << "m_algorType: " << m_algorType << " not in defult algorType."
            << "\n";
    std::cout<<sstream.str()<<std::endl;
    Log::Info(sstream.str());
    Log::Finalise();
  } else {
    m_algorType = *it;
    int nPosition = std::distance(algorType.begin(), it);
    algorIndex = nPosition;
    switch (nPosition) {
      case 0:
        ndPtr.reset(new GloalDet());
        break;
      case 1:
        ndPtr.reset(new LocalBallDet());
        break;
      case 2:
        ndPtr.reset(new LocalClubDet());
        break;
      default:
        break;
    }
  }
  ndPtr->init(config);
  #ifdef USE_MNN
  sstream << "MNNmodelPath: " << config.MNNmodelPath << "\n";
  #else
  sstream << "NCNNmodelPath: " << config.NCNNmodelPath << "\n";
  sstream << "NCNNmodelParam: " << config.NCNNmodelParam << "\n";
  #endif
  sstream << "meanVals:" << config.meanVals[0] << "," << config.meanVals[1]
          << "," << config.meanVals[2] << "\n";
  sstream << "normVals:" << config.normVals[0] << "," << config.normVals[1]
          << "," << config.normVals[2] << "\n";
  sstream << "inputWidth: " << config.inputWidth << "\n";
  sstream << "inputHeight: " << config.inputHeight << "\n";

  std::string strides_str;
  strides_str += "strides:";
  for (auto &stride : config.strides)
    strides_str += std::to_string(stride) + ",";
  sstream << strides_str << "\n";
  sstream << "regMax: " << config.regMax << "\n";

  sstream << "threads: " << config.numThreads << "\n";
  sstream << "score: " << config.scoreThreshold << "\n";
  sstream << "nms: " << config.nmsThreshold << "\n";
  sstream << "num_class: " << config.numClass << "\n";
  std::string classNames_str;
  classNames_str += "className:";
  for (auto &name : config.classNames) classNames_str += name + ",";
  sstream << classNames_str << "\n";
  sstream << "resultStyle: " << config.resultStyle << "\n";

  sstream << "xOffset:" << config.xOffset << "\n";
  sstream << "yOffset:" << config.yOffset << "\n";
  sstream << "xWidth:" << config.xWidth << "\n";
  sstream << "yWidth:" << config.yWidth << "\n";

  sstream << "filter open: " << config.filterOpen << "\n";
  std::string filterscore_str;
  filterscore_str += "filter scores:";
  for (auto &name : config.filterScore)
    filterscore_str += std::to_string(name) + ",";
  sstream << filterscore_str << "\n";

  sstream << "save dir: " << config.saveDir << "\n";
  Log::Info("load from conf yaml:" + sstream.str());
  return 0;
}

std::vector<BoxInfo> GolfDetection::run(cv::Mat src, float ball_center_x,
                                        float ball_center_y,
                                        ShootType shoottype,
                                        std::string savedPath) {
  std::vector<BoxInfo> rawBoxes;
  if (algorIndex == -1) {
    Log::Error("detection algor not initalized, please check");
    Log::Finalise();
    return rawBoxes;
  }
  if (m_algorType != algorType[algorIndex]) {
    Log::Error("load config yaml " + m_algorType + ", but use " +
               algorType[algorIndex] + " function.");
    Log::Finalise();
    return rawBoxes;
  }
  if (src.empty()) {
    Log::Error("image is empty ,please check!");
    Log::Finalise();
    return rawBoxes;
  }
  object_rect effect_roi;
  cv::Mat roi_image;
  ndPtr->st = shoottype;
  ndPtr->preprocess(src, roi_image, effect_roi, ball_center_x, ball_center_y);
  std::vector<BoxInfo> detectedBoxes;
  ndPtr->detect(roi_image, detectedBoxes);
  ndPtr->postprocess(src, detectedBoxes, rawBoxes, effect_roi, savedPath);
  return rawBoxes;
}
}  // namespace golf