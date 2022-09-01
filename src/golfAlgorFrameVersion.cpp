#include "golfAlgorFrameVersion.hpp"
namespace golf {
std::string getGolfAlgorFrameVersion() {
#ifdef USE_MNN
  std::string str = "MNN";
#else
  std::string str = "NCNN";
#endif
  return str;
}
}  // namespace golf