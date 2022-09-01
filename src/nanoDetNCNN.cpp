#include "nanoDetNCNN.hpp"

#include "algorithm.hpp"
inline float fast_exp(float x) {
  union {
    uint32_t i;
    float f;
  } v;
  v.i = (1 << 23) * (1.4426950409 * x + 126.93490512f);
  return v.f;
}

inline float sigmoid(float x) { return 1.0f / (1.0f + fast_exp(-x)); }

template <typename _Tp>
int activation_function_softmax(const _Tp *src, _Tp *dst, int length) {
  const _Tp alpha = *std::max_element(src, src + length);
  _Tp denominator{0};

  for (int i = 0; i < length; ++i) {
    dst[i] = fast_exp(src[i] - alpha);
    denominator += dst[i];
  }

  for (int i = 0; i < length; ++i) {
    dst[i] /= denominator;
  }

  return 0;
}

static void generate_grid_center_priors(
    const int input_height, const int input_width, std::vector<int> &strides,
    std::vector<CenterPrior> &center_priors) {
  for (int i = 0; i < (int)strides.size(); i++) {
    int stride = strides[i];
    int feat_w = std::ceil((float)input_width / stride);
    int feat_h = std::ceil((float)input_height / stride);
    for (int y = 0; y < feat_h; y++) {
      for (int x = 0; x < feat_w; x++) {
        CenterPrior ct;
        ct.x = x;
        ct.y = y;
        ct.stride = stride;
        center_priors.push_back(ct);
      }
    }
  }
}

bool NanoDetNCNN::init(Config conf) {
  inputParam = conf;
  this->Net = new ncnn::Net();
  this->Net->load_param(inputParam.NCNNmodelParam.c_str());
  this->Net->load_model(inputParam.NCNNmodelPath.c_str());

  // generate center priors in format of (x, y, stride)
  generate_grid_center_priors(inputParam.inputHeight, inputParam.inputWidth,
                              inputParam.strides, center_priors);

  return true;
}

NanoDetNCNN::~NanoDetNCNN() { delete this->Net; }

int NanoDetNCNN::detect(cv::Mat &raw_image, std::vector<BoxInfo> &result_list) {
  // double start = ncnn::get_current_time();  //ms
  image_h = raw_image.rows;
  image_w = raw_image.cols;

  ncnn::Mat input;
  input = ncnn::Mat::from_pixels(raw_image.data, ncnn::Mat::PIXEL_GRAY2RGB,
                                 image_w, image_h);
  input.substract_mean_normalize(&inputParam.meanVals[0],
                                 &inputParam.normVals[0]);

  ncnn::Extractor ex = this->Net->create_extractor();
  ex.set_light_mode(true);
  ex.set_num_threads(inputParam.numThreads);
  ex.input(input_name.c_str(), input);
  ncnn::Mat out;
  ex.extract(output_name.c_str(), out);

  // get output data
  std::vector<std::vector<BoxInfo>> results;
  results.resize(inputParam.numClass);

  // nchw
  decode_infer(out, center_priors, inputParam.scoreThreshold, results);
  for (int i = 0; i < (int)results.size(); i++) {
    nms(results[i], inputParam.nmsThreshold);
    for (auto box : results[i]) {
      box.x1 = box.x1 / inputParam.inputWidth * image_w;
      box.x2 = box.x2 / inputParam.inputWidth * image_w;
      box.y1 = box.y1 / inputParam.inputHeight * image_h;
      box.y2 = box.y2 / inputParam.inputHeight * image_h;
      result_list.emplace_back(box);
    }
  }

  // double end = ncnn::get_current_time();
  // double time = end - start;
  // std::cout<<"detect time: "<<time<<" ms"<<std::endl;
  return 0;
}

void NanoDetNCNN::decode_infer(ncnn::Mat &pred,
                               std::vector<CenterPrior> center_priors_,
                               float threshold,
                               std::vector<std::vector<BoxInfo>> &results) {
  const int num_points = center_priors_.size();
  for (int idx = 0; idx < num_points; idx++) {
    const int ct_x = center_priors_[idx].x;
    const int ct_y = center_priors_[idx].y;
    const int stride = center_priors_[idx].stride;

    // preds is a tensor with shape [num_points, num_channels]
    const float *scores = pred.row(idx);

    float score = 0;
    int cur_label = 0;
    for (int label = 0; label < inputParam.numClass; label++) {
      if (scores[label] > score) {
        score = scores[label];
        cur_label = label;
      }
    }
    if (score > threshold) {
      const float *bbox_pred = pred.row(idx) + inputParam.numClass;
      results[cur_label].emplace_back(
          disPred2Bbox(bbox_pred, cur_label, score, ct_x, ct_y, stride));
    }
  }
}

BoxInfo NanoDetNCNN::disPred2Bbox(const float *&dfl_det, int label, float score,
                                  int x, int y, int stride) {
  // v2.2 plus
  float ct_x = x * stride;
  float ct_y = y * stride;
  std::vector<float> dis_pred;
  dis_pred.resize(4);
  for (int i = 0; i < 4; i++) {
    float dis = 0;
    float *dis_after_sm = new float[inputParam.regMax + 1];
    activation_function_softmax(dfl_det + i * (inputParam.regMax + 1),
                                dis_after_sm, inputParam.regMax + 1);
    for (int j = 0; j < inputParam.regMax + 1; j++) {
      dis += j * dis_after_sm[j];
    }
    dis *= stride;
    dis_pred[i] = dis;
    delete[] dis_after_sm;
  }
  float xmin = (std::max)(ct_x - dis_pred[0], .0f);
  float ymin = (std::max)(ct_y - dis_pred[1], .0f);
  float xmax = (std::min)(ct_x + dis_pred[2], (float)inputParam.inputWidth);
  float ymax = (std::min)(ct_y + dis_pred[3], (float)inputParam.inputHeight);
  return BoxInfo{xmin, ymin, xmax, ymax, score, label};
}

void NanoDetNCNN::nms(std::vector<BoxInfo> &input_boxes, float NMS_THRESH) {
  std::sort(input_boxes.begin(), input_boxes.end(),
            [](BoxInfo a, BoxInfo b) { return a.score > b.score; });
  std::vector<float> vArea(input_boxes.size());
  for (int i = 0; i < int(input_boxes.size()); ++i) {
    vArea[i] = (input_boxes.at(i).x2 - input_boxes.at(i).x1 + 1) *
               (input_boxes.at(i).y2 - input_boxes.at(i).y1 + 1);
  }
  for (int i = 0; i < int(input_boxes.size()); ++i) {
    for (int j = i + 1; j < int(input_boxes.size());) {
      float xx1 = (std::max)(input_boxes[i].x1, input_boxes[j].x1);
      float yy1 = (std::max)(input_boxes[i].y1, input_boxes[j].y1);
      float xx2 = (std::min)(input_boxes[i].x2, input_boxes[j].x2);
      float yy2 = (std::min)(input_boxes[i].y2, input_boxes[j].y2);
      float w = (std::max)(float(0), xx2 - xx1 + 1);
      float h = (std::max)(float(0), yy2 - yy1 + 1);
      float inter = w * h;
      float ovr = inter / (vArea[i] + vArea[j] - inter);
      if (ovr >= NMS_THRESH) {
        input_boxes.erase(input_boxes.begin() + j);
        vArea.erase(vArea.begin() + j);
      } else {
        j++;
      }
    }
  }
}
