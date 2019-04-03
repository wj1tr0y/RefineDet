#ifdef USE_TRT

#include <string>
#include <vector>
#include "caffe/layers/trt_layer.hpp"
#include "caffe/solver.hpp"

namespace caffe {

// Logger for TensorRT info/warning/errors
class TRTLogger : public nvinfer1::ILogger {
 public:
  TRTLogger()
      : TRTLogger(Severity::kWARNING) {}
  explicit TRTLogger(Severity severity)
      : reportableSeverity(severity) {}

  void log(Severity severity, const char* msg) override {
    // suppress messages with severity enum value greater than the reportable
    if (severity > reportableSeverity) return;
    switch (severity) {
      case Severity::kINTERNAL_ERROR:
        LOG(FATAL) << msg;
        break;
      case Severity::kERROR:
        LOG(ERROR) << msg;
        break;
      case Severity::kWARNING:
        LOG(WARNING) << msg;
        break;
      case Severity::kINFO:
        LOG(INFO) << msg;
        break;
    }
  }
  Severity reportableSeverity{Severity::kWARNING};
};

static TRTLogger gLogger;

template <typename Ftype, typename Btype>
TRTLayer<Ftype, Btype>::~TRTLayer() {
  if (context_) {
    context_->destroy();
  }
  if (engine_) {
    engine_->destroy();
  }
}

template <typename Ftype, typename Btype>
void TRTLayer<Ftype, Btype>::LayerSetUp(
    const vector<Blob*>& bottom, const vector<Blob*>& top) {
//  batch_size_ = this->layer_param_.trt_param().batch_size();
  top_k_ = this->layer_param_.trt_param().top_k();
  std::string deploy_path = this->layer_param_.trt_param().deploy();
  std::string model_path = this->layer_param_.trt_param().model();
  input_ = this->layer_param_.bottom(0);
  IBuilder* builder = createInferBuilder(gLogger);
  INetworkDefinition* network = builder->createNetwork();
  ICaffeParser* parser = createCaffeParser();
  const IBlobNameToTensor* blobNameToTensor = parser->parse(deploy_path.c_str(),
      model_path.c_str(), *network, DataType::kFLOAT);
  for (int i = 0; i < this->layer_param_.trt_param().deploy_outputs_size(); ++i) {
    network->markOutput(*blobNameToTensor->
        find(this->layer_param_.trt_param().deploy_outputs(i).c_str()));
  }
  output_ = this->layer_param_.trt_param().deploy_outputs(0);
  builder->setMaxBatchSize(batch_size_);
  builder->setMaxWorkspaceSize(1 << 24);

  TRTParameter_TRTMode trt_mode = this->layer_param_.trt_param().trt_mode();
  if (trt_mode == TRTParameter_TRTMode::TRTParameter_TRTMode_INT8) {
    calibrator_ = make_shared<PreGeneratedCalibrator>(
        this->layer_param_.trt_param().calibrator().c_str());
    builder->setInt8Calibrator(calibrator_.get());
    builder->setInt8Mode(true);
  } else if (trt_mode == TRTParameter_TRTMode::TRTParameter_TRTMode_FP16) {
    builder->setHalf2Mode(true);
  }

  engine_ = builder->buildCudaEngine(*network);
  CHECK_NOTNULL(engine_);
  network->destroy();
  parser->destroy();
  builder->destroy();
//  shutdownProtobufLibrary();  //?
  context_ = engine_->createExecutionContext();
  stream_ = Caffe::thread_stream(0);
  CHECK_EQ(2, engine_->getNbBindings());
  inputIndex_ = engine_->getBindingIndex(input_.c_str());
  outputIndex_ = engine_->getBindingIndex(output_.c_str());

  Dims dims = engine_->getBindingDimensions(outputIndex_);
  vector<int> top_shape(dims.nbDims + 1);
  top_shape[0] = batch_size_;
  for (int i = 0; i < dims.nbDims; ++i) {
    top_shape[i + 1] = dims.d[i];
  }
  top[0]->Reshape(top_shape);
  num_labels_ = top_shape[1];
}

template <typename Ftype, typename Btype>
void TRTLayer<Ftype, Btype>::Reshape(const vector<Blob*>& bottom, const vector<Blob*>& top) {
  const Solver* psolver = this->parent_solver();
  if (psolver != nullptr && probs_.count() == 0) {
    int iters = psolver->param().max_iter();
    std::vector<int> probs_shape{num_labels_, iters};
    probs_.Reshape(probs_shape);
  }
  if (probs_.count() > 0) {
    buffers_[outputIndex_] = probs_.mutable_gpu_data() + itr_count_ * num_labels_;
    ++itr_count_;
  }
  CHECK_LE(top_k_, bottom[0]->count() / bottom[1]->count())
    << "top_k must be less than or equal to the number of classes.";
}

template <typename Ftype, typename Btype>
void TRTLayer<Ftype, Btype>::Forward_cpu(const vector<Blob*>& bottom,
    const vector<Blob*>& top) {
  buffers_[inputIndex_] = const_cast<float*>(bottom[0]->gpu_data<float>());
  context_->enqueue(batch_size_, buffers_, stream_, nullptr);
  cudaStreamSynchronize(stream_);
  const Ftype* bottom_label = bottom[1]->cpu_data<Ftype>();
  // NOLINT_NEXT_LINE(runtime/int)
  const long label_value = std::lround(bottom_label[0]);
  labels_.push_back(label_value);
}

template <typename Ftype, typename Btype>
std::string TRTLayer<Ftype, Btype>::print_stats() const {
  // NOLINT_NEXT_LINE(runtime/int)
  std::vector<std::pair<float, long>> prob_vector_(num_labels_);
  std::vector<int> probs_shape = probs_.shape();
  const int iters = probs_shape[1];
  const float* p = probs_.cpu_data();
  int hits1 = 0;
  int hits = 0;
  for (int i = 0; i < iters; ++i) {
    // NOLINT_NEXT_LINE(runtime/int)
    long label_value = labels_[i];

    const float* pi = p + i * num_labels_;
    for (int l = 0; l < num_labels_; ++l) {
      prob_vector_[l] = std::make_pair(pi[l], l);  // TODO labels are 0,1,2,... ?
    }

    std::partial_sort(prob_vector_.begin(), prob_vector_.begin() + top_k_, prob_vector_.end(),
        // NOLINT_NEXT_LINE(runtime/int)
        std::greater<std::pair<float, long>>());

    if (prob_vector_[0].second == label_value) {
      ++hits1;
    }
    for (int k = 0; k < top_k_; ++k) {
      if (prob_vector_[k].second == label_value) {
        ++hits;
        break;
      }
    }
  }

  std::ostringstream os;
  os << std::endl;
  os << "Labels:         " << num_labels_ << std::endl;
  os << "Iterations:     " << iters << std::endl;
  os << "Top 1 hits:     " << hits1 << std::endl;
  os << "Top 1 accuracy: " << (float)hits1 / (float)iters << std::endl;
  if (top_k_ > 1) {
    os << "Top " << top_k_ << " hits:     " << hits << std::endl;
    os << "Top " << top_k_ << " accuracy: " << (float)hits / (float)iters << std::endl;
  }
  return os.str();
}

INSTANTIATE_CLASS_FB(TRTLayer);
REGISTER_LAYER_CLASS(TRT);

}  // namespace caffe

#endif // USE_TRT
