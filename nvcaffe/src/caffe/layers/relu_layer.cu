#include <algorithm>
#include <vector>
#include <device_launch_parameters.h>

#include "caffe/layers/relu_layer.hpp"

namespace caffe {

template <typename Dtype>
__global__ void ReLUForward(const int n, const Dtype* in, Dtype* out, float negative_slope) {
  CUDA_KERNEL_LOOP(index, n) {
    out[index] = !signbit(in[index]) ? in[index] : Dtype(in[index] * negative_slope);
  }
}

template <typename Dtype>
__global__ void ReLUForward0(const int n, const Dtype* in, Dtype* out) {
  CUDA_KERNEL_LOOP(index, n) {
    out[index] = !signbit(in[index]) ? in[index] : Dtype(0);
  }
}

template <typename Ftype, typename Btype>
void ReLULayer<Ftype, Btype>::Forward_gpu(const vector<Blob*>& bottom,
    const vector<Blob*>& top) {
  const Ftype* bottom_data = bottom[0]->gpu_data<Ftype>();
  Ftype* top_data = top[0]->mutable_gpu_data<Ftype>();

  const int count = bottom[0]->count();
  float negative_slope = this->layer_param_.relu_param().negative_slope();
  if (negative_slope != 0.F) {
    // NOLINT_NEXT_LINE(whitespace/operators)
    ReLUForward <<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS, 0, Caffe::thread_stream()>>>(
        count, bottom_data, top_data, negative_slope);
  } else {
    // NOLINT_NEXT_LINE(whitespace/operators)
    ReLUForward0 <<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS, 0, Caffe::thread_stream()>>>(
        count, bottom_data, top_data);
  }
  CUDA_POST_KERNEL_CHECK;
  CUDA_CHECK(cudaStreamSynchronize(Caffe::thread_stream()));
}

template <typename Dtype>
__global__ void ReLUBackward(const int n, const Dtype* in_diff,
    const Dtype* in_data, Dtype* out_diff, float negative_slope) {
  CUDA_KERNEL_LOOP(index, n) {
    out_diff[index] = in_diff[index] * ((in_data[index] > 0)
        + (in_data[index] <= 0) * negative_slope);
  }
}

template <typename Ftype, typename Btype>
void ReLULayer<Ftype, Btype>::Backward_gpu(const vector<Blob*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob*>& bottom) {
  if (propagate_down[0]) {
    const Btype* bottom_data = bottom[0]->gpu_data<Btype>();
    const Btype* top_diff = top[0]->gpu_diff<Btype>();
    Btype* bottom_diff = bottom[0]->mutable_gpu_diff<Btype>();
    const int count = bottom[0]->count();
    float negative_slope = this->layer_param_.relu_param().negative_slope();
    // NOLINT_NEXT_LINE(whitespace/operators)
    ReLUBackward<<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS, 0, Caffe::thread_stream()>>>(
        count, top_diff, bottom_data, bottom_diff, negative_slope);
    CUDA_POST_KERNEL_CHECK;
    CUDA_CHECK(cudaStreamSynchronize(Caffe::thread_stream()));
  }
}

INSTANTIATE_LAYER_GPU_FUNCS_FB(ReLULayer);

}  // namespace caffe
