#include <algorithm>
#include <device_launch_parameters.h>

#include "caffe/layers/softmax_layer.hpp"

namespace caffe {

template <typename Dtype, typename Mtype>
__global__ void kernel_channel_max(const int num, const int channels,
    const int spatial_dim, const Dtype* data, Mtype* out) {
  CUDA_KERNEL_LOOP(index, num * spatial_dim) {
    int n = index / spatial_dim;
    int s = index % spatial_dim;
    Mtype maxval = -max_dtype<Mtype>();
    for (int c = 0; c < channels; ++c) {
      maxval = max((Mtype)data[(n * channels + c) * spatial_dim + s], maxval);
    }
    out[index] = maxval;
  }
}

template <typename Dtype, typename Mtype>
__global__ void kernel_channel_subtract(const int count,
    const int num, const int channels,
    const int spatial_dim, const Dtype* channel_max, Mtype* data) {
  CUDA_KERNEL_LOOP(index, count) {
    int n = index / channels / spatial_dim;
    int s = index % spatial_dim;
    data[index] -= (Mtype)channel_max[n * spatial_dim + s];
  }
}

template <typename Dtype, typename Mtype>
__global__ void kernel_exp(const int count, const Dtype* data, Mtype* out) {
  CUDA_KERNEL_LOOP(index, count) {
    out[index] = exp((Mtype)data[index]);
  }
}

template <typename Dtype, typename Mtype>
__global__ void kernel_channel_sum(const int num, const int channels,
    const int spatial_dim, const Dtype* data, Mtype* channel_sum) {
  CUDA_KERNEL_LOOP(index, num * spatial_dim) {
    int n = index / spatial_dim;
    int s = index % spatial_dim;
    Mtype sum = 0.F;
    for (int c = 0; c < channels; ++c) {
      sum += (Mtype)data[(n * channels + c) * spatial_dim + s];
    }
    channel_sum[index] = sum;
  }
}

template <typename Dtype, typename Mtype>
__global__ void kernel_channel_div(const int count,
    const int num, const int channels,
    const int spatial_dim, const Dtype* channel_sum, Mtype* data) {
  CUDA_KERNEL_LOOP(index, count) {
    int n = index / channels / spatial_dim;
    int s = index % spatial_dim;
    Dtype cs = channel_sum[n * spatial_dim + s];
    data[index] /= (Mtype)(cs > min_dtype<Dtype>() || cs < - min_dtype<Dtype>() ?
        cs : min_dtype<Dtype>());
  }
}

template <typename Dtype, typename Mtype>
__global__ void kernel_channel_dot(const int num, const int channels,
    const int spatial_dim, const Dtype* data_1, const Dtype* data_2,
    Mtype* channel_dot) {
  CUDA_KERNEL_LOOP(index, num * spatial_dim) {
    int n = index / spatial_dim;
    int s = index % spatial_dim;
    Mtype dot = 0.F;
    for (int c = 0; c < channels; ++c) {
      dot += (Mtype)data_1[(n * channels + c) * spatial_dim + s]
           * (Mtype)data_2[(n * channels + c) * spatial_dim + s];
    }
    channel_dot[index] = dot;
  }
}

template <typename Ftype, typename Btype>
void SoftmaxLayer<Ftype, Btype>::Forward_gpu(const vector<Blob*>& bottom,
    const vector<Blob*>& top) {
  const Ftype* bottom_data = bottom[0]->gpu_data<Ftype>();
  Ftype* top_data = top[0]->mutable_gpu_data<Ftype>();
  Ftype* scale_data = scale_.mutable_gpu_data();
  int count = bottom[0]->count();
  int channels = top[0]->shape(softmax_axis_);
  caffe_copy(count, bottom_data, top_data);
  cudaStream_t stream = Caffe::thread_stream();
  // We need to subtract the max to avoid numerical issues, compute the exp,
  // and then normalize.
  // compute max
  // NOLINT_NEXT_LINE(whitespace/operators)
  kernel_channel_max<<<CAFFE_GET_BLOCKS(outer_num_ * inner_num_),
      CAFFE_CUDA_NUM_THREADS, 0, stream>>>(outer_num_, channels, inner_num_, top_data,
      scale_data);
  // subtract
  // NOLINT_NEXT_LINE(whitespace/operators)
  kernel_channel_subtract<<<CAFFE_GET_BLOCKS(count),
      CAFFE_CUDA_NUM_THREADS, 0, stream>>>(count, outer_num_, channels, inner_num_,
      scale_data, top_data);
  // exponentiate
  // NOLINT_NEXT_LINE(whitespace/operators)
  kernel_exp<<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS, 0, stream>>>(
      count, top_data, top_data);
  // sum after exp
  // NOLINT_NEXT_LINE(whitespace/operators)
  kernel_channel_sum<<<CAFFE_GET_BLOCKS(outer_num_ * inner_num_),
      CAFFE_CUDA_NUM_THREADS, 0, stream>>>(outer_num_, channels, inner_num_, top_data,
      scale_data);
  // divide
  // NOLINT_NEXT_LINE(whitespace/operators)
  kernel_channel_div<<<CAFFE_GET_BLOCKS(count),
      CAFFE_CUDA_NUM_THREADS, 0, stream>>>(count, outer_num_, channels, inner_num_,
      scale_data, top_data);
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

template <typename Ftype, typename Btype>
void SoftmaxLayer<Ftype, Btype>::Backward_gpu(const vector<Blob*>& top,
    const vector<bool>& propagate_down, const vector<Blob*>& bottom) {
  const Btype* top_diff = top[0]->gpu_diff<Btype>();
  const Btype* top_data = top[0]->gpu_data<Btype>();
  Btype* bottom_diff = bottom[0]->mutable_gpu_diff<Btype>();
  Ftype* scale_data = scale_.mutable_gpu_data();
  int count = top[0]->count();
  int channels = top[0]->shape(softmax_axis_);
  caffe_copy(count, top_diff, bottom_diff);
  cudaStream_t stream = Caffe::thread_stream();
  // Compute inner1d(top_diff, top_data) and subtract them from the bottom diff.
  // NOLINT_NEXT_LINE(whitespace/operators)
  kernel_channel_dot<<<CAFFE_GET_BLOCKS(outer_num_ * inner_num_),
      CAFFE_CUDA_NUM_THREADS, 0, stream>>>(outer_num_, channels, inner_num_,
      top_diff, top_data, scale_data);
  // NOLINT_NEXT_LINE(whitespace/operators)
  kernel_channel_subtract<<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS, 0, stream>>>(
      count, outer_num_, channels, inner_num_, scale_data, bottom_diff);
  CUDA_CHECK(cudaStreamSynchronize(stream));
  // elementwise multiplication
  caffe_gpu_mul(top[0]->count(), bottom_diff, top_data, bottom_diff);
}

INSTANTIATE_LAYER_GPU_FUNCS_FB(SoftmaxLayer);


}  // namespace caffe
