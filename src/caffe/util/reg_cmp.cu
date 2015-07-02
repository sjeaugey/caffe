#include "caffe/common.hpp"
#include "caffe/util/reg_cmp.hpp"

namespace caffe {

template <typename Dtype>
__global__ void reg_cmp_L1_kernel(const int count, Dtype* data, Dtype* diff, Dtype* hist_data,
    Dtype weight_decay, Dtype lr, Dtype momentum) {
  CUDA_KERNEL_LOOP(i, count) {
    hist_data[i] = diff[i] = ((data[i] > Dtype(0) - data[i] < Dtype(0))*weight_decay+diff[i])*lr + momentum*hist_data[i];
  }
}

template <typename Dtype>
__global__ void reg_cmp_L2_kernel(const int count, Dtype* data, Dtype* diff, Dtype* hist_data,
    Dtype weight_decay, Dtype lr, Dtype momentum) {
  CUDA_KERNEL_LOOP(i, count) {
    hist_data[i] = diff[i] = (data[i]*weight_decay+diff[i])*lr + momentum*hist_data[i];
  }
}

template <typename Dtype>
__global__ void reg_cmp_upd_L1_kernel(const int count, Dtype* data, Dtype* diff, Dtype* hist_data,
    Dtype weight_decay, Dtype lr, Dtype momentum) {
  CUDA_KERNEL_LOOP(i, count) {
    hist_data[i] = diff[i] = ((data[i] > Dtype(0) - data[i] < Dtype(0))*weight_decay+diff[i])*lr + momentum*hist_data[i];
    data[i] -= diff[i];
  }
}

template <typename Dtype>
__global__ void reg_cmp_upd_L2_kernel(const int count, Dtype* data, Dtype* diff, Dtype* hist_data,
    Dtype weight_decay, Dtype lr, Dtype momentum) {
  CUDA_KERNEL_LOOP(i, count) {
    hist_data[i] = diff[i] = (data[i]*weight_decay+diff[i])*lr + momentum*hist_data[i];
    data[i] -= diff[i];
  }
}

template<> 
void caffe_gpu_reg_cmp(const int count, float* data, float* diff, float* hist_data,
    float weight_decay, float lr, float momentum, int reg_type, int update,
    cudaStream_t stream) {
  if (update == 0) {
    if (reg_type == 1) {
      reg_cmp_L1_kernel<float><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS, 0, stream>>>(
          count, data, diff, hist_data, weight_decay, lr, momentum);
    } else if (reg_type == 2) {
      reg_cmp_L2_kernel<float><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS, 0, stream>>>(
          count, data, diff, hist_data, weight_decay, lr, momentum);
    }
  } else {
    if (reg_type == 1) {
      reg_cmp_upd_L1_kernel<float><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS, 0, stream>>>(
          count, data, diff, hist_data, weight_decay, lr, momentum);
    } else if (reg_type == 2) {
      reg_cmp_upd_L2_kernel<float><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS, 0, stream>>>(
          count, data, diff, hist_data, weight_decay, lr, momentum);
    }
  }
}

template<> 
void caffe_gpu_reg_cmp(const int count, double* data, double* diff, double* hist_data,
    double weight_decay, double lr, double momentum, int reg_type, int update,
    cudaStream_t stream) {
  if (update ==0) {
    if (reg_type == 1) {
      reg_cmp_L1_kernel<double><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS, 0, stream>>>(
          count, data, diff, hist_data, weight_decay, lr, momentum);
    } else if (reg_type == 2) {
      reg_cmp_L2_kernel<double><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS, 0, stream>>>(
          count, data, diff, hist_data, weight_decay, lr, momentum);
    }
  } else {
    if (reg_type == 1) {
      reg_cmp_upd_L1_kernel<double><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS, 0, stream>>>(
          count, data, diff, hist_data, weight_decay, lr, momentum);
    } else if (reg_type == 2) {
      reg_cmp_upd_L2_kernel<double><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS, 0, stream>>>(
          count, data, diff, hist_data, weight_decay, lr, momentum);
    }
  }
}

}  // namespace caffe
