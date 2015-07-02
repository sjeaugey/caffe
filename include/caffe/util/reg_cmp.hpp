#ifndef CAFFE_UTIL_REG_CMP_H_
#define CAFFE_UTIL_REG_CMP_H_

#ifndef CPU_ONLY
#include <cuda_runtime.h>
#endif

#include "caffe/common.hpp"
#include "caffe/util/device_alternate.hpp"

namespace caffe {

template <typename Dtype>
void caffe_cpu_reg_cmp(const int count, Dtype* data, Dtype* diff, Dtype* hist_data,
    Dtype weight_decay, Dtype lr, Dtype momentum, int reg_type, int update);

#ifndef CPU_ONLY  // GPU

template <typename Dtype>
void caffe_gpu_reg_cmp(const int count, Dtype* data, Dtype* diff, Dtype* hist_data,
    Dtype weight_decay, Dtype lr, Dtype momentum, int reg_type, int update,
    cudaStream_t stream);

#endif  // !CPU_ONLY

}  // namespace caffe


#endif
