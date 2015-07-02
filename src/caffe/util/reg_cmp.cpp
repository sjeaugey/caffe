#include "caffe/common.hpp"
#include "caffe/util/reg_cmp.hpp"

namespace caffe {

template<> 
void caffe_cpu_reg_cmp(const int count, float* data, float* diff, float* hist_data,
    float weight_decay, float lr, float momentum, int reg_type, int update) {
  for (int i=0; i<count; i++) {
    float data_i = reg_type == 1 ? data[i] : ((reg_type == 2) ? (data[i]*weight_decay+diff[i]) : 0);
    hist_data[i] = diff[i] = (data_i*weight_decay+diff[i])*lr + momentum*hist_data[i];
    if (update) data[i] -= diff[i];
  }
}

template<> 
void caffe_cpu_reg_cmp(const int count, double* data, double* diff, double* hist_data,
    double weight_decay, double lr, double momentum, int reg_type, int update) {
  for (int i=0; i<count; i++) {
    double data_i = reg_type == 1 ? data[i] : ((reg_type == 2) ? (data[i]*weight_decay+diff[i]) : 0);
    hist_data[i] = diff[i] = (data_i*weight_decay+diff[i])*lr + momentum*hist_data[i];
    if (update) data[i] -= diff[i];
  }
}
}  // namespace caffe
