#include <stdlib.h>
#include "caffe/util/coll.h"

#define SYNC_PERIOD 8

template <typename Dtype>
__global__ void pipeline_bcast_kernel(int* my_progress, Dtype* my_data, 
  int* next_progress, Dtype* next_data, const int size) {
  int tid = threadIdx.x;

  int sync = SYNC_PERIOD;

  // Each block manages its portion of the buffer
  int block_size = (size+gridDim.x-1)/gridDim.x;
  int start = block_size*blockIdx.x;
  int end = block_size*(blockIdx.x+1);
  if (end > size) end = size;
  int progress = (my_progress == NULL) ? end : ((volatile int*)my_progress)[blockIdx.x];
  __threadfence_system();

  for (int index = start + tid;
       index < end;
       index += blockDim.x) {
    if (progress < index) {
      while ((progress = ((volatile int*)my_progress)[blockIdx.x]) < index) {}
      __threadfence_system();
    }
    if (next_data != NULL) {
      next_data[index] = my_data[index]; // Copy data
    }
    if (next_progress != NULL) {
      if (--sync == 0) {
        __syncthreads();
        if (tid == 0) {
          __threadfence_system();
          next_progress[blockIdx.x] = index + blockDim.x; 
        }
        sync = SYNC_PERIOD;
      }
    }
  }
  if (next_progress != NULL) {
    // Don't forget the last update
    __syncthreads();
    if (tid == 0) {
      __threadfence_system();
      next_progress[blockIdx.x] = size;
    }
  }
}

template <typename Dtype>
__global__ void pipeline_sum_kernel(int* my_progress, Dtype* red_data, 
  Dtype* my_data, int* next_progress, Dtype* next_data, Dtype factor,
  const int size) {
  int tid = threadIdx.x;

  int sync = SYNC_PERIOD;

  // Each block manages its portion of the buffer
  int block_size = (size+gridDim.x-1)/gridDim.x;
  int start = block_size*blockIdx.x;
  int end = block_size*(blockIdx.x+1);
  if (end > size) end = size;
  int progress = (my_progress == NULL) ? end : ((volatile int*)my_progress)[blockIdx.x];
  __threadfence_system();

  for (int index = start + tid;
       index < end;
       index += blockDim.x) {
    if (progress < index) {
      while ((progress = ((volatile int*)my_progress)[blockIdx.x]) < index) {}
      __threadfence_system();
    }
    if (my_progress != NULL) {
      // Add it to my data
      red_data[index] += my_data[index];
    }
    if (next_progress != NULL) {
      next_data[index] = red_data[index]; // Send it to next
      if (--sync == 0) {
        __syncthreads();
        if (tid == 0) {
          __threadfence_system();
          next_progress[blockIdx.x] = index + blockDim.x; 
        }
        sync = SYNC_PERIOD;
      }
    } else {
      red_data[index] *= factor;
    }
  }
  if (next_progress != NULL) {
    // Don't forget the last update
    __syncthreads();
    if (tid == 0) {
      __threadfence_system();
      next_progress[blockIdx.x] = size;
    }
  }
}

template <>
void multi_gpu_pipeline_bcast<float>(int *my_progress, float* my_data, int *next_progress, float* next_data, const int size, const int grid_dim, cudaStream_t stream) {
  pipeline_bcast_kernel<float><<<grid_dim, COLL_NUM_THREADS, 0, stream>>>(
      my_progress, my_data, next_progress, next_data, size);
}

template <>
void multi_gpu_pipeline_bcast<double>(int *my_progress, double* my_data, int *next_progress, double* next_data, const int size, const int grid_dim, cudaStream_t stream) {
  pipeline_bcast_kernel<double><<<grid_dim, COLL_NUM_THREADS, 0, stream>>>(
      my_progress, my_data, next_progress, next_data, size);
}

template<>
void multi_gpu_pipeline_sum<float>(int *my_progress, float* red_data, float* my_data, int *next_progress, float* next_data, float factor, const int size, const int grid_dim, cudaStream_t stream) {
  pipeline_sum_kernel<float><<<grid_dim, COLL_NUM_THREADS, 0, stream>>>(
      my_progress, red_data, my_data, next_progress, next_data, factor, size);
}

template<>
void multi_gpu_pipeline_sum<double>(int *my_progress, double* red_data, double* my_data, int *next_progress, double* next_data, double factor, const int size, const int grid_dim, cudaStream_t stream) {
  pipeline_sum_kernel<double><<<grid_dim, COLL_NUM_THREADS, 0, stream>>>(
      my_progress, red_data, my_data, next_progress, next_data, factor, size);
}
