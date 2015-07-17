#include <cuda_runtime.h>

#define COLL_NUM_THREADS 1024

template <typename Dtype>
void multi_gpu_pipeline_bcast(int *my_progress, Dtype* my_data,
    int *next_progress, Dtype* next_data, const int size, const int grid_dim,
    cudaStream_t stream);

template <typename Dtype>
void multi_gpu_pipeline_sum(int *my_progress, Dtype* red_data, Dtype* my_data,
    int *next_progress, Dtype* next_data, Dtype factor, const int size,
    const int grid_dim, cudaStream_t stream);

template <typename Dtype>
void multi_gpu_ring_sum(const int rank, const int nranks, int *my_progress,
    Dtype* red_data, Dtype* my_data, int *next_progress, Dtype* next_red_data,
    Dtype* next_data, Dtype factor, const int size, const int grid_dim,
    cudaStream_t stream);
