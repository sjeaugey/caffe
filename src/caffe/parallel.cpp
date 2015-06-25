#ifndef CPU_ONLY
#include <cuda_runtime.h>
#endif
#include <glog/logging.h>
#include <stdio.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/stat.h>

#include <cstdlib>
#include <sstream>
#include <string>
#include <vector>

#include "boost/thread.hpp"
#include "caffe/caffe.hpp"
#include "caffe/parallel.hpp"
#include "caffe/util/coll.h"

#define GRID_DIM 8

namespace caffe {

enum Op {
  copy,
  replace_cpu,
  replace_gpu,
  replace_cpu_diff,
  replace_gpu_diff
};

template<typename Dtype>
static void apply_buffers(const vector<shared_ptr<Blob<Dtype> > >& blobs,
                          Dtype* buffer, size_t total_size, Op op) {
  Dtype* ptr = buffer;
  for (int i = 0; i < blobs.size(); ++i) {
    int size = blobs[i]->count();
    switch (op) {
      case copy: {
        // Init buffer to current values of blobs
        caffe_copy(size,
                   reinterpret_cast<const Dtype*>(blobs[i]->data()->cpu_data()),
                   ptr);
        break;
      }
      case replace_cpu:
        blobs[i]->data()->set_cpu_data(ptr);
        break;
      case replace_gpu:
        blobs[i]->data()->set_gpu_data(ptr);
        break;
      case replace_cpu_diff:
        blobs[i]->diff()->set_cpu_data(ptr);
        break;
      case replace_gpu_diff:
        blobs[i]->diff()->set_gpu_data(ptr);
        break;
    }
    ptr += size;
  }
  CHECK_EQ(total_size, ptr - buffer);
}

// Buffer size necessary to store given blobs
template<typename Dtype>
static size_t total_size(const vector<shared_ptr<Blob<Dtype> > >& params) {
  size_t size = 0;
  for (int i = 0; i < params.size(); ++i)
    size += params[i]->count();
  return size;
}

template<typename Dtype>
Params<Dtype>::Params(shared_ptr<Solver<Dtype> > root_solver)
    : size_(total_size<Dtype>(root_solver->net()->params())),
      data_(),
      diff_() {
}

template<typename Dtype>
GPUParams<Dtype>::GPUParams(shared_ptr<Solver<Dtype> > root_solver, int device)
    : Params<Dtype>(root_solver) {
#ifndef CPU_ONLY
  int initial_device;
  CUDA_CHECK(cudaGetDevice(&initial_device));

  // Allocate device buffers
  CUDA_CHECK(cudaSetDevice(device));
  CUDA_CHECK(cudaMalloc(&data_, size_ * sizeof(Dtype)));

  // Copy blob values
  const vector<shared_ptr<Blob<Dtype> > >& net = root_solver->net()->params();
  apply_buffers(net, data_, size_, copy);

  CUDA_CHECK(cudaMalloc(&diff_, size_ * sizeof(Dtype)));
  caffe_gpu_set(size_, Dtype(0), diff_);

  CUDA_CHECK(cudaSetDevice(initial_device));
#else
  NO_GPU;
#endif
}

template<typename Dtype>
GPUParams<Dtype>::~GPUParams() {
#ifndef CPU_ONLY
  CUDA_CHECK(cudaFree(data_));
  CUDA_CHECK(cudaFree(diff_));
#endif
}

template<typename Dtype>
void GPUParams<Dtype>::configure(Solver<Dtype>* solver) const {
  const vector<shared_ptr<Blob<Dtype> > >& net = solver->net()->params();
  apply_buffers(net, data_, size_, replace_gpu);
  apply_buffers(net, diff_, size_, replace_gpu_diff);
}

//

void DevicePair::compute(const vector<int> devices, vector<DevicePair>* pairs) {
#ifndef CPU_ONLY
  pairs->push_back(DevicePair(-1, devices[0]));
  for (int i=0; i < devices.size()-1; i++) {
    pairs->push_back(DevicePair(devices[i], devices[i+1]));
  }
#else
  NO_GPU;
#endif
}

//

template<typename Dtype>
P2PSync<Dtype>::P2PSync(shared_ptr<Solver<Dtype> > root_solver,
                        P2PSync<Dtype>* parent, const SolverParameter& param)
    : GPUParams<Dtype>(root_solver, param.device_id()),
      parent_(parent),
      child_(),
      queue_(),
      initial_iter_(root_solver->iter()),
      solver_() {
#ifndef CPU_ONLY
  int initial_device;
  CUDA_CHECK(cudaGetDevice(&initial_device));
  const int self = param.device_id();
  CUDA_CHECK(cudaSetDevice(self));

  if (parent == NULL) {
    solver_ = root_solver;
  } else {
    Caffe::set_root_solver(false);
    solver_.reset(new Solver<Dtype>(param));
    Caffe::set_root_solver(true);
  }
  this->configure(solver_.get());
  solver_->add_callback(this);

  CUDA_CHECK(cudaSetDevice(initial_device));
#else
  NO_GPU;
#endif
}

template<typename Dtype>
void P2PSync<Dtype>::SetupP2PAccess() {
#ifndef CPU_ONLY
  int initial_device;
  CUDA_CHECK(cudaGetDevice(&initial_device));
  const int self = solver_->param().device_id();
  CUDA_CHECK(cudaSetDevice(self));

  cudaStreamCreateWithFlags(&cuda_stream_, cudaStreamNonBlocking);
  //cuda_stream_ = cudaStreamDefault;
  //cudaStreamCreate(&cuda_stream_);

  if (parent_) {
    // Enable p2p access between devices
    const int peer = parent_->solver_->param().device_id();
    int access;
    CUDA_CHECK(cudaDeviceCanAccessPeer(&access, self, peer));
    if (access) {
      CUDA_CHECK(cudaDeviceEnablePeerAccess(peer, 0));
    } else {
      CHECK(false) << "Fatal : P2P access is needed between all GPUs for pipeline";
    }
  }

  CUDA_CHECK(cudaMalloc(&parent_grads_, size_ * sizeof(Dtype)));
  CUDA_CHECK(cudaMalloc(&offset_, GRID_DIM*sizeof(int)));
  CUDA_CHECK(cudaMemset(offset_, -1, GRID_DIM*sizeof(int)));

  if (child_) {
    const int peer = child_->solver_->param().device_id();
    int access;
    CUDA_CHECK(cudaDeviceCanAccessPeer(&access, self, peer));
    if (access) {
      CUDA_CHECK(cudaDeviceEnablePeerAccess(peer, 0));
    } else {
      CHECK(false) << "Fatal : P2P access is needed between all GPUs for pipeline";
    }
  }
  CUDA_CHECK(cudaSetDevice(initial_device));
#else
  NO_GPU;
#endif
}

template<typename Dtype>
P2PSync<Dtype>::~P2PSync() {
#ifndef CPU_ONLY
  int initial_device;
  CUDA_CHECK(cudaGetDevice(&initial_device));
  const int self = solver_->param().device_id();
  CUDA_CHECK(cudaSetDevice(self));

  cudaStreamDestroy(cuda_stream_);
  if (child_) {
    const int peer = child_->solver_->param().device_id();
    int access;
    CUDA_CHECK(cudaDeviceCanAccessPeer(&access, self, peer));
    if (access) {
      CUDA_CHECK(cudaDeviceDisablePeerAccess(peer));
    }
  }
  if (parent_) {
    const int peer = parent_->solver_->param().device_id();
    int access;
    CUDA_CHECK(cudaDeviceCanAccessPeer(&access, self, peer));
    if (access) {
      CUDA_CHECK(cudaDeviceDisablePeerAccess(peer));
    }
  }

  CUDA_CHECK(cudaSetDevice(initial_device));
#endif
}

template<typename Dtype>
void P2PSync<Dtype>::InternalThreadEntry() {
  Caffe::SetDevice(solver_->param().device_id());
  CHECK(Caffe::root_solver());
  Caffe::set_root_solver(false);
  // See if there is a defined seed and reset random state if so
  if (solver_->param().random_seed() >= 0) {
    // Fetch random seed and modulate by device ID to make sure
    // everyone doesn't have the same seed.  We seem to have some
    // solver instability if we have everyone with the same seed
    Caffe::set_random_seed(
        solver_->param().random_seed() + solver_->param().device_id());
  }
  solver_->Step(solver_->param().max_iter() - initial_iter_);
}

template<typename Dtype>
void P2PSync<Dtype>::soft_barrier() {
  // CPU barrier to avoid busy-polling on the GPU.
  if (child_) queue_.pop();
  if (parent_) parent_->queue_.push(this);
  if (parent_) queue_.pop();
  if (child_) child_->queue_.push(this);
}

template<typename Dtype>
void P2PSync<Dtype>::on_start(Timer* timer, ostringstream* timing) {
#ifndef CPU_ONLY
  CUDA_CHECK(cudaStreamSynchronize(cudaStreamDefault));
  multi_gpu_pipeline_bcast(
    parent_ ? offset_ : NULL,
    data_,
    child_ ? child_->offset_ : NULL,
    child_ ? child_->data_ : NULL,
    size_,
    GRID_DIM,
    cuda_stream_);
  CUDA_CHECK(cudaStreamSynchronize(cuda_stream_));
#endif
}

template<typename Dtype>
void P2PSync<Dtype>::on_gradients_ready(Timer* timer, ostringstream* timing) {
#ifndef CPU_ONLY
  CUDA_CHECK(cudaStreamSynchronize(cudaStreamDefault));
  multi_gpu_pipeline_sum(
    child_ ? offset_ : NULL,
    diff_,
    parent_grads_,
    parent_ ? parent_->offset_ : NULL,
    parent_ ? parent_->parent_grads_ : NULL,
    (Dtype)1.0 / Caffe::solver_count(), // Mult factor
    size_,
    GRID_DIM,
    cuda_stream_);
  CUDA_CHECK(cudaStreamSynchronize(cuda_stream_));
#endif
}

template<typename Dtype>
void P2PSync<Dtype>::run(shared_ptr<Solver<Dtype> > root,
                         const vector<int>& gpus) {
  int nranks = gpus.size();
  SolverParameter param(root->param());
  vector<shared_ptr<P2PSync<Dtype> > > syncs(nranks);
  syncs[0].reset(new P2PSync<Dtype>(root, NULL, param));
  for (int i = 1; i < nranks; i++) {
    param.set_device_id(gpus[i]);
    // Set parent_/child_
    syncs[i].reset(new P2PSync<Dtype>(root, syncs[i-1].get(), param));
    syncs[i-1].get()->child_=syncs[i].get();
  }
  for (int i = 0; i < syncs.size(); ++i) {
    syncs[i]->SetupP2PAccess();
  }

  LOG(INFO)<< "Starting Optimization";

  for (int i = 1; i < syncs.size(); ++i) {
    syncs[i]->StartInternalThread();
  }

  // Run root solver on current thread
  syncs[0]->solver_->Solve();

  for (int i = 1; i < syncs.size(); ++i) {
    syncs[i]->StopInternalThread();
  }
}

template<typename Dtype>
void P2PSync<Dtype>::divide_batch_size(NetParameter* net) {
  int solver_count = Caffe::solver_count();
  for (int i = 0; i < net->layer_size(); ++i) {
    string m = "Batch size must be divisible by the number of solvers (GPUs)";
    if (net->layer(i).has_data_param()) {
      if (net->layer(i).data_param().has_batch_size()) {
        uint32_t total = net->layer(i).data_param().batch_size();
        uint32_t batch = total / solver_count;
        CHECK(batch * solver_count == total) << m;
        net->mutable_layer(i)->mutable_data_param()->set_batch_size(batch);

        // Also adjust the prefetch count, as it is shared by all solvers
        uint32_t prefetch = net->layer(i).data_param().prefetch();
        net->mutable_layer(i)->mutable_data_param()->set_prefetch(
            prefetch * solver_count);
      }
    }
    if (net->layer(i).has_hdf5_data_param()) {
      if (net->layer(i).hdf5_data_param().has_batch_size()) {
        uint32_t total = net->layer(i).hdf5_data_param().batch_size();
        uint32_t batch = total / solver_count;
        CHECK(batch * solver_count == total) << m;
        net->mutable_layer(i)->mutable_hdf5_data_param()->set_batch_size(batch);
      }
    }
    if (net->layer(i).has_image_data_param()) {
      if (net->layer(i).image_data_param().has_batch_size()) {
        uint32_t total = net->layer(i).image_data_param().batch_size();
        uint32_t batch = total / solver_count;
        CHECK(batch * solver_count == total) << m;
        net->mutable_layer(i)->mutable_image_data_param()->set_batch_size(
            batch);
      }
    }
    if (net->layer(i).has_memory_data_param()) {
      if (net->layer(i).memory_data_param().has_batch_size()) {
        uint32_t total = net->layer(i).memory_data_param().batch_size();
        uint32_t batch = total / solver_count;
        CHECK(batch * solver_count == total) << m;
        net->mutable_layer(i)->mutable_memory_data_param()->set_batch_size(
            batch);
      }
    }
    if (net->layer(i).has_window_data_param()) {
      if (net->layer(i).window_data_param().has_batch_size()) {
        uint32_t total = net->layer(i).window_data_param().batch_size();
        uint32_t batch = total / solver_count;
        CHECK(batch * solver_count == total) << m;
        net->mutable_layer(i)->mutable_window_data_param()->set_batch_size(
            batch);
      }
    }
  }
}

INSTANTIATE_CLASS(Params);
INSTANTIATE_CLASS(GPUParams);
INSTANTIATE_CLASS(P2PSync);

}  // namespace caffe

