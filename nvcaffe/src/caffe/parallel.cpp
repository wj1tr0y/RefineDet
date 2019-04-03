#include <cuda_runtime.h>
#include <glog/logging.h>
#include <boost/thread.hpp>
#include <boost/thread/latch.hpp>

#include "caffe/caffe.hpp"
#include "caffe/parallel.hpp"

#ifdef USE_NCCL
#include "caffe/util/nccl.hpp"
#ifndef NCCL_MAJOR
#error "NCCL_MAJOR is not defined. Please check NCCL installation"
#endif
#define CAFFE_NCCL_VER (NCCL_MAJOR*10000 + NCCL_MINOR*100)
#endif

namespace caffe {

int P2PManager::global_rank_  = 0;
int P2PManager::global_count_ = 1;
char P2PManager::host_name_[_POSIX_HOST_NAME_MAX + 1];

void P2PManager::Init(int *argc, char ***argv) {
#ifdef USE_MPI
  MPI_Init(argc, argv);
//  int provided = 0;
//  MPI_Init_thread(argc, argv, MPI_THREAD_MULTIPLE, &provided);
//  if (provided < MPI_THREAD_MULTIPLE)
//  {std::cerr << "ERROR: The MPI library does not have full thread support" << std::endl;
//    MPI_Abort(MPI_COMM_WORLD, 1);
//  }
  MPI_Comm_rank(MPI_COMM_WORLD, &global_rank_);
  MPI_Comm_size(MPI_COMM_WORLD, &global_count_);
#ifdef DEBUG
//  std::cout << "PID " << getpid() << " on " << caffe::P2PManager::host_name()
//            << " ready for attach" << std::endl;
//  int i = 0;
//  while (0 == i) {
//    sleep(30);
//    MPI_Bcast((void*) &i, sizeof(int), MPI_BYTE, 0, MPI_COMM_WORLD);
//  }
#endif
#endif
  host_name_[_POSIX_HOST_NAME_MAX] = '\0';
  gethostname(host_name_, _POSIX_HOST_NAME_MAX);

  LOG(INFO) << "P2PManager::Init"
#ifdef USE_MPI
            << ", global rank: ["<< P2PManager::global_rank()
            << " of " << P2PManager::global_count() << "]"
#endif
            << " @ " << P2PManager::host_name();
}

P2PManager::P2PManager(shared_ptr<Solver> root_solver,
    int nranks, int devices, const SolverParameter& solver_param) :
      nranks_(nranks),
      syncs_(devices),
      root_solver_(root_solver),
      bar_(devices) {
#ifndef USE_NCCL
  LOG(FATAL) << "USE_NCCL must be specified for multi-GPU mode";
#else
  LOG_IF(FATAL, CAFFE_NCCL_VER < 20200) << "NCCL 2.2 or higher is required";
  if (global_rank_ == 0) {
    NCCL_CHECK(ncclGetUniqueId(&nccl_id_));
  }
#endif
#ifdef USE_MPI
  MPI_Bcast(&nccl_id_, sizeof(nccl_id_), MPI_BYTE, 0, MPI_COMM_WORLD);
#endif
}

P2PManager::~P2PManager() {
#ifdef USE_MPI
  if (P2PManager::global_count() > 1) {  // TODO
    MPI_Finalize();
  }
#endif
}

void P2PManager::Run(const vector<int>& gpus) {
  const int universal_rank_count = (int)gpus.size() * P2PManager::global_count();
#ifdef USE_NCCL
  CHECK_EQ(nranks_, gpus.size() * P2PManager::global_count());
  CHECK_EQ(nranks_, Caffe::solver_count());
#else
  LOG(FATAL) << "Multi-GPU execution not available - rebuild with USE_NCCL";
#endif  // USE_NCCL
  SolverParameter param = root_solver_->param();
  this->shared_ = make_shared<SharedScores<float>>(nranks_);
  for (int i = 0; i < gpus.size(); ++i) {
    param.set_device_id(gpus[i]);
    const int universal_rank = (int)gpus.size() * P2PManager::global_rank() + i;
    LOG(INFO) << "Starting sync " << i << " (of total " << gpus.size() << "), {"
              << universal_rank << "." << universal_rank_count << "}";
    syncs_[i].reset(new P2PSync(this, root_solver_, universal_rank, universal_rank_count, param));
    syncs_[i]->shared_ = this->shared_;
  }

  LOG(INFO) << "Starting Optimization";

  for (int i = 0; i < syncs_.size(); ++i) {
    syncs_[i]->StartInternalThread(true, static_cast<uint64_t>(param.random_seed() +
                                                               P2PManager::global_rank()));
  }
  for (int i = 0; i < syncs_.size(); ++i) {
    syncs_[i]->WaitAll();
  }

  std::ostringstream os;
  os.precision(4);
  float total_perf = this->root_solver_->perf_report(os, syncs_[0]->target_device_);
  if (P2PManager::global_count() > 1) {
    LOG_IF(INFO, P2PManager::global_rank() == 0)
        << "Node {" << P2PManager::global_rank() << "} root " << os.str();
  } else {
    LOG(INFO) << "Root " << os.str();
  }
  for (int i = 1; i < syncs_.size(); ++i) {
    std::ostringstream os;
    os.precision(4);
    total_perf += syncs_[i]->solver_->perf_report(os, syncs_[i]->target_device_, 5 /* "Root " */);
    if (P2PManager::global_count() > 1) {
      LOG_IF(INFO, P2PManager::global_rank() == 0)
          << "Node {" << P2PManager::global_rank() << "} " << os.str();
    } else {
      LOG(INFO) << os.str();
    }
  }

#ifdef USE_MPI
  if (syncs_.size() > 1) {
    LOG(INFO) << "Node {" << P2PManager::global_rank() << "} multi-GPU performance: "
              << total_perf << " img/sec";
  }
  MPI_Allreduce(MPI_IN_PLACE, &total_perf, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
  if (P2PManager::global_rank() == 0) {
    LOG(INFO) << "Overall performance: " << total_perf << " img/sec";
  }
#else
  if (syncs_.size() > 1) {
    LOG(INFO) << "Overall multi-GPU performance: " << total_perf << " img/sec";
  }
#endif
}

void P2PManager::EarlyCancel(P2PSync* killed) {
  for (int i = 0; i < syncs_.size(); ++i) {
    syncs_[i]->solver_->request_early_exit();
    syncs_[i]->StopInternalThread(false);
  }
}

P2PSync::P2PSync(P2PManager* mgr, shared_ptr<Solver> root_solver,
    int rank, int nranks, const SolverParameter& solver_param)
    : InternalThread(solver_param.device_id(), rank, 1, false),
      mgr_(mgr),
      rank_(rank),
      nranks_(nranks),
      initial_iter_(root_solver->iter()),
      solver_(),
      root_solver_(root_solver),
      solver_param_(solver_param) {
#ifndef USE_NCCL
  LOG(FATAL) << "USE_NCCL := 1 must be specified for multi-GPU";
#endif
  CHECK_EQ(target_device_, solver_param_.device_id());
  LOG(INFO) << "[" << rank << " - " << this->target_device_ << "] P2PSync adding callback";
}

P2PSync::~P2PSync() {
#ifdef USE_NCCL
  ncclCommDestroy(nccl_comm_);
#endif
}

void P2PSync::InternalThreadEntry() {
  CHECK_EQ(nranks_, Caffe::solver_count());
  CHECK_EQ(target_device_, Caffe::current_device());
  if (rank_ % (nranks_ / P2PManager::global_count()) == 0) {
    Caffe::set_root_solver(true);
    solver_ = root_solver_;
    solver_->root_add_callback(this);
  } else {
    Caffe::set_root_solver(false);
    solver_.reset(caffe::SolverRegistry::CreateSolver(solver_param_, root_solver_.get(), rank_));
  }
  solver_->set_callback(this);

#ifdef USE_NCCL
#ifdef USE_MPI
  LOG(INFO) << "MPI global: ["<< P2PManager::global_rank()
            << " of " << P2PManager::global_count()
            << "] P2PSync: [" << rank_
            << " of " << nranks_
            << "] @ " << P2PManager::host_name();

  // https://docs.nvidia.com/deeplearning/sdk/nccl-developer-guide/index.html
  //initializing NCCL, group API is required around ncclCommInitRank as it is
  //called across multiple GPUs in each thread/process
  NCCL_CHECK(ncclGroupStart());
  CUDA_CHECK(cudaSetDevice(target_device_));
  NCCL_CHECK(ncclCommInitRank(&nccl_comm_,
                              nranks_,
                              mgr_->nccl_id(),
                              rank_));
  NCCL_CHECK(ncclGroupEnd());
#else
  soft_barrier();
  NCCL_CHECK(ncclCommInitRank(&nccl_comm_,
                              nranks_,
                              mgr_->nccl_id(),
                              rank_));
  soft_barrier();
#endif
#endif

  comm_stream_ = CudaStream::create(true);

  LOG(INFO) << "[" << rank_ << " - " << target_device_ << "] P2PSync added callback";
  // See if there is a defined seed and reset random state if so
  if (solver_->param().random_seed() >= 0) {
    // Fetch random seed and modulate by device ID to make sure
    // everyone doesn't have the same seed.  We seem to have some
    // solver instability if we have everyone with the same seed
    Caffe::set_random_seed(solver_->param().random_seed() + static_cast<uint64_t>(rank_));
  } else {
    // Or system generated one
    Caffe::set_random_seed(Caffe::SEED_NOT_SET);
  }

  if (solver_->Solve()) {
    mgr_->EarlyCancel(this);
  }
}

void P2PSync::soft_barrier() {
  // CPU barrier to avoid busy-polling on the GPU.
  mgr_->bar_wait();
}

void P2PSync::on_start(const vector<shared_ptr<Blob>>& net) {
#ifdef USE_NCCL
  int count = 0;
  NCCL_CHECK(ncclCommCount(nccl_comm_, &count));
  CHECK_EQ(count, nranks_);
  for (int i = 0; i < net.size(); ++i) {
    Blob* param = net[i].get();
    const Type param_type = param->data_type();
    NCCL_CHECK(ncclGroupStart());
    NCCL_CHECK(ncclBcast(param->current_mutable_data_memory(true),
        even(param->count()),
        nccl::nccl_type(param_type),
        0,
        nccl_comm_,
        comm_stream()));
    NCCL_CHECK(ncclGroupEnd());
    CUDA_CHECK(cudaStreamSynchronize(comm_stream()));
  }
#endif  // USE_NCCL
}

void P2PSync::allreduce(int param_id) {
#ifdef USE_NCCL
  const shared_ptr<Blob>& param = solver_->net()->learnable_params()[param_id];
  allreduce_bucket(param->count(), param->current_mutable_diff_memory(true),
                   param->diff_type());
#endif  // USE_NCCL
}

void P2PSync::allreduce_bucket(size_t count, void* bucket, Type type) {
#ifdef USE_NCCL
  CHECK(bucket);
  NCCL_CHECK(ncclAllReduce(bucket,
                           bucket,
                           count,
                           nccl::nccl_type(type),
                           ncclSum,
                           nccl_comm_,
                           comm_stream()));
#endif  // USE_NCCL
}

// master thread gets aggregate of results for output
void P2PSync::aggregateTestResults(float* loss, vector<float>* scores) {
  // only run on master thread
  if (this->rank_ == 0) {
    // initialize results
    *loss = 0.F;
    for (size_t i = 0; i < scores->size(); ++i) {
      (*scores)[i] = 0.F;
    }
    // all test threads
    for (size_t i = 0; i < nranks_; ++i) {
      vector<float>& shared_scr = shared_->rank_scores(this->rank_);
      *loss += shared_scr[0];
      // all scores within each test thread
      for (size_t j = 0; j < scores->size(); ++j) {
        (*scores)[j] += shared_scr[j+1];
      }
    }
  }
}

void P2PSync::saveTestResults(float loss, const vector<float>& scores) {
  vector<float>& shared_scr = shared_->rank_scores(this->rank_);
  CHECK_GE(shared_scr.size(), scores.size() + 1);
  shared_scr[0] = loss;
  for (size_t i = 0; i < scores.size(); ++i) {
    shared_scr[i+1] = scores[i];
  }
}

uint32_t batch_per_gpu(uint32_t total) {
  int device_count = Caffe::device_in_use_per_host_count();
  if (total == 0 || total % device_count != 0) {
    uint32_t new_total = total + (device_count - (total % device_count));
    LOG(WARNING) << "Batch size must be divisible by the number of solvers (GPUs): "
        << "it's been adjusted from " << total << " to " << new_total;
    total = new_total;
  }
  return total / device_count;
}

unsigned int P2PSync::divide_batch_size(NetParameter* net) {
  unsigned int ret = 0U;
  for (int i = 0; i < net->layer_size(); ++i) {
    if (net->layer(i).has_data_param()) {
      if (net->layer(i).data_param().has_batch_size()) {
        uint32_t total = net->layer(i).data_param().batch_size();
        uint32_t batch = batch_per_gpu(total);
        net->mutable_layer(i)->mutable_data_param()->set_batch_size(batch);
        ret = batch;
      }
    }
    if (net->layer(i).has_hdf5_data_param()) {
      if (net->layer(i).hdf5_data_param().has_batch_size()) {
        uint32_t total = net->layer(i).hdf5_data_param().batch_size();
        uint32_t batch = batch_per_gpu(total);
        net->mutable_layer(i)->mutable_hdf5_data_param()->set_batch_size(batch);
        if (ret == 0U) {
          ret = batch;
        }
      }
    }
    if (net->layer(i).has_image_data_param()) {
      if (net->layer(i).image_data_param().has_batch_size()) {
        uint32_t total = net->layer(i).image_data_param().batch_size();
        uint32_t batch = batch_per_gpu(total);
        net->mutable_layer(i)->mutable_image_data_param()->set_batch_size(batch);
        if (ret == 0U) {
          ret = batch;
        }
      }
    }
    if (net->layer(i).has_memory_data_param()) {
      if (net->layer(i).memory_data_param().has_batch_size()) {
        uint32_t total = net->layer(i).memory_data_param().batch_size();
        uint32_t batch = batch_per_gpu(total);
        net->mutable_layer(i)->mutable_memory_data_param()->set_batch_size(batch);
        if (ret == 0U) {
          ret = batch;
        }
      }
    }
    if (net->layer(i).has_window_data_param()) {
      if (net->layer(i).window_data_param().has_batch_size()) {
        uint32_t total = net->layer(i).window_data_param().batch_size();
        uint32_t batch = batch_per_gpu(total);
        net->mutable_layer(i)->mutable_window_data_param()->set_batch_size(batch);
        if (ret == 0U) {
          ret = batch;
        }
      }
    }
  }
  return ret;
}

}  // namespace caffe
