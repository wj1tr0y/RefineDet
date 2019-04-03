#include <algorithm>
#include <vector>

#include "caffe/sgd_solvers.hpp"

namespace caffe {

template<typename Gtype, typename Wtype, typename Htype>
void SAG_reg_update_and_clear_gpu(int N,
    Gtype* g, Wtype* w,  Htype* h,
    float momentum, float rate, const std::string& reg_type, float decay,
    void *handle, bool clear_grads);

template<typename Dtype>
void SAGSolver<Dtype>::ComputeUpdateValue(int param_id, void* handle, float rate,
    bool clear_grads) {
  if (this->param_.debug_info()) {
    SGDSolver<Dtype>::PrintParams(param_id);
  }
  Blob* param = this->net_->learnable_params()[param_id].get();
  TBlob<Dtype>* history = this->history_[param_id].get();

  float local_rate = SAGSolver<Dtype>::GetLocalRate(param_id);
  const bool larc = this->param_.larc();
  if (larc) {
    const string& larc_policy = this->param_.larc_policy();
    if (larc_policy == "scale") {
      local_rate = rate * local_rate;
    } else if (larc_policy == "clip") {
      local_rate = std::min(rate, local_rate);
    } else {
      LOG(FATAL) << "Unknown larc policy: " << larc_policy;
    }
  } else {
    local_rate = rate * local_rate;
  }

  // Compute the update to history, then copy it to the parameter diff.
  const float momentum = this->GetMomentum();
  if (Caffe::mode() == Caffe::CPU) {
    caffe_cpu_axpby<Dtype>(param->count(), local_rate, param->cpu_diff<Dtype>(), momentum,
        history->mutable_cpu_data());
    caffe_copy<Dtype>(param->count(), history->cpu_data(), param->mutable_cpu_diff<Dtype>());
    param->Update();
    if (clear_grads) {
      param->set_diff(0.F);
    }
  } else if (Caffe::mode() == Caffe::GPU) {
    const std::string& reg_type = this->param_.regularization_type();
    float decay = SGDSolver<Dtype>::local_decay(param_id);
//    if (this->iter_ <= 1) {
//      caffe_copy<Dtype>(N, param->gpu_diff<Dtype>(), history->mutable_gpu_data());
//      if (reg_type == "L2"){
//        caffe_gpu_axpy<Dtype>(N, Dtype(decay), param->gpu_data<Dtype>(),
//            history->mutable_gpu_data());
//      }
//    }
    const Type wtype = param->data_type();
    const Type gtype = param->diff_type();

    if (gtype == tp<float16>()) {
      SAG_reg_update_and_clear_gpu<float16, Dtype, Dtype>(param->count(),
          param->mutable_gpu_diff<float16>(),
          param->mutable_gpu_data<Dtype>(),
          history->mutable_gpu_data(),
          momentum, local_rate, reg_type, decay,  handle, clear_grads);
    } else if (gtype == tp<float>()) {
      if (wtype == tp<float>()) {
        SAG_reg_update_and_clear_gpu<float, float, Dtype>(param->count(),
            param->mutable_gpu_diff<float>(),
            param->mutable_gpu_data<float>(),
            history->mutable_gpu_data(),
            momentum, local_rate, reg_type, decay, handle, clear_grads);
      } else {
        SAG_reg_update_and_clear_gpu<float, Dtype, Dtype>(param->count(),
            param->mutable_gpu_diff<float>(),
            param->mutable_gpu_data<Dtype>(),
            history->mutable_gpu_data(),
            momentum, local_rate, reg_type, decay, handle, clear_grads);
      }
    } else if (gtype == tp<double>()) {
      SAG_reg_update_and_clear_gpu<double, Dtype, Dtype>(param->count(),
          param->mutable_gpu_diff<double>(),
          param->mutable_gpu_data<Dtype>(),
          history->mutable_gpu_data(),
          momentum, local_rate, reg_type, decay,  handle, clear_grads);
    } else {
      LOG(FATAL) << "Gradient type " << Type_Name(gtype) << " is not supported";
    }
  } else {
    LOG(FATAL) << "Unknown caffe mode: " << Caffe::mode();
  }
}

template<typename Dtype>
float SAGSolver<Dtype>::GetLocalRate(int param_id) {
  const vector<float>& net_params_lr = this->net_->params_lr();
  float local_lr = net_params_lr[param_id];
  if (this->net_->global_grad_scale_enabled() || this->param_.larc()) {
    shared_ptr<Blob> param = this->net_->learnable_params()[param_id];
    TBlob<Dtype>* hist = this->history_[param_id].get();
    const int type_id = this->net_->learnable_types()[0] == param->diff_type() ? 0 : 1;
    if (!is_precise(this->net_->learnable_params()[param_id]->diff_type())) {
      this->net_->update_wgrad_max(this->net_->learnable_params()[param_id].get(), type_id);
    }
    if (this->param_.larc()) {
      float m_norm = std::sqrt(hist->sumsq_data());
      const float wgrad_norm = std::sqrt(param->sumsq_diff(type_id));
      const float w_norm = std::sqrt(param->sumsq_data(type_id));
      const float larc_eta = this->param_.larc_eta();
      float rate = 1.F;
      if (w_norm > 0.F && m_norm > 0.F) {
        rate =  larc_eta * w_norm / m_norm;
      }
      if (local_lr > 0.) {
        local_lr = rate;
      }
      if (this->param_.larc_turbo())  {
        if (this->iter_ > 1) {
          float g_m_dot;
          const int N = param->count();
          caffe_gpu_dot<Dtype>(N, param->gpu_diff<Dtype>(),  hist->gpu_data(), &g_m_dot);
          float g1_go_corr = 0.F;
          if (wgrad_norm > 0.F && m_norm > 0.F) {
             g1_go_corr = g_m_dot / (wgrad_norm * m_norm);
          }
          const float beta =  0.95F;
          this->larc_g_corr_[param_id] =
              beta * this->larc_g_corr_[param_id] + (1.F - beta) * g1_go_corr;
          float boost = 1.0F - 0.9F* this->larc_g_corr_[param_id];
          local_lr = local_lr * boost;
        }
      }// end of turbo
    } // end of larc
  }
  return local_lr;
}

INSTANTIATE_CLASS(SAGSolver);
REGISTER_SOLVER_CLASS(SAG);

}  // namespace caffe
