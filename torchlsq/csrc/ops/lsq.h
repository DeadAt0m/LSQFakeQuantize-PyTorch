#pragma once

#include <torch/extension.h>
#include <torch/library.h>
#include "../macros.h"

namespace quantops {
namespace ops {


API_EXPORT torch::Tensor lsq(const torch::Tensor& x,
                             const torch::Tensor& scale,
                             const torch::Tensor& zero_point,
                             const int64_t quant_min,
                             const int64_t quant_max,
                             const int64_t axis,
                             const bool use_grad_scaling,
                             const double grad_scale,
                             const bool is_affine,
                             const bool is_perchannel,
                             const bool do_param_init);



namespace detail {


torch::Tensor lsq_forward_per_tensor(const torch::Tensor& x,
                                     const torch::Tensor& scale,
                                     const torch::Tensor& zero_point,
                                     const int64_t quant_min,
                                     const int64_t quant_max,
                                     const bool use_grad_scaling,
                                     const double grad_scale,
                                     const bool sym,
                                     const bool init_mode);                                                         
std::tuple<torch::Tensor, 
           torch::Tensor, 
           torch::Tensor> lsq_backward_per_tensor(const torch::Tensor& grad,
                                                  const torch::Tensor& x,
                                                  const torch::Tensor& scale,
                                                  const torch::Tensor& zero_point,
                                                  const int64_t quant_min,
                                                  const int64_t quant_max,
                                                  const bool use_grad_scaling,
                                                  const double grad_scale,
                                                  const bool sym,
                                                  const bool init_mode);
    
torch::Tensor lsq_forward_per_channel(const torch::Tensor& x,
                                      const torch::Tensor& scale,
                                      const torch::Tensor& zero_point,
                                      const int64_t axis,
                                      const int64_t quant_min,
                                      const int64_t quant_max,
                                      const bool use_grad_scaling,
                                      const double grad_scale,
                                      const bool sym,
                                      const bool init_mode);

std::tuple<torch::Tensor, 
           torch::Tensor, 
           torch::Tensor> lsq_backward_per_channel(const torch::Tensor& grad,
                                                   const torch::Tensor& x,
                                                   const torch::Tensor& scale,
                                                   const torch::Tensor& zero_point,
                                                   const int64_t axis,
                                                   const int64_t quant_min,
                                                   const int64_t quant_max,
                                                   const bool use_grad_scaling,
                                                   const double grad_scale,
                                                   const bool sym,
                                                   const bool init_mode);


} // namespace detail

} // namespace ops
} // namespace quantops