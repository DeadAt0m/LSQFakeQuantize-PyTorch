#include "lsq.h"

#include <torch/types.h>

namespace quantops {
namespace ops {
    
namespace detail {


torch::Tensor lsq_forward_per_tensor(const torch::Tensor& x,
                                     const torch::Tensor& scale,
                                     const torch::Tensor& zero_point,
                                     const int64_t quant_min,
                                     const int64_t quant_max,
                                     const bool use_grad_scaling,
                                     const double grad_scaler,
                                     const bool sym,
                                     const bool init_mode){
    static auto op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("torchlsq::lsq_forward_per_tensor", "")
          .typed<decltype(lsq_forward_per_tensor)>();
    return op.call(x, scale, zero_point, quant_min, quant_max, use_grad_scaling, grad_scaler, sym, init_mode);
}                                              


std::tuple<torch::Tensor, 
           torch::Tensor, 
           torch::Tensor> lsq_backward_per_tensor(const torch::Tensor& grad,
                                                  const torch::Tensor& x,
                                                  const torch::Tensor& scale,
                                                  const torch::Tensor& zero_point,
                                                  const int64_t quant_min,
                                                  const int64_t quant_max,
                                                  const bool use_grad_scaling,
                                                  const double grad_scaler,
                                                  const bool sym,
                                                  const bool init_mode){
    static auto op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("torchlsq::lsq_backward_per_tensor", "")
          .typed<decltype(lsq_backward_per_tensor)>();
    return op.call(grad, x, scale, zero_point, quant_min, quant_max, use_grad_scaling, grad_scaler, sym, init_mode);                                                  
}
    
torch::Tensor lsq_forward_per_channel(const torch::Tensor& x,
                                      const torch::Tensor& scale,
                                      const torch::Tensor& zero_point,
                                      const int64_t axis,
                                      const int64_t quant_min,
                                      const int64_t quant_max,
                                      const bool use_grad_scaling,
                                      const double grad_scaler,
                                      const bool sym,
                                      const bool init_mode){
    static auto op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("torchlsq::lsq_forward_per_channel", "")
          .typed<decltype(lsq_forward_per_channel)>();
    return op.call(x, scale, zero_point, axis, quant_min, quant_max, use_grad_scaling, grad_scaler, sym, init_mode);                                      
}

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
                                                   const double grad_scaler,
                                                   const bool sym,
                                                   const bool init_mode){
    static auto op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("torchlsq::lsq_backward_per_channel", "")
          .typed<decltype(lsq_backward_per_channel)>();
    return op.call(grad, x, scale, zero_point, axis, quant_min, quant_max, use_grad_scaling, grad_scaler, sym, init_mode);                                                         
}


} // namespace detail

torch::Tensor lsq(const torch::Tensor& x,
                  const torch::Tensor& scale,
                  const torch::Tensor& zero_point,
                  const int64_t quant_min,
                  const int64_t quant_max,
                  const int64_t axis,
                  const bool use_grad_scaling,
                  const double grad_scaler,
                  const bool is_affine,
                  const bool is_perchannel,
                  const bool do_param_init){
    // we avoid the torch.Scalar type in scale/zero_point in per_tensor case.
    TORCH_CHECK(scale.dim() == 1, "scale should be a 1-D tensor, even in per tensor case(please, avoid torch.Scalar too)");
    TORCH_CHECK(zero_point.dim() == 1, "zero point should be a 1-D tensor, even in per tensor case(please, avoid torch.Scalar too)");

    if (is_perchannel){
        // normalize scale and zero_point shape for per_channel case 
        int64_t size = std::max(scale.size(0),zero_point.size(0));
        torch::Tensor _scale = (size==scale.size(0))?scale:scale.repeat({size});
        torch::Tensor _zero_point = (size==zero_point.size(0))?zero_point:zero_point.repeat({size});
        return quantops::ops::detail::lsq_forward_per_channel(x, _scale, _zero_point,
                                                         axis, quant_min, quant_max, use_grad_scaling, grad_scaler, !is_affine, do_param_init);
    } else {
        return quantops::ops::detail::lsq_forward_per_tensor(x, scale, zero_point, quant_min, quant_max, use_grad_scaling, grad_scaler, !is_affine, do_param_init);
    }
}

    
TS_TORCH_LIBRARY_FRAGMENT(torchlsq, m) {
    m.def(TORCH_SELECTIVE_SCHEMA(
          "torchlsq::lsq_forward_per_tensor(Tensor x, Tensor scale, Tensor zero_point, int quant_min, int quant_max, bool use_grad_scaling, float grad_scaler, bool sym, bool init_mode) -> Tensor"));
    m.def(TORCH_SELECTIVE_SCHEMA(
          "torchlsq::lsq_backward_per_tensor(Tensor grad, Tensor x, Tensor scale, Tensor zero_point, int quant_min, int quant_max, bool use_grad_scaling, float grad_scaler, bool sym, bool init_mode) -> (Tensor, Tensor, Tensor)"));
    m.def(TORCH_SELECTIVE_SCHEMA(
          "torchlsq::lsq_forward_per_channel(Tensor x, Tensor scale, Tensor zero_point, int axis, int quant_min, int quant_max, bool use_grad_scaling, float grad_scaler, bool sym, bool init_mode) -> Tensor"));
    m.def(TORCH_SELECTIVE_SCHEMA(
          "torchlsq::lsq_backward_per_channel(Tensor grad, Tensor x, Tensor scale, Tensor zero_point, int axis, int quant_min, int quant_max, bool use_grad_scaling, float grad_scaler, bool sym, bool init_mode) -> (Tensor, Tensor, Tensor)"));
}    
    
    
} // namespace ops
} // namespace quantops