#include "lsq.h"

#include <torch/types.h>

namespace quantops {
namespace ops {
    
namespace detail {

                                   

torch::Tensor lsq_forward_per_tensor(const torch::Tensor& x,
                                     const torch::Tensor& scale,
                                     const torch::Tensor& shift,
                                     const int64_t quant_min,
                                     const int64_t quant_max,
                                     const int64_t type_min,
                                     const int64_t type_max,
                                     const bool use_grad_scaling,
                                     const double grad_scaler,
                                     const bool sym,
                                     const bool eval_mode,
                                     const bool init_mode){
    static auto op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("torchlsq::lsq_forward_per_tensor", "")
          .typed<decltype(lsq_forward_per_tensor)>();
    return op.call(x, scale, shift, quant_min, quant_max, type_min, type_max, 
                   use_grad_scaling, grad_scaler, sym, eval_mode, init_mode);
}                                              


std::tuple<torch::Tensor, 
           torch::Tensor, 
           torch::Tensor> lsq_backward_per_tensor(const torch::Tensor& grad,
                                                  const torch::Tensor& x,
                                                  const torch::Tensor& scale,
                                                  const torch::Tensor& shift,
                                                  const int64_t quant_min,
                                                  const int64_t quant_max,
                                                  const int64_t type_min,
                                                  const int64_t type_max,
                                                  const bool use_grad_scaling,
                                                  const double grad_scaler,
                                                  const bool sym,
                                                  const bool eval_mode,
                                                  const bool init_mode){
    static auto op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("torchlsq::lsq_backward_per_tensor", "")
          .typed<decltype(lsq_backward_per_tensor)>();
    return op.call(grad, x, scale, shift, quant_min, quant_max, type_min, type_max,
                   use_grad_scaling, grad_scaler, sym, eval_mode, init_mode);                                                  
}
    
torch::Tensor lsq_forward_per_channel(const torch::Tensor& x,
                                      const torch::Tensor& scale,
                                      const torch::Tensor& shift,
                                      const int64_t axis,
                                      const int64_t quant_min,
                                      const int64_t quant_max,
                                      const int64_t type_min,
                                      const int64_t type_max,
                                      const bool use_grad_scaling,
                                      const double grad_scaler,
                                      const bool sym,
                                      const bool eval_mode,
                                      const bool init_mode){
    static auto op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("torchlsq::lsq_forward_per_channel", "")
          .typed<decltype(lsq_forward_per_channel)>();
    return op.call(x, scale, shift, axis, quant_min, quant_max, type_min, type_max,
                   use_grad_scaling, grad_scaler, sym, eval_mode, init_mode);                                      
}

std::tuple<torch::Tensor, 
           torch::Tensor, 
           torch::Tensor> lsq_backward_per_channel(const torch::Tensor& grad,
                                                   const torch::Tensor& x,
                                                   const torch::Tensor& scale,
                                                   const torch::Tensor& shift,
                                                   const int64_t axis,
                                                   const int64_t quant_min,
                                                   const int64_t quant_max,
                                                   const int64_t type_min,
                                                   const int64_t type_max,
                                                   const bool use_grad_scaling,
                                                   const double grad_scaler,
                                                   const bool sym,
                                                   const bool eval_mode,
                                                   const bool init_mode){
    static auto op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("torchlsq::lsq_backward_per_channel", "")
          .typed<decltype(lsq_backward_per_channel)>();
    return op.call(grad, x, scale, shift, axis, quant_min, quant_max, type_min, type_max,
                   use_grad_scaling, grad_scaler, sym, eval_mode, init_mode);                                                         
}


} // namespace detail

torch::Tensor lsq(const torch::Tensor& x,
                  const torch::Tensor& scale,
                  const torch::Tensor& shift,
                  const int64_t quant_min,
                  const int64_t quant_max,
                  const int64_t type_min,
                  const int64_t type_max,
                  const int64_t axis,
                  const bool use_grad_scaling,
                  const double grad_scaler,
                  const bool is_affine,
                  const bool is_perchannel,
                  const bool eval_mode,
                  const bool init_mode){
    // we avoid the torch.Scalar type in scale/zero_point in per_tensor case.
    TORCH_CHECK(scale.dim() == 1, "scale should be a 1-D tensor, even in per tensor case(please, avoid torch.Scalar too)");
    TORCH_CHECK(shift.dim() == 1, "shift should be a 1-D tensor, even in per tensor case(please, avoid torch.Scalar too)");

    if (is_perchannel){
        // normalize scale and zero_point shape for per_channel case 
        int64_t size = std::max(scale.size(0), shift.size(0));
        torch::Tensor _scale = (size==scale.size(0))?scale:scale.repeat({size});
        torch::Tensor _shift = (size==shift.size(0))?shift:shift.repeat({size});
        return quantops::ops::detail::lsq_forward_per_channel(x, _scale, _shift,
                                                            axis, quant_min, quant_max, type_min, type_max,
                                                            use_grad_scaling, grad_scaler, !is_affine, eval_mode, init_mode);
    } else {
        return quantops::ops::detail::lsq_forward_per_tensor(x, scale, shift, quant_min, quant_max, type_min, type_max,
                                                           use_grad_scaling, grad_scaler, !is_affine, eval_mode, init_mode);
    }
}

    
TS_TORCH_LIBRARY_FRAGMENT(torchlsq, m) {
    m.def(TORCH_SELECTIVE_SCHEMA(
          "torchlsq::lsq_forward_per_tensor(Tensor x, Tensor scale, Tensor shift, int quant_min, int quant_max, int type_min, int type_max, bool use_grad_scaling, float grad_scaler, bool sym, bool eval_mode, bool init_mode) -> Tensor"));
    m.def(TORCH_SELECTIVE_SCHEMA(
          "torchlsq::lsq_backward_per_tensor(Tensor grad, Tensor x, Tensor scale, Tensor shift, int quant_min, int quant_max, int type_min, int type_max, bool use_grad_scaling, float grad_scaler, bool sym, bool eval_mode, bool init_mode) -> (Tensor, Tensor, Tensor)"));
    m.def(TORCH_SELECTIVE_SCHEMA(
          "torchlsq::lsq_forward_per_channel(Tensor x, Tensor scale, Tensor shift, int axis, int quant_min, int quant_max, int type_min, int type_max, bool use_grad_scaling, float grad_scaler, bool sym, bool eval_mode, bool init_mode) -> Tensor"));
    m.def(TORCH_SELECTIVE_SCHEMA(
          "torchlsq::lsq_backward_per_channel(Tensor grad, Tensor x, Tensor scale, Tensor shift, int axis, int quant_min, int quant_max, int type_min, int type_max, bool use_grad_scaling, float grad_scaler, bool sym, bool eval_mode, bool init_mode) -> (Tensor, Tensor, Tensor)"));
}    
    
} // namespace ops
} // namespace quantops
