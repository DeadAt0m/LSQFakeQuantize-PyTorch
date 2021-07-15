#if !defined(QUANTOPS_CPU)
#define QUANTOPS_CPU

#include "../lsq.h"

#include <torch/autograd.h>
#include <torch/types.h>

namespace quantops {
namespace ops {

namespace {



class LSQPerTensorFunction : public torch::autograd::Function<LSQPerTensorFunction> {
    public:
        static torch::autograd::variable_list forward(torch::autograd::AutogradContext* ctx,
                                                      const torch::Tensor& input,
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
            at::AutoNonVariableTypeMode g;
            auto output = detail::lsq_forward_per_tensor(input, scale, shift, quant_min, quant_max, type_min, type_max, use_grad_scaling, grad_scaler, sym, eval_mode, init_mode);
            ctx->saved_data["quant_min"] = quant_min;
            ctx->saved_data["quant_max"] = quant_max;
            ctx->saved_data["type_min"] = type_min;
            ctx->saved_data["type_max"] = type_max;
            ctx->saved_data["use_grad_scaling"] = use_grad_scaling;
            ctx->saved_data["grad_scaler"] = grad_scaler;
            ctx->saved_data["sym"] = sym;
            ctx->saved_data["eval_mode"] = eval_mode;
            ctx->saved_data["init_mode"] = init_mode;
            ctx->save_for_backward({input, scale, shift});
            return {output};
        }

        static torch::autograd::variable_list backward(torch::autograd::AutogradContext* ctx,
                                                       const torch::autograd::variable_list& grad_output) {
            auto saved = ctx->get_saved_variables();
            auto input = saved[0];
            auto scale = saved[1];
            auto shift = saved[2];
            auto quant_min = ctx->saved_data["quant_min"].toInt();
            auto quant_max = ctx->saved_data["quant_max"].toInt();
            auto type_min = ctx->saved_data["type_min"].toInt();
            auto type_max = ctx->saved_data["type_max"].toInt();
            auto sym = ctx->saved_data["sym"].toBool();
            auto use_grad_scaling = ctx->saved_data["use_grad_scaling"].toBool();
            auto grad_scaler = ctx->saved_data["grad_scaler"].toDouble();
            auto eval_mode = ctx->saved_data["eval_mode"].toBool();
            auto init_mode = ctx->saved_data["init_mode"].toBool();
            
            auto result = detail::lsq_backward_per_tensor(grad_output[0], input, scale, shift,
                                                          quant_min, quant_max, type_min, type_max, use_grad_scaling, grad_scaler,
                                                          sym, eval_mode, init_mode);
            
            auto grad_input = std::get<0>(result);
            auto grad_scale = std::get<1>(result);
            auto grad_shift = std::get<2>(result);
            return {grad_input, grad_scale, grad_shift,
                    torch::Tensor(), torch::Tensor(), torch::Tensor(), torch::Tensor(), torch::Tensor(),
                    torch::Tensor(), torch::Tensor(), torch::Tensor(), torch::Tensor()};
                
        }
};

// Hack for backward working during dispatch
class LSQPerTensorBackwardFunction: public torch::autograd::Function<LSQPerTensorBackwardFunction> {
    public:
        static torch::autograd::variable_list forward(torch::autograd::AutogradContext* ctx,
                                                      const torch::Tensor& grad,
                                                      const torch::Tensor& input,
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
                                                      const bool init_mode) {
            at::AutoNonVariableTypeMode g;
            auto result = detail::lsq_backward_per_tensor(grad, input, scale, shift,
                                                          quant_min, quant_max, type_min, type_max,
                                                          use_grad_scaling, grad_scaler, sym, eval_mode, init_mode);
            auto grad_input =  std::get<0>(result);
            auto grad_scale =  std::get<1>(result);
            auto grad_shift =  std::get<2>(result);

            return {grad_input, grad_scale, grad_shift};
        }

        static torch::autograd::variable_list backward(torch::autograd::AutogradContext* ctx,
                                                       const torch::autograd::variable_list& grad_output) {
            TORCH_CHECK(0, "double backwards on lsq_per_tensor not supported");
        }
};    


class LSQPerChannelFunction : public torch::autograd::Function<LSQPerChannelFunction> {
    public:
        static torch::autograd::variable_list forward(torch::autograd::AutogradContext* ctx,
                                                      const torch::Tensor& input,
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
            at::AutoNonVariableTypeMode g;
            auto output = detail::lsq_forward_per_channel(input, scale, shift, axis, quant_min, quant_max,
                                                          type_min, type_max, use_grad_scaling, grad_scaler, 
                                                          sym, eval_mode, init_mode);
            ctx->saved_data["axis"] = axis;
            ctx->saved_data["quant_min"] = quant_min;
            ctx->saved_data["quant_max"] = quant_max;
            ctx->saved_data["type_min"] = type_min;
            ctx->saved_data["type_max"] = type_max;
            ctx->saved_data["use_grad_scaling"] = use_grad_scaling;
            ctx->saved_data["grad_scaler"] = grad_scaler;
            ctx->saved_data["sym"] = sym;
            ctx->saved_data["eval_mode"] = eval_mode;
            ctx->saved_data["init_mode"] = init_mode;
            ctx->save_for_backward({input, scale, shift});
            return {output};
        }

        static torch::autograd::variable_list backward(torch::autograd::AutogradContext* ctx,
                                                       const torch::autograd::variable_list& grad_output) {
            auto saved = ctx->get_saved_variables();
            auto input = saved[0];
            auto scale = saved[1];
            auto shift = saved[2];
            auto axis = ctx->saved_data["axis"].toInt();
            auto quant_min = ctx->saved_data["quant_min"].toInt();
            auto quant_max = ctx->saved_data["quant_max"].toInt();
            auto type_min = ctx->saved_data["type_min"].toInt();
            auto type_max = ctx->saved_data["type_max"].toInt();
            auto use_grad_scaling = ctx->saved_data["use_grad_scaling"].toBool();
            auto grad_scaler = ctx->saved_data["grad_scaler"].toDouble();
            auto sym = ctx->saved_data["sym"].toBool();
            auto eval_mode = ctx->saved_data["eval_mode"].toBool();
            auto init_mode = ctx->saved_data["init_mode"].toBool();
            
            auto result = detail::lsq_backward_per_channel(grad_output[0], input, scale, shift,
                                                           axis, quant_min, quant_max, type_min, type_max,
                                                           use_grad_scaling, grad_scaler, 
                                                           sym, eval_mode, init_mode);
            auto grad_input =  std::get<0>(result);
            auto grad_scale =  std::get<1>(result);
            auto grad_shift =  std::get<2>(result);
            return {grad_input, grad_scale, grad_shift, torch::Tensor(), torch::Tensor(), torch::Tensor(), torch::Tensor(),
                    torch::Tensor(), torch::Tensor(), torch::Tensor(), torch::Tensor(), torch::Tensor(), torch::Tensor()};
                
        }
};

// Hack for backward working during dispatch
class LSQPerChannelBackwardFunction: public torch::autograd::Function<LSQPerChannelBackwardFunction> {
    public:
        static torch::autograd::variable_list forward(torch::autograd::AutogradContext* ctx,
                                                      const torch::Tensor& grad,
                                                      const torch::Tensor& input,
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
                                                      const bool init_mode) {
            at::AutoNonVariableTypeMode g;
            auto result = detail::lsq_backward_per_channel(grad, input, scale, shift,
                                                           axis, quant_min, quant_max, type_min, type_max, 
                                                           use_grad_scaling, grad_scaler, 
                                                           sym, eval_mode, init_mode);
            
            auto grad_input =  std::get<0>(result);
            auto grad_scale =  std::get<1>(result);
            auto grad_shift =  std::get<2>(result);

            return {grad_input, grad_scale, grad_shift};
        }

        static torch::autograd::variable_list backward(torch::autograd::AutogradContext* ctx,
                                                      const torch::autograd::variable_list& grad_output) {
            TORCH_CHECK(0, "double backwards on lsq_per_channel not supported");
        }
};


torch::Tensor lsq_per_tensor_autograd(const torch::Tensor& x,
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
                                      const bool init_mode) {
    return LSQPerTensorFunction::apply(x, scale, shift, quant_min, quant_max, type_min, type_max,
                                       use_grad_scaling, grad_scaler, sym, eval_mode, init_mode)[0];
}    
                                                                   
std::tuple<torch::Tensor, 
           torch::Tensor, 
           torch::Tensor> lsq_per_tensor_autograd_backward(const torch::Tensor& grad,
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
    auto result = LSQPerTensorBackwardFunction::apply(grad, x, scale, shift, quant_min, quant_max, type_min, type_max,
                                                      use_grad_scaling, grad_scaler, sym, eval_mode, init_mode);
    return std::make_tuple(result[0], result[1], result[2]);
}  
    
torch::Tensor lsq_per_channel_autograd(const torch::Tensor& x,
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
                                       const bool init_mode) {
    return LSQPerChannelFunction::apply(x, scale, shift, axis, quant_min, quant_max, type_min, type_max, 
                                        use_grad_scaling, grad_scaler, sym, eval_mode, init_mode)[0];
}    
                                                                   
std::tuple<torch::Tensor, 
           torch::Tensor, 
           torch::Tensor> lsq_per_channel_autograd_backward(const torch::Tensor& grad,
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
    auto result = LSQPerChannelBackwardFunction::apply(grad, x, scale, shift, axis, quant_min, quant_max, type_min, type_max,
                                                       use_grad_scaling, grad_scaler, sym, eval_mode, init_mode);
    return std::make_tuple(result[0], result[1], result[2]);
}  
    
       
} // end of anonymous namespace

TORCH_LIBRARY_IMPL(torchlsq, Autograd, m) {
    m.impl(
        TORCH_SELECTIVE_NAME("torchlsq::lsq_forward_per_tensor"),
        TORCH_FN(lsq_per_tensor_autograd));
    m.impl(
        TORCH_SELECTIVE_NAME("torchlsq::lsq_backward_per_tensor"),
        TORCH_FN(lsq_per_tensor_autograd_backward));
    m.impl(
        TORCH_SELECTIVE_NAME("torchlsq::lsq_forward_per_channel"),
        TORCH_FN(lsq_per_channel_autograd));
    m.impl(
        TORCH_SELECTIVE_NAME("torchlsq::lsq_backward_per_channel"),
        TORCH_FN(lsq_per_channel_autograd_backward));
}

} // namespace ops
} // namespace quantops

#endif