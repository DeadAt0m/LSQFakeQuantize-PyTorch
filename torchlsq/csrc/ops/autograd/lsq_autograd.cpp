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
                                                      const torch::Tensor& zero_point,
                                                      const int64_t quant_min,
                                                      const int64_t quant_max,
                                                      const bool use_grad_scaling,
                                                      const double grad_scaler,
                                                      const bool sym,
                                                      const bool init_mode){
            at::AutoNonVariableTypeMode g;
            auto output = detail::lsq_forward_per_tensor(input, scale, zero_point, quant_min, quant_max, use_grad_scaling, grad_scaler, sym, init_mode);
            ctx->saved_data["quant_min"] = quant_min;
            ctx->saved_data["quant_max"] = quant_max;
            ctx->saved_data["use_grad_scaling"] = use_grad_scaling;
            ctx->saved_data["grad_scaler"] = grad_scaler;
            ctx->saved_data["sym"] = sym;
            ctx->saved_data["init_mode"] = init_mode;
            ctx->save_for_backward({input, scale, zero_point});
            return {output};
        }

        static torch::autograd::variable_list backward(torch::autograd::AutogradContext* ctx,
                                                       const torch::autograd::variable_list& grad_output) {
            auto saved = ctx->get_saved_variables();
            auto input = saved[0];
            auto scale = saved[1];
            auto zero_point = saved[2];
            auto quant_min = ctx->saved_data["quant_min"].toInt();
            auto quant_max = ctx->saved_data["quant_max"].toInt();
            auto sym = ctx->saved_data["sym"].toBool();
            auto use_grad_scaling = ctx->saved_data["use_grad_scaling"].toBool();
            auto grad_scaler = ctx->saved_data["grad_scaler"].toDouble();
            auto init_mode = ctx->saved_data["init_mode"].toBool();
            
            auto result = detail::lsq_backward_per_tensor(grad_output[0], input, scale, zero_point,
                                                          quant_min, quant_max, use_grad_scaling, grad_scaler, sym, init_mode);
            auto grad_input =  std::get<0>(result);
            auto grad_scale =  std::get<1>(result);
            auto grad_zero_point = std::get<2>(result);
            return {grad_input, grad_scale, grad_zero_point, torch::Tensor(), torch::Tensor(), torch::Tensor(), torch::Tensor(), torch::Tensor(), torch::Tensor()};
                
        }
};

// Hack for backward working during dispatch
class LSQPerTensorBackwardFunction: public torch::autograd::Function<LSQPerTensorBackwardFunction> {
    public:
        static torch::autograd::variable_list forward(torch::autograd::AutogradContext* ctx,
                                                      const torch::Tensor& grad,
                                                      const torch::Tensor& input,
                                                      const torch::Tensor& scale,
                                                      const torch::Tensor& zero_point,
                                                      const int64_t quant_min,
                                                      const int64_t quant_max,
                                                      const bool use_grad_scaling,
                                                      const double grad_scaler,
                                                      const bool sym,
                                                      const bool init_mode) {
            at::AutoNonVariableTypeMode g;
            auto result = detail::lsq_backward_per_tensor(grad, input, scale, zero_point,
                                                          quant_min, quant_max, use_grad_scaling, grad_scaler, sym, init_mode);
            auto grad_input =  std::get<0>(result);
            auto grad_scale =  std::get<1>(result);
            auto grad_zero_point = std::get<2>(result);

            return {grad_input, grad_scale, grad_zero_point};
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
                                                      const torch::Tensor& zero_point,
                                                      const int64_t axis,
                                                      const int64_t quant_min,
                                                      const int64_t quant_max,
                                                      const bool use_grad_scaling,
                                                      const double grad_scaler,
                                                      const bool sym,
                                                      const bool init_mode){
            at::AutoNonVariableTypeMode g;
            auto output = detail::lsq_forward_per_channel(input, scale, zero_point, axis, quant_min, quant_max, use_grad_scaling, grad_scaler, sym, init_mode);
            ctx->saved_data["axis"] = axis;
            ctx->saved_data["quant_min"] = quant_min;
            ctx->saved_data["quant_max"] = quant_max;
            ctx->saved_data["use_grad_scaling"] = use_grad_scaling;
            ctx->saved_data["grad_scaler"] = grad_scaler;
            ctx->saved_data["sym"] = sym;
            ctx->saved_data["init_mode"] = init_mode;
            ctx->save_for_backward({input, scale, zero_point});
            return {output};
        }

        static torch::autograd::variable_list backward(torch::autograd::AutogradContext* ctx,
                                                       const torch::autograd::variable_list& grad_output) {
            auto saved = ctx->get_saved_variables();
            auto input = saved[0];
            auto scale = saved[1];
            auto zero_point = saved[2];
            auto axis = ctx->saved_data["axis"].toInt();
            auto quant_min = ctx->saved_data["quant_min"].toInt();
            auto quant_max = ctx->saved_data["quant_max"].toInt();
            auto use_grad_scaling = ctx->saved_data["use_grad_scaling"].toBool();
            auto grad_scaler = ctx->saved_data["grad_scaler"].toDouble();
            auto sym = ctx->saved_data["sym"].toBool();
            auto init_mode = ctx->saved_data["init_mode"].toBool();
            
            auto result = detail::lsq_backward_per_channel(grad_output[0], input, scale, zero_point,
                                                           axis, quant_min, quant_max, use_grad_scaling, grad_scaler, sym, init_mode);
            auto grad_input =  std::get<0>(result);
            auto grad_scale =  std::get<1>(result);
            auto grad_zero_point = std::get<2>(result);
            return {grad_input, grad_scale, grad_zero_point, torch::Tensor(),
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
                                                      const torch::Tensor& zero_point,
                                                      const int64_t axis,
                                                      const int64_t quant_min,
                                                      const int64_t quant_max,
                                                      const bool use_grad_scaling,
                                                      const double grad_scaler,
                                                      const bool sym,
                                                      const bool init_mode) {
            at::AutoNonVariableTypeMode g;
            auto result = detail::lsq_backward_per_channel(grad, input, scale, zero_point,
                                                           axis, quant_min, quant_max, use_grad_scaling, grad_scaler, sym, init_mode);
            auto grad_input =  std::get<0>(result);
            auto grad_scale =  std::get<1>(result);
            auto grad_zero_point = std::get<2>(result);

            return {grad_input, grad_scale, grad_zero_point};
        }

        static torch::autograd::variable_list backward(torch::autograd::AutogradContext* ctx,
                                                      const torch::autograd::variable_list& grad_output) {
            TORCH_CHECK(0, "double backwards on lsq_per_channel not supported");
        }
};


torch::Tensor lsq_per_tensor_autograd(const torch::Tensor& x,
                                      const torch::Tensor& scale,
                                      const torch::Tensor& zero_point,
                                      const int64_t quant_min,
                                      const int64_t quant_max,
                                      const bool use_grad_scaling,
                                      const double grad_scaler,
                                      const bool sym,
                                      const bool init_mode) {
    return LSQPerTensorFunction::apply(x, scale, zero_point, quant_min, quant_max, use_grad_scaling, grad_scaler, sym, init_mode)[0];
}    
                                                                   
std::tuple<torch::Tensor, 
           torch::Tensor, 
           torch::Tensor> lsq_per_tensor_autograd_backward(const torch::Tensor& grad,
                                                           const torch::Tensor& x,
                                                           const torch::Tensor& scale,
                                                           const torch::Tensor& zero_point,
                                                           const int64_t quant_min,
                                                           const int64_t quant_max,
                                                           const bool use_grad_scaling,
                                                           const double grad_scaler,
                                                           const bool sym,
                                                           const bool init_mode){
    auto result = LSQPerTensorBackwardFunction::apply(grad, x, scale, zero_point, quant_min, quant_max, use_grad_scaling, grad_scaler, sym, init_mode);
    return std::make_tuple(result[0], result[1], result[2]);
}  
    
torch::Tensor lsq_per_channel_autograd(const torch::Tensor& x,
                                       const torch::Tensor& scale,
                                       const torch::Tensor& zero_point,
                                       const int64_t axis,
                                       const int64_t quant_min,
                                       const int64_t quant_max,
                                       const bool use_grad_scaling,
                                       const double grad_scaler,
                                       const bool sym,
                                       const bool init_mode) {
    return LSQPerChannelFunction::apply(x, scale, zero_point, axis, quant_min, quant_max, use_grad_scaling, grad_scaler, sym, init_mode)[0];
}    
                                                                   
std::tuple<torch::Tensor, 
           torch::Tensor, 
           torch::Tensor> lsq_per_channel_autograd_backward(const torch::Tensor& grad,
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
    auto result = LSQPerChannelBackwardFunction::apply(grad, x, scale, zero_point, axis, quant_min, quant_max, use_grad_scaling, grad_scaler, sym, init_mode);
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