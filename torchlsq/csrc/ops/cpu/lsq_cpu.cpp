#if !defined(QUANTOPS_CPU)
#define QUANTOPS_CPU

#include <torch/extension.h>
#include "../global_scope.h"
#include "../kernels/lsq_kernel.h"
#include <ATen/native/cpu/Loops.h>
#include <torch/library.h>

namespace quantops {
namespace ops { 

namespace {

torch::Tensor lsq_forward_per_tensor_impl(const torch::Tensor& x,
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
                                          const bool init_mode)
{
    TORCH_CHECK(x.scalar_type() == scale.scalar_type(), "`input` and `scale` must have the same floating-point type");
    TORCH_CHECK(x.scalar_type() == shift.scalar_type(), "`input` and `shift` must have the same floating-point type");

    torch::Tensor output = torch::empty_like(x, x.options(), torch::MemoryFormat::Preserve);
    auto iter = torch::TensorIteratorConfig()
                .check_all_same_dtype(false)
                .add_output(output)
                .add_input(x)
                .build();
    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(),
                               "lsq_forward_per_tensor_impl", 
        [&] {
                const scalar_t qmin = static_cast<scalar_t>(quant_min);
                const scalar_t qmax = static_cast<scalar_t>(quant_max);
                const scalar_t tmin = static_cast<scalar_t>(type_min);
                const scalar_t tmax = static_cast<scalar_t>(type_max);
                const scalar_t b = shift[0].item<scalar_t>();
                const scalar_t s = std::max(static_cast<scalar_t>(std::abs(scale[0].item<scalar_t>())),
                                            std::numeric_limits<scalar_t>::epsilon());
                const scalar_t inv_s = static_cast<scalar_t>(1) / s;
                at::native::cpu_kernel(iter, [=](scalar_t x) -> scalar_t {
                    return lsq_forward_kernel_per_tensor<scalar_t>(x, s, inv_s, b, qmin, qmax, tmin, tmax, init_mode);
                });
            });
    return output;
}

    
std::tuple<torch::Tensor, 
           torch::Tensor, 
           torch::Tensor> lsq_backward_per_tensor_impl(const torch::Tensor& grad,
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
                                                       const bool init_mode)
{
    TORCH_CHECK(grad.scalar_type() == x.scalar_type(), "`grad` and `input` must have the same floating-point type");
    TORCH_CHECK(scale.scalar_type() == x.scalar_type(), "`grad` and `scale` must have the same floating-point type");
    TORCH_CHECK(shift.scalar_type() == x.scalar_type(),  "`grad` and `shift` must have the same floating-point type");
    TORCH_CHECK(x.numel() == grad.numel(), "`x` and `grad` are not the same size");
    if (x.numel() <= 0) {
        return std::make_tuple(x, scale, shift);
    }
               
    torch::Tensor dx = torch::empty_like(x, x.options(), torch::MemoryFormat::Preserve);
    torch::Tensor ds_buffer = torch::empty_like(x, x.options(), torch::MemoryFormat::Preserve);
    torch::Tensor db_buffer = torch::empty_like(x, x.options(), torch::MemoryFormat::Preserve);

    auto iter = torch::TensorIteratorConfig()
                .add_output(dx)
                .add_output(ds_buffer)
                .add_output(db_buffer)
                .add_input(grad)
                .add_input(x)
                .build();

    AT_DISPATCH_FLOATING_TYPES(grad.scalar_type(),
                               "lsq_backward_per_tensor_impl", 
        [&] {
                const scalar_t qmin = static_cast<scalar_t>(quant_min);
                const scalar_t qmax = static_cast<scalar_t>(quant_max);
                const scalar_t tmin = static_cast<scalar_t>(type_min);
                const scalar_t tmax = static_cast<scalar_t>(type_max);
                const scalar_t b = shift[0].item<scalar_t>();
                const scalar_t s = std::max(static_cast<scalar_t>(std::abs(scale[0].item<scalar_t>())),
                                            std::numeric_limits<scalar_t>::epsilon());
                const scalar_t inv_s = static_cast<scalar_t>(1) / s;
                const scalar_t _grad_scaler = use_grad_scaling?static_cast<scalar_t>(grad_scaler / sqrt(x.numel()*qmax)):
                                                              static_cast<scalar_t>(grad_scaler);
                iter.for_each([&](char** data, const int64_t* strides, int64_t n) {
                    for (int64_t i = 0; i < n; i++) {
                        scalar_t* grad_ref = (scalar_t*)(data[3] + i * strides[3]);
                        scalar_t* x_ref = (scalar_t*)(data[4] + i * strides[4]);
                        std::tuple<scalar_t, scalar_t, scalar_t> res = eval_mode?lsq_backward_kernel_per_tensor_eval<scalar_t>(*grad_ref, 
                                                                                                                               *x_ref, 
                                                                                                                                s,
                                                                                                                                inv_s,
                                                                                                                                b,
                                                                                                                                qmin,
                                                                                                                                qmax,
                                                                                                                                tmin,
                                                                                                                                tmax,
                                                                                                                                init_mode):
                                                                                 lsq_backward_kernel_per_tensor<scalar_t>(*grad_ref, 
                                                                                                                          *x_ref, 
                                                                                                                           s,
                                                                                                                           inv_s,
                                                                                                                           b,
                                                                                                                           qmin,
                                                                                                                           qmax,
                                                                                                                           tmin,
                                                                                                                           tmax,
                                                                                                                           _grad_scaler,
                                                                                                                           sym,
                                                                                                                           init_mode);
    
                        *(scalar_t*)(data[0] + i * strides[0]) = std::get<0>(res);
                        *(scalar_t*)(data[1] + i * strides[1]) = std::get<1>(res);
                        *(scalar_t*)(data[2] + i * strides[2]) = std::get<2>(res);
                    }
                });
            });
    torch::Tensor ds = ds_buffer.sum().unsqueeze(0).to(scale.device());
    torch::Tensor db = db_buffer.sum().unsqueeze(0).to(shift.device());
    return std::make_tuple(dx, ds, db);
}



torch::Tensor lsq_forward_per_channel_impl(const torch::Tensor& x,
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
                                           const bool init_mode)
{
    TORCH_CHECK(x.scalar_type() == scale.scalar_type(), "`input` and `scale` must have the same floating-point type");
    TORCH_CHECK(x.scalar_type() == shift.scalar_type(), "`input` and `shift` must have the same floating-point type");
    TORCH_CHECK(scale.numel() == shift.numel(), "scale and shift need to have the same dimensions");
    TORCH_CHECK(scale.numel() == x.size(axis), "dimensions of scale and shift are not consistent with input tensor")
    TORCH_CHECK(axis >= 0 && axis <= x.dim(), "`axis` must be between 0 and number of dimensions of input");
    
    torch::Tensor output = torch::empty_like(x, x.options(), torch::MemoryFormat::Preserve);


    std::vector<int64_t> expected_shape(x.dim(), 1);
    expected_shape[axis] = x.size(axis);

    auto iter = torch::TensorIteratorConfig()
                .check_all_same_dtype(false)
                .add_output(output)
                .add_input(x)
                .add_input(torch::_unsafe_view(scale, expected_shape))
                .add_input(torch::_unsafe_view(shift, expected_shape))
                .build();


    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(),
                               "lsq_forward_per_channel_impl", 
        [&] {
                const scalar_t qmin = static_cast<scalar_t>(quant_min);
                const scalar_t qmax = static_cast<scalar_t>(quant_max);
                const scalar_t tmin = static_cast<scalar_t>(type_min);
                const scalar_t tmax = static_cast<scalar_t>(type_max);
                const scalar_t eps = std::numeric_limits<scalar_t>::epsilon();
                at::native::cpu_kernel(iter, [=](scalar_t x, scalar_t s, scalar_t b) -> scalar_t {
                    return lsq_forward_kernel_per_channel<scalar_t>(x, s, b, qmin, qmax, tmin, tmax, init_mode, eps);
                });
            });
    return output;
}

    
    
std::tuple<torch::Tensor, 
           torch::Tensor, 
           torch::Tensor> lsq_backward_per_channel_impl(const torch::Tensor& grad,
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
                                                        const bool init_mode)
{
    TORCH_CHECK(grad.scalar_type() == x.scalar_type(), "`grad` and `input` must have the same floating-point type");
    TORCH_CHECK(scale.scalar_type() == x.scalar_type(), "`grad` and `scale` must have the same floating-point type");
    TORCH_CHECK(shift.scalar_type() == x.scalar_type(),  "`grad` and `shift` must have the same floating-point type");
    TORCH_CHECK(x.numel() == grad.numel(), "`x` and `grad` are not the same size");
    TORCH_CHECK(scale.numel() == shift.numel(), "scale and shift need to have the same dimensions");
    TORCH_CHECK(scale.numel() == x.size(axis), "dimensions of scale and shift are not consistent with input tensor")
    TORCH_CHECK(axis >= 0 && axis < x.dim(), "`axis` must be between 0 and number of dimensions of input");
    if (x.numel() <= 0) {
        return std::make_tuple(x, scale, shift);
    }

    torch::Tensor dx = torch::empty_like(x, x.options(), torch::MemoryFormat::Preserve);
    torch::Tensor ds_buffer = torch::empty_like(x, x.options(), torch::MemoryFormat::Preserve);
    torch::Tensor db_buffer = torch::empty_like(x, x.options(), torch::MemoryFormat::Preserve);

    std::vector<int64_t> expected_shape(x.dim(), 1);
    expected_shape[axis] = x.size(axis);
    
    auto iter = torch::TensorIteratorConfig()
                .add_output(dx)
                .add_output(ds_buffer)
                .add_output(db_buffer)
                .add_input(grad)
                .add_input(x)
                .add_input(torch::_unsafe_view(scale, expected_shape))
                .add_input(torch::_unsafe_view(shift, expected_shape))
                .build();

    AT_DISPATCH_FLOATING_TYPES(grad.scalar_type(),
                               "lsq_backward_per_channel_impl", 
        [&] {
                const scalar_t qmin = static_cast<scalar_t>(quant_min);
                const scalar_t qmax = static_cast<scalar_t>(quant_max);
                const scalar_t tmin = static_cast<scalar_t>(type_min);
                const scalar_t tmax = static_cast<scalar_t>(type_max);
                const scalar_t eps = std::numeric_limits<scalar_t>::epsilon();
                const scalar_t _grad_scaler = use_grad_scaling?static_cast<scalar_t>(grad_scaler / sqrt(x.numel()*qmax/x.size(axis))):
                                                              static_cast<scalar_t>(grad_scaler);
                iter.for_each([&](char** data, const int64_t* strides, int64_t n) {
                    for (int64_t i = 0; i < n; i++) {
                        scalar_t* grad_ref = (scalar_t*)(data[3] + i * strides[3]);
                        scalar_t* x_ref = (scalar_t*)(data[4] + i * strides[4]);
                        scalar_t* s_ref = (scalar_t*)(data[5] + i * strides[5]);
                        scalar_t* b_ref = (scalar_t*)(data[6] + i * strides[6]);
                        std::tuple<scalar_t, scalar_t, scalar_t> res = eval_mode?lsq_backward_kernel_per_channel_eval<scalar_t>(*grad_ref,
                                                                                                                                *x_ref, 
                                                                                                                                *s_ref, 
                                                                                                                                *b_ref,
                                                                                                                                 qmin, 
                                                                                                                                 qmax,
                                                                                                                                 tmin,
                                                                                                                                 tmax,
                                                                                                                                 init_mode,
                                                                                                                                 eps):
                                                                                 lsq_backward_kernel_per_channel<scalar_t>(*grad_ref,
                                                                                                                           *x_ref, 
                                                                                                                           *s_ref, 
                                                                                                                           *b_ref,
                                                                                                                            qmin, 
                                                                                                                            qmax,
                                                                                                                            tmin,
                                                                                                                            tmax,
                                                                                                                            _grad_scaler, 
                                                                                                                            sym,
                                                                                                                            init_mode,
                                                                                                                            eps);
                        *(scalar_t*)(data[0] + i * strides[0]) = std::get<0>(res);
                        *(scalar_t*)(data[1] + i * strides[1]) = std::get<1>(res);
                        *(scalar_t*)(data[2] + i * strides[2]) = std::get<2>(res);
                    }
                });
            });              
               
    std::vector<int64_t> axis_for_reduction(x.ndimension());
    std::iota(std::begin(axis_for_reduction), std::end(axis_for_reduction), 0);
    axis_for_reduction.erase(axis_for_reduction.begin() + axis);
               
    torch::Tensor ds = ds_buffer.sum(axis_for_reduction).to(scale.device());
    torch::Tensor db = db_buffer.sum(axis_for_reduction).to(shift.device());
    return std::make_tuple(dx, ds, db);
}

} // end of anonymous namespace

TORCH_LIBRARY_IMPL(torchlsq, CPU, m) {
    m.impl(
        TORCH_SELECTIVE_NAME("torchlsq::lsq_forward_per_tensor"),
        TORCH_FN(lsq_forward_per_tensor_impl));
    m.impl(
        TORCH_SELECTIVE_NAME("torchlsq::lsq_backward_per_tensor"),
        TORCH_FN(lsq_backward_per_tensor_impl));
    m.impl(
        TORCH_SELECTIVE_NAME("torchlsq::lsq_forward_per_channel"),
        TORCH_FN(lsq_forward_per_channel_impl));
    m.impl(
        TORCH_SELECTIVE_NAME("torchlsq::lsq_backward_per_channel"),
        TORCH_FN(lsq_backward_per_channel_impl));
}

} // namespace ops
} // namespace quantops


#endif