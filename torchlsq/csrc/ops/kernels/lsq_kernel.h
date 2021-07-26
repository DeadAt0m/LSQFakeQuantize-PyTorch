#include "../global_scope.h"

// STDLIB is pre-compiler alias for std/thrust where it is required

template <typename scalar_t>
API_HOST API_DEVICE API_INLINE scalar_t lsq_forward_kernel_per_tensor(scalar_t x, scalar_t s, scalar_t inv_s, 
                                                                      scalar_t b, 
                                                                      scalar_t qmin, scalar_t qmax, 
                                                                      scalar_t tmin, scalar_t tmax,
                                                                      bool init_mode)
{
    const scalar_t zp = static_cast<scalar_t>(FASTROUND(FMIN(tmax, FMAX(tmin, -b*inv_s))));
    return init_mode?x:(static_cast<scalar_t>((FASTROUND(FMIN(qmax, FMAX(qmin, x * inv_s + zp))) - zp) * s));
}



//Temporary set of separate backward functions for GPU

template <typename scalar_t>
API_HOST API_DEVICE API_INLINE scalar_t lsq_backward_kernel_per_tensor_dX(scalar_t grad, 
                                                                          scalar_t x,
                                                                          scalar_t s,
                                                                          scalar_t inv_s,
                                                                          scalar_t b, 
                                                                          scalar_t qmin, 
                                                                          scalar_t qmax,
                                                                          scalar_t tmin,
                                                                          scalar_t tmax,
                                                                          bool init_mode){
    
    const scalar_t zp = static_cast<scalar_t>(FASTROUND(FMIN(tmax, FMAX(tmin, -b*inv_s))));
    const scalar_t xq = FMAX(FMIN(x * inv_s + zp, qmax), qmin);
    // during initialization, just pass grad without changes
    const scalar_t mask = init_mode?static_cast<scalar_t>(1):static_cast<scalar_t>((qmin < xq) && (xq < qmax));
    return grad*mask;
}

template <typename scalar_t>
API_HOST API_DEVICE API_INLINE scalar_t lsq_backward_kernel_per_tensor_dS(scalar_t grad, 
                                                                          scalar_t x,
                                                                          scalar_t s,
                                                                          scalar_t inv_s,
                                                                          scalar_t b, 
                                                                          scalar_t qmin, 
                                                                          scalar_t qmax,
                                                                          scalar_t tmin,
                                                                          scalar_t tmax,
                                                                          scalar_t grad_scaler,
                                                                          bool init_mode){
    const scalar_t zp = static_cast<scalar_t>(FASTROUND(FMIN(tmax, FMAX(tmin, -b*inv_s))));
    const scalar_t xq = FMAX(FMIN(x * inv_s + zp, qmax), qmin);
    const bool mask = ((qmin < xq) && (xq < qmax));
    // directly minimize ||x_r - x||F^2 for initialization s and zp via optimizer
    // replace grad on  d(||x_r - x||F^2)/dx_r = 2*(x_r - x)
    const scalar_t xfq = (FASTROUND(xq) - zp) * s;
    const scalar_t _grad = init_mode?static_cast<scalar_t>(2*(xfq-x)):grad;  
    // scale grad
    const scalar_t border_scale = (xq <= qmin)?(_grad*(qmin - zp)):(_grad*(qmax - zp));
    const scalar_t dS = mask?static_cast<scalar_t>(_grad*(xfq-x)*inv_s):border_scale;  
    return dS*grad_scaler;
}

template <typename scalar_t>
API_HOST API_DEVICE API_INLINE scalar_t lsq_backward_kernel_per_tensor_dB(scalar_t grad, 
                                                                          scalar_t x,
                                                                          scalar_t s,
                                                                          scalar_t inv_s,
                                                                          scalar_t b, 
                                                                          scalar_t qmin, 
                                                                          scalar_t qmax,
                                                                          scalar_t tmin,
                                                                          scalar_t tmax,
                                                                          scalar_t grad_scaler,
                                                                          bool sym,
                                                                          bool init_mode){
    const scalar_t zp = static_cast<scalar_t>(FASTROUND(FMIN(tmax, FMAX(tmin, -b*inv_s))));
    const scalar_t xq = FMAX(FMIN(x * inv_s + zp, qmax), qmin);
    const bool mask = ((qmin < xq) && (xq < qmax));
    // directly minimize ||x_r - x||F^2 for initialization s and zp via optimizer
    // replace grad on  d(||x_r - x||F^2)/dx_r = 2*(x_r - x)
    const scalar_t xfq = (FASTROUND(xq) - zp) * s;
    const scalar_t _grad = init_mode?static_cast<scalar_t>(2*(xfq-x)):grad;   
    // shift grad (during symmetric quantization, do not pass grad for shift)
    const scalar_t dB = sym?static_cast<scalar_t>(0):(static_cast<scalar_t>(!mask)*_grad);      
    return dB*grad_scaler;
}



// combined backward for future

template <typename scalar_t>
API_HOST API_DEVICE API_INLINE STDLIB::tuple<scalar_t, scalar_t, scalar_t> lsq_backward_kernel_per_tensor(scalar_t grad, 
                                                                                                          scalar_t x,
                                                                                                          scalar_t s,
                                                                                                          scalar_t inv_s,
                                                                                                          scalar_t b, 
                                                                                                          scalar_t qmin, 
                                                                                                          scalar_t qmax,
                                                                                                          scalar_t tmin,
                                                                                                          scalar_t tmax,
                                                                                                          scalar_t grad_scaler,
                                                                                                          bool sym,
                                                                                                          bool init_mode)
{
    const scalar_t zp = static_cast<scalar_t>(FASTROUND(FMIN(tmax, FMAX(tmin, -b*inv_s))));
    const scalar_t xq = FMAX(FMIN(x * inv_s + zp, qmax), qmin);
    const bool mask = ((qmin < xq) && (xq < qmax));
    
    // during initialization, just pass grad without changes
    const scalar_t dX = init_mode?grad:(grad * static_cast<scalar_t>(mask));
    // directly minimize ||x_r - x||F^2 for initialization s and zp via optimizer
    // replace grad on  d(||x_r - x||F^2)/dx_r = 2*(x_r - x)
    const scalar_t xfq = (FASTROUND(xq) - zp) * s;
    const scalar_t _grad = init_mode?static_cast<scalar_t>(2*(xfq-x)):grad;  
    // shift grad (during symmetric quantization, do not pass grad for shift)
    const scalar_t dB= sym?static_cast<scalar_t>(0):(static_cast<scalar_t>(!mask)*_grad);   
    // scale grad
    const scalar_t border_scale = (xq <= qmin)?(_grad*(qmin - zp)):(_grad*(qmax - zp));
    const scalar_t dS = mask?static_cast<scalar_t>(_grad*(xfq-x)*inv_s):border_scale;  
    return {dX, dS*grad_scaler, dB*grad_scaler};
}

template <typename scalar_t>
API_HOST API_DEVICE API_INLINE STDLIB::tuple<scalar_t, scalar_t, scalar_t> lsq_backward_kernel_per_tensor_eval(scalar_t grad, 
                                                                                                               scalar_t x,
                                                                                                               scalar_t s,
                                                                                                               scalar_t inv_s,
                                                                                                               scalar_t b, 
                                                                                                               scalar_t qmin, 
                                                                                                               scalar_t qmax,
                                                                                                               scalar_t tmin,
                                                                                                               scalar_t tmax,
                                                                                                               bool init_mode)
{
    const scalar_t zp = static_cast<scalar_t>(FASTROUND(FMIN(tmax, FMAX(tmin, -b*inv_s))));
    const scalar_t xq = FMAX(FMIN(x * inv_s + zp, qmax), qmin);
    const bool mask = ((qmin < xq) && (xq < qmax));
    // during initialization, just pass grad without changes
    const scalar_t dX = init_mode?grad:(grad * static_cast<scalar_t>(mask));
    // because we in eval mode, we do not need to compute params' gradients
    const scalar_t zero = static_cast<scalar_t>(0);
    return {dX, zero, zero};
}




template <typename scalar_t>
API_HOST API_DEVICE API_INLINE scalar_t lsq_forward_kernel_per_channel(scalar_t x, scalar_t s, scalar_t b,
                                                                       scalar_t qmin, scalar_t qmax, 
                                                                       scalar_t tmin, scalar_t tmax,
                                                                       bool init_mode,
                                                                       scalar_t eps)
{
    const scalar_t _s = FMAX(eps, ABS(s));
    const scalar_t inv_s = static_cast<scalar_t>(1)/_s;
    return lsq_forward_kernel_per_tensor(x, _s, inv_s, b, qmin, qmax, tmin, tmax, init_mode);
}


//Temporary set of separate backward functions for GPU
template <typename scalar_t>
API_HOST API_DEVICE API_INLINE scalar_t lsq_backward_kernel_per_channel_dX(scalar_t grad, 
                                                                           scalar_t x,
                                                                           scalar_t s,
                                                                           scalar_t b, 
                                                                           scalar_t qmin, 
                                                                           scalar_t qmax,
                                                                           scalar_t tmin, 
                                                                           scalar_t tmax,
                                                                           bool init_mode,
                                                                           scalar_t eps)
{
    const scalar_t _s = FMAX(eps, ABS(s));
    const scalar_t inv_s = static_cast<scalar_t>(1)/_s;
    return lsq_backward_kernel_per_tensor_dX(grad, x, _s, inv_s, b, qmin, qmax, tmin, tmax, init_mode);
}

template <typename scalar_t>
API_HOST API_DEVICE API_INLINE scalar_t lsq_backward_kernel_per_channel_dS(scalar_t grad, 
                                                                           scalar_t x,
                                                                           scalar_t s,
                                                                           scalar_t b,
                                                                           scalar_t qmin, 
                                                                           scalar_t qmax,
                                                                           scalar_t tmin, 
                                                                           scalar_t tmax,
                                                                           scalar_t grad_scaler,
                                                                           bool init_mode,
                                                                           scalar_t eps)
{
    const scalar_t _s = FMAX(eps, ABS(s));
    const scalar_t inv_s = static_cast<scalar_t>(1)/_s;
    return lsq_backward_kernel_per_tensor_dS(grad, x, _s, inv_s, b, qmin, qmax, tmin, tmax, grad_scaler, init_mode);
}

template <typename scalar_t>
API_HOST API_DEVICE API_INLINE scalar_t lsq_backward_kernel_per_channel_dB(scalar_t grad, 
                                                                            scalar_t x,
                                                                            scalar_t s,
                                                                            scalar_t b,
                                                                            scalar_t qmin, 
                                                                            scalar_t qmax,
                                                                            scalar_t tmin, 
                                                                            scalar_t tmax,
                                                                            scalar_t grad_scaler,
                                                                            bool sym,
                                                                            bool init_mode,
                                                                            scalar_t eps)
{
    const scalar_t _s = FMAX(eps, ABS(s));
    const scalar_t inv_s = static_cast<scalar_t>(1)/_s;
    return lsq_backward_kernel_per_tensor_dB(grad, x, _s, inv_s, b, qmin, qmax, tmin, tmax, grad_scaler, sym, init_mode);
}




// combined backward for future
template <typename scalar_t>
API_HOST API_DEVICE API_INLINE STDLIB::tuple<scalar_t, scalar_t, scalar_t> lsq_backward_kernel_per_channel(scalar_t grad, 
                                                                                                           scalar_t x,
                                                                                                           scalar_t s,
                                                                                                           scalar_t b, 
                                                                                                           scalar_t qmin, 
                                                                                                           scalar_t qmax,
                                                                                                           scalar_t tmin, 
                                                                                                           scalar_t tmax,
                                                                                                           scalar_t grad_scaler,
                                                                                                           bool sym,
                                                                                                           bool init_mode,
                                                                                                           scalar_t eps)
{
    const scalar_t _s = FMAX(eps, ABS(s));
    const scalar_t inv_s = static_cast<scalar_t>(1)/_s;  
    return lsq_backward_kernel_per_tensor(grad, x, _s, inv_s, b, qmin, qmax, tmin, tmax, grad_scaler, sym, init_mode);
}

template <typename scalar_t>
API_HOST API_DEVICE API_INLINE STDLIB::tuple<scalar_t, scalar_t, scalar_t> lsq_backward_kernel_per_channel_eval(scalar_t grad, 
                                                                                                                scalar_t x,
                                                                                                                scalar_t s,
                                                                                                                scalar_t b, 
                                                                                                                scalar_t qmin, 
                                                                                                                scalar_t qmax,
                                                                                                                scalar_t tmin, 
                                                                                                                scalar_t tmax,
                                                                                                                bool init_mode,
                                                                                                                scalar_t eps)
{
    const scalar_t _s = FMAX(eps, ABS(s));
    const scalar_t inv_s = static_cast<scalar_t>(1)/_s;  
    return lsq_backward_kernel_per_tensor_eval(grad, x, _s, inv_s, b, qmin, qmax, tmin, tmax, init_mode);
}
