#include "../global_scope.h"

// STDLIB is pre-compiler alias for std/thrust where it is required


template <typename scalar_t>
API_HOST API_DEVICE API_INLINE scalar_t lsq_forward_kernel_per_tensor(scalar_t x, scalar_t s, scalar_t inv_s, 
                                                                      scalar_t zp, scalar_t qmin, 
                                                                      scalar_t qmax, bool sym, bool init_mode)
{
    scalar_t _zp = sym?static_cast<scalar_t>(0):zp;
    return init_mode?x:static_cast<scalar_t>((std::nearbyint(FMIN(qmax, FMAX(qmin, x * inv_s + _zp))) - _zp) * s);
}



//Temporary set of separate backward functions for GPU

template <typename scalar_t>
API_HOST API_DEVICE API_INLINE scalar_t lsq_backward_kernel_per_tensor_dX(scalar_t grad, 
                                                                          scalar_t x,
                                                                          scalar_t inv_s,
                                                                          scalar_t zp, 
                                                                          scalar_t qmin, 
                                                                          scalar_t qmax,
                                                                          bool sym,
                                                                          bool init_mode){
    const scalar_t _zp = sym?static_cast<scalar_t>(0):zp;
    const scalar_t xq = FMAX(FMIN(x * inv_s + _zp, qmax), qmin);
    // during initialization, just pass grad without changes
    const scalar_t mask = init_mode?static_cast<scalar_t>(1):static_cast<scalar_t>((qmin < xq) && (xq < qmax));
    return grad*mask;
}

template <typename scalar_t>
API_HOST API_DEVICE API_INLINE scalar_t lsq_backward_kernel_per_tensor_dS(scalar_t grad, 
                                                                          scalar_t x,
                                                                          scalar_t s,
                                                                          scalar_t inv_s,
                                                                          scalar_t zp, 
                                                                          scalar_t qmin, 
                                                                          scalar_t qmax,
                                                                          scalar_t grad_scaler,
                                                                          bool sym,
                                                                          bool init_mode){
    const scalar_t _zp = sym?static_cast<scalar_t>(0):zp;
    const scalar_t xq = FMAX(FMIN(x * inv_s + _zp, qmax), qmin);
    const bool mask = ((qmin < xq) && (xq < qmax));
    // directly minimize ||x_r - x||F^2 for initialization s and zp via optimizer
    // replace grad on  d(||x_r - x||F^2)/dx_r = 2*(x_r - x)
    const scalar_t xfq = (std::nearbyint(xq) - _zp) * s;
    const scalar_t _grad = init_mode?static_cast<scalar_t>(2*(xfq-x)):grad;     
    // scale grad
    const scalar_t border_scale = (xq <= qmin)?(_grad*(qmin - _zp)):(_grad*(qmax - _zp));
    const scalar_t dS = mask?static_cast<scalar_t>(_grad*(xfq-x)*inv_s):border_scale;
//     const scalar_t dS = mask?static_cast<scalar_t>(_grad*(std::nearbyint(xq)-xq)):border_scale;    
    return dS*grad_scaler;
}

template <typename scalar_t>
API_HOST API_DEVICE API_INLINE scalar_t lsq_backward_kernel_per_tensor_dZP(scalar_t grad, 
                                                                           scalar_t x,
                                                                           scalar_t s,
                                                                           scalar_t inv_s,
                                                                           scalar_t zp, 
                                                                           scalar_t qmin, 
                                                                           scalar_t qmax,
                                                                           scalar_t grad_scaler,
                                                                           bool sym,
                                                                           bool init_mode){
    const scalar_t _zp = sym?static_cast<scalar_t>(0):zp;
    const scalar_t xq = FMAX(FMIN(x * inv_s + _zp, qmax), qmin);
    const bool mask = ((qmin < xq) && (xq < qmax));
    // directly minimize ||x_r - x||F^2 for initialization s and zp via optimizer
    // replace grad on  d(||x_r - x||F^2)/dx_r = 2*(x_r - x)
    const scalar_t xfq = (std::nearbyint(xq) - _zp) * s;
    const scalar_t _grad = init_mode?static_cast<scalar_t>(2*(xfq-x)):grad;  
    // zero_point grad (during symmetric quantization, do not pass grad for zp)
    const scalar_t dZP = sym?static_cast<scalar_t>(0):((-1)*s*static_cast<scalar_t>(!mask)*_grad);      
    return dZP*grad_scaler;
}



// combined backward for future

template <typename scalar_t>
API_HOST API_DEVICE API_INLINE STDLIB::tuple<scalar_t, scalar_t, scalar_t> lsq_backward_kernel_per_tensor(scalar_t grad, 
                                                                                                          scalar_t x,
                                                                                                          scalar_t s,
                                                                                                          scalar_t inv_s,
                                                                                                          scalar_t zp, 
                                                                                                          scalar_t qmin, 
                                                                                                          scalar_t qmax,
                                                                                                          scalar_t grad_scaler,
                                                                                                          bool sym,
                                                                                                          bool init_mode)
{
    const scalar_t _zp = sym?static_cast<scalar_t>(0):zp;
    const scalar_t xq = FMAX(FMIN(x * inv_s + _zp, qmax), qmin);
    const bool mask = ((qmin < xq) && (xq < qmax));
    // during initialization, just pass grad without changes
    const scalar_t dX = init_mode?grad:(grad * static_cast<scalar_t>(mask));
    // directly minimize ||x_r - x||F^2 for initialization s and zp via optimizer
    // replace grad on  d(||x_r - x||F^2)/dx_r = 2*(x_r - x)
    const scalar_t xfq = (std::nearbyint(xq) - _zp) * s;
    const scalar_t _grad = init_mode?static_cast<scalar_t>(2*(xfq-x)):grad;  
    // zero_point grad (during symmetric quantization, do not pass grad for zp)
    const scalar_t dZP = sym?static_cast<scalar_t>(0):((-1)*s*static_cast<scalar_t>(!mask)*_grad);    
    // scale grad
    const scalar_t border_scale = (xq <= qmin)?(_grad*(qmin - _zp)):(_grad*(qmax - _zp));
    const scalar_t dS = mask?static_cast<scalar_t>(_grad*(xfq-x)*inv_s):border_scale;
    return {dX, dS*grad_scaler, dZP*grad_scaler};
}



template <typename scalar_t>
API_HOST API_DEVICE API_INLINE scalar_t lsq_forward_kernel_per_channel(scalar_t x, scalar_t s, 
                                                                       scalar_t zp, scalar_t qmin, 
                                                                       scalar_t qmax, bool sym, bool init_mode,
                                                                       scalar_t eps)
{
    const scalar_t _s = FMAX(eps, ABS(s));
    const scalar_t inv_s = static_cast<scalar_t>(1)/_s;
    const scalar_t _zp = sym?static_cast<scalar_t>(0):zp;
    return init_mode?x:static_cast<scalar_t>((std::nearbyint(FMIN(qmax, FMAX(qmin, x * inv_s + _zp))) - _zp) * _s);
}


//Temporary set of separate backward functions for GPU
template <typename scalar_t>
API_HOST API_DEVICE API_INLINE scalar_t lsq_backward_kernel_per_channel_dX(scalar_t grad, 
                                                                           scalar_t x,
                                                                           scalar_t s,
                                                                           scalar_t zp, 
                                                                           scalar_t qmin, 
                                                                           scalar_t qmax,
                                                                           bool sym,
                                                                           bool init_mode)
{
    const scalar_t inv_s = static_cast<scalar_t>(1)/s;
    const scalar_t _zp = sym?static_cast<scalar_t>(0):zp;
    const scalar_t xq = FMAX(FMIN(x * inv_s + _zp, qmax), qmin);
    // during initialization, just pass grad without changes
    const scalar_t mask = init_mode?static_cast<scalar_t>(1):static_cast<scalar_t>((qmin < xq) && (xq < qmax));
    return grad*mask;
}

template <typename scalar_t>
API_HOST API_DEVICE API_INLINE scalar_t lsq_backward_kernel_per_channel_dS(scalar_t grad, 
                                                                           scalar_t x,
                                                                           scalar_t s,
                                                                           scalar_t zp, 
                                                                           scalar_t qmin, 
                                                                           scalar_t qmax,
                                                                           scalar_t grad_scaler,
                                                                           bool sym,
                                                                           bool init_mode)
{
    const scalar_t inv_s = static_cast<scalar_t>(1)/s;
    const scalar_t _zp = sym?static_cast<scalar_t>(0):zp;
    const scalar_t xq = FMAX(FMIN(x * inv_s + _zp, qmax), qmin);
    const bool mask = ((qmin < xq) && (xq < qmax));
    // directly minimize ||x_r - x||F^2 for initialization s and zp via optimizer
    // replace grad on  d(||x_r - x||F^2)/dx_r = 2*(x_r - x)
    const scalar_t xfq = (std::nearbyint(xq) - _zp) * s;
    const scalar_t _grad = init_mode?static_cast<scalar_t>(2*(xfq-x)):grad;     
    // scale grad
    const scalar_t border_scale = (xq <= qmin)?(_grad*(qmin - _zp)):(_grad*(qmax - _zp));
    const scalar_t dS = mask?static_cast<scalar_t>(_grad*(xfq-x)*inv_s):border_scale;    
    return dS*grad_scaler;
}

template <typename scalar_t>
API_HOST API_DEVICE API_INLINE scalar_t lsq_backward_kernel_per_channel_dZP(scalar_t grad, 
                                                                            scalar_t x,
                                                                            scalar_t s,
                                                                            scalar_t zp, 
                                                                            scalar_t qmin, 
                                                                            scalar_t qmax,
                                                                            scalar_t grad_scaler,
                                                                            bool sym,
                                                                            bool init_mode)
{
    const scalar_t inv_s = static_cast<scalar_t>(1)/s;
    const scalar_t _zp = sym?static_cast<scalar_t>(0):zp;
    const scalar_t xq = FMAX(FMIN(x * inv_s + _zp, qmax), qmin);
    const bool mask = ((qmin < xq) && (xq < qmax));
    // directly minimize ||x_r - x||F^2 for initialization s and zp via optimizer
    // replace grad on  d(||x_r - x||F^2)/dx_r = 2*(x_r - x)
    const scalar_t xfq = (std::nearbyint(xq) - _zp) * s;
    const scalar_t _grad = init_mode?static_cast<scalar_t>(2*(xfq-x)):grad;     
    // zero_point grad (during symmetric quantization, do not pass grad for zp)
    const scalar_t dZP = sym?static_cast<scalar_t>(0):((-1)*s*static_cast<scalar_t>(!mask)*_grad);      
    return dZP*grad_scaler;
}




// combined backward for future
template <typename scalar_t>
API_HOST API_DEVICE API_INLINE STDLIB::tuple<scalar_t, scalar_t, scalar_t> lsq_backward_kernel_per_channel(scalar_t grad, 
                                                                                                           scalar_t x,
                                                                                                           scalar_t s,
                                                                                                           scalar_t zp, 
                                                                                                           scalar_t qmin, 
                                                                                                           scalar_t qmax,
                                                                                                           scalar_t grad_scaler,
                                                                                                           bool sym,
                                                                                                           bool init_mode)
{
    const scalar_t inv_s = static_cast<scalar_t>(1)/s;
    const scalar_t _zp = sym?static_cast<scalar_t>(0):zp;
    const scalar_t xq = FMAX(FMIN(x * inv_s + _zp, qmax), qmin);
    const bool mask = ((qmin < xq) && (xq < qmax));
    // during initialization, just pass grad without changes
    const scalar_t dX = init_mode?grad:(grad * static_cast<scalar_t>(mask));
    // directly minimize ||x_r - x||F^2 for initialization s and zp via optimizer
    // replace grad on  d(||x_r - x||F^2)/dx_r = 2*(x_r - x)
    const scalar_t xfq = (std::nearbyint(xq) - _zp) * s;
    const scalar_t _grad = init_mode?static_cast<scalar_t>(2*(xfq-x)):grad;     
    // zero_point grad (during symmetric quantization, do not pass grad for zp)
    const scalar_t dZP = sym?static_cast<scalar_t>(0):((-1)*s*static_cast<scalar_t>(!mask)*_grad);    
    // scale grad
    const scalar_t border_scale = (xq <= qmin)?(_grad*(qmin - _zp)):(_grad*(qmax - _zp));
    const scalar_t dS = mask?static_cast<scalar_t>(_grad*(xfq-x)*inv_s):border_scale;    
    return {dX, dS*grad_scaler, dZP*grad_scaler};
}

