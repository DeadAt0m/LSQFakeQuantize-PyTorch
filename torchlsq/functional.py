import torch
from .extension import _assert_has_ops


Tensor = torch.Tensor

def lsq(x: Tensor, scale: Tensor, zero_point: Tensor,
        quant_min: int = 0, 
        quant_max: int = 255,
        axis: int = 1,
        use_grad_scaling: bool = True, 
        grad_scaler: float = 1.,
        is_affine: bool = True,
        is_perchannel: bool = False,
        do_param_init: bool = False) -> Tensor:
    '''
        Implementation of Learned Step Size Quantization(LSQ+) (https://arxiv.org/abs/2004.09576)

        Theory:
            Method emulate the quantization and dequantization of input treating `scale` and `zero_point` as learnable parameters
            Forward pass:
               Recovered input(quantize->dequantize): x_r = (round(clamp(x/scale + zero_point,quant_min,quant_max)) - zero_point) * scale,
            
            Backward pass:
                Consider x_q = round(clamp(x/scale + zero_point,quant_min,quant_max))

                w.r.t input: 1,  if ((quant_min < x_q) && (x_q < quant_max))
                             0,  otherwise

                w.r.t scale: round(x/scale + zero_point) - x/scale + zero_point, if ((quant_min < x_q) && (x_q < quant_max)) 
                             quant_min, if x_q <= quant_min
                             quant_max, if x_q >= quant_max

                w.r.t zero_point:  0,  if ((quant_min < x_q) && (x_q < quant_max))
                                   1, otherwise

        There is several restrictions on `quant_min` and `quant_max` values:
            1) quant_min < quant_max
            2) in case of symmetric it must includes 0 in range: `quant_min` < 0 < `quant_max`
        Note! By default in 8-bit assymetric quantization will used: `quant_min` = 0, `quant_max` = 255

        By default the affine quantization is used (defined by `is_affine` flag)
        In case of symmetric(`is_affine`=False), the zero_point forcible treat as 0 both during forward and backward pass

        Implementation natively support per_tensor(default) and per_channel schemes via `is_perchannel` flag.

  
        We support parameters initialization mode via `do_param_init` flag.
            In original article the authors propose to init `scale` and `zero_point` for affine case by minimizing: ||x_r-x||F^2
            Instead of using it as additional loss, we incorporate this natively inside code,
            due to this function supposed to be work inside Pytorch Observer class.
        When `do_param_init`=True only 'scale' and 'zero_point' will be updated treating ||x_r-x||F^2 as main loss function.

        Based on article we support scaling of derivatives on `scale` and `zero_point` on 1/sqrt(n_elements(x)*quant_max)
        (See https://arxiv.org/abs/1902.08153 for details)


        Arguments:
            x(torch.tensor) - input tensor
            scale(torch.tensor) - what the name said, 1D tensor(dim==1), even in per_tensor case.
            zero_point(torch.tensor) - what the name said, 1D tensor(dim==1), even in per_tensor case.
            quant_min(int) - lower bound of quantized range. Default: 0 (for assymetric quantization)
            quant_max(int) - upper bound of quantized range. Default: 255.
            axis(int) - dimension for per channel quantization scheme. Default: 1.
            use_grad_scaling(bool) - Apply scale to scale and zero_point gradient. Default: True.
            grad_scaler(float) - additional grad scaler, helps to compensate the learning rate. Default: 1.
            is_affine(bool) - Control assymmetric(true) or symmetric(false) quantization type. Default: true(assymmetric).
            is_perchannel(bool) - Control quantization scheme: per_channel(true) and per_tensor(false). Default: false(per_tensor)
            do_param_init(bool) - Control parameters initialization(see description above). Default: false(not used)
    '''
    _assert_has_ops()
    if not is_affine:
        assert quant_min <= 0 < quant_max, 'quantization range must be covered 0 in symmetric quantization' 
    return torch.ops.torchlsq.lsq(x, scale, zero_point, quant_min, quant_max,
                                  axis, use_grad_scaling, grad_scaler, is_affine, is_perchannel, do_param_init)
