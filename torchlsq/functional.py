import torch
from .extension import _assert_has_ops


Tensor = torch.Tensor


def lsq(x: Tensor, scale: Tensor, shift: Tensor,
        quant_min: int = 0, 
        quant_max: int = 255,
        type_min: int = None,
        type_max: int = None,
        axis: int = 1,
        use_grad_scaling: bool = True, 
        grad_scaler: float = 1.,
        is_affine: bool = True,
        is_perchannel: bool = False,
        eval_mode: bool = False,
        init_mode: bool = False) -> Tensor:
    '''
        Pytorch FakeQuantizer Observer with Learned Step Size Quantization(LSQ+) (https://arxiv.org/abs/2004.09576)
        
        Theory:
            The Pytorch quantization scheme is x_q = round(clamp(x/scale + zero_point,quant_min,quant_max); 
                where zero_point is integer and have same type as x_q.
            
            In the article, however, used another approach: x_q = round(clamp((x - shift)/scale,quant_min,quant_max); 
                shift - have float type here. We adapt such scheme via implicit conversion of shift to zero_point
            
            Following article, LSQ+ emulate the quantization and dequantization of input treating `scale` and `shift` as learnable parameters.

            Forward pass(quantize->dequantize):
               1) Emualate zero_point behaviour zero_point = clamp(-shift/scale, type_min, type_max); where type_min(type_max) is min(max) of quantized type (i.e int8/uint8)
               2) Quantized: x_q = round(clamp(x/scale + zero_point, quant_min, quant_max)
               3) Dequantized: x_r = x_q * scale + shift
        
            Backward pass:
                Consider, 1) zero_point = clamp(-shift/scale, type_min, type_max);
                          2) x_q = round(clamp(x/scale + zero_point, quant_min, quant_max));
                          3) x̂ = (x - shift) / scale

                w.r.t input: 1,  if ((quant_min < x_q) && (x_q < quant_max))
                             0,  otherwise

                w.r.t scale: round(x̂) - x̂, if ((quant_min < x̂) && (x̂ < quant_max)) 
                             quant_min,    if x̂ <= quant_min
                             quant_max,    if x̂ >= quant_max

                w.r.t shfit:  0,  if ((quant_min < x̂) && x̂ < quant_max))
                              1,  otherwise
                                   

        Note! By default in 8-bit assymetric quantization will used: `quant_min` = 0, `quant_max` = 255

        By default the affine quantization is used (defined by `is_affine` flag)
        In case of symmetric(`is_affine`=False), the shift forcible treat as 0 both during backward pass

        Implementation natively support per_tensor(default) and per_channel schemes via `is_perchannel` flag.

  
        We support parameters initialization mode via `init_mode` flag.
            In original article the authors propose to init `scale` and `shift` for affine case by minimizing: ||x_r-x||F^2
            Instead of using it as additional loss, we incorporate this natively inside code,
            due to this function supposed to be work inside Pytorch Observer class.
        When `init_mode`=True only 'scale' and 'shift' will be updated treating ||x_r-x||F^2 as main loss function.

        Based on article we support scaling of derivatives on `scale` and `shift` on 1/sqrt(n_elements(x)*quant_max)
        (See https://arxiv.org/abs/1902.08153 for details)

        Note! type_min/type_max is stands for numerical limits of used quantized type.
              For Pytorch e.g. quint8 and qint8. They may corresponds to quant_min/quant_max,
              but is not always true, for example for unsigned 7-bit the quant_min == type_min, but quant_min < type_max.

        Arguments:
            x(torch.tensor) - input tensor
            scale(torch.tensor) - what the name said, 1D tensor(dim==1), even in per_tensor case.
            shift(torch.tensor) - what the name said, 1D tensor(dim==1), even in per_tensor case.
            quant_min(int) - lower bound of quantized range. Default: 0 (for assymetric quantization)
            quant_max(int) - upper bound of quantized range. Default: 255.
            type_min(int) - lower bound of quantized type. Default: None (= quant_min)
            type_max(int) - upper bound of quantized type. Default: None (= quant_max)
            axis(int) - dimension for per channel quantization scheme. Default: 1.
            use_grad_scaling(bool) - Apply scale to scale and zero_point gradient. Default: True.
            grad_scaler(float) - additional grad scaler, helps to compensate the learning rate. Default: 1.
            is_affine(bool) - Control assymmetric(true) or symmetric(false) quantization type. Default: true(assymmetric).
            is_perchannel(bool) - Control quantization scheme: per_channel(true) and per_tensor(false). Default: false(per_tensor)
            eval_mode(bool) - Force LSQ work like PyTorch FakeQuantizer(only fake quant) if enabled. Default: false.
            init_mode(bool) - Control parameters initialization(see description above). Default: false(not used)
    '''
    _assert_has_ops()
    if not is_affine:
        assert quant_min <= 0 <= quant_max, 'quantization range must be covered 0 in symmetric quantization' 
    type_min = quant_min if type_min is None else type_min
    type_max = quant_max if type_max is None else type_max
        
    return torch.ops.torchlsq.lsq(x, scale, shift, quant_min, quant_max, type_min, type_max,
                                     axis, use_grad_scaling, grad_scaler, is_affine, is_perchannel,
                                     eval_mode, init_mode)
    