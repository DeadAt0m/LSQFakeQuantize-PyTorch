Pytorch FakeQuantizer Observer with Learned Step Size Quantization(LSQ+) (https://arxiv.org/abs/2004.09576)
 (LSQ  https://arxiv.org/abs/1902.08153 and LSQ+ https://arxiv.org/abs/2004.09576).
 
 Can be considered as replacement of [FakeQuantize Observer in Pytorch](https://pytorch.org/docs/stable/torch.quantization.html#torch.quantization.FakeQuantize) and moreover it fully integrated in [torch.quantization](https://pytorch.org/docs/stable/quantization.html) environment

(**I am not the author** any of mentioned articles, I just implement this for my own purposes)
## !FOR PYTORCH >= 1.7! ##

## Theory
 
The Pytorch quantization scheme is `x_q = round(clamp(x/scale + zero_point,quant_min,quant_max)`; 
                                         where `zero_point` is integer and have same type as `x_q`.
            
In the article, however, used another approach: ```x_q = round(clamp((x - shift)/scale,quant_min,quant_max)`; 
`shift` - have float type here. We adapt such scheme via implicit conversion of `shift` to `zero_point`
            
Following article, LSQ+ emulate the quantization and dequantization of input treating `scale` and `shift` as learnable parameters.

### Forward pass

  (quantize->dequantize):
    1) Emualate zero_point behaviour zero_point = clamp(-shift/scale, type_min, type_max); where type_min(type_max) is min(max) of quantized type (i.e int8/uint8)
    2) Quantized: x_q = round(clamp(x/scale + zero_point, quant_min, quant_max)
    3) Dequantized: x_r = x_q * scale + shift

### Backward pass  

    Consider, 1) zero_point = clamp(-shift/scale, type_min, type_max);
              2) x_q = round(clamp(x/scale + zero_point, quant_min, quant_max));

    w.r.t input: 1,  if ((quant_min < x_q) && (x_q < quant_max))
                 0,  otherwise

    w.r.t scale: (x_r - x)/scale, if ((quant_min < x_q) && (x_q < quant_max)) 
                 quant_min - zero_point,    if x_q <= quant_min
                 quant_max - zero_point,    if x_q >= quant_max

    w.r.t shfit:  0,  if ((quant_min < x_q) && x_q < quant_max))
                  1,  otherwise

## Implementation details:

* Based on article, we support scaling of derivatives on `scale` and `shift` on `1/sqrt(n_elements(x)*quant_max)`
  (See https://arxiv.org/abs/1902.08153 for details)


* [Pytorch Quantization](https://pytorch.org/docs/stable/quantization.html) assumes using `qint8` for weights quantization and `quint8` for activation.
   Based on that, `qint8` dtype considered as Observer for weights and `quint8` for activation.

  
  
* Weight case initialization:
        Is always happens statically following by:
            s_init = max(|µ − 3 ∗ σ|, |µ + 3 ∗ σ|)/2^b−1, 
                     where µ and σ  mean and std of weights tensor correspondingly.
                     
                     
* Activation case initializations:

  ***a)*** In original article the authors propose to init `scale` and `shift` for affine case by minimizing: ||x_r-x||F^2
           Instead of using it as additional loss, we incorporate this natively inside code,
           due to this function supposed to be work inside Pytorch Observer class.
                
  ***b)*** Alternatively, their is possibility to initialize `scale` and `shift` via  observer.
           We highly suggest use the [MovingAverage(PerChannel)MinMaxObserver](https://pytorch.org/docs/stable/torch.quantization.html#torch.quantization.MovingAverageMinMaxObserver) to track running min and max values
                
  For control a) and b) use `init_mode` argument during initialization 
  The number of batches during which initialization happens controls by `n_batches`.

* LSQ can work with any bitrate, however for Pytorch compatiblity, we limited bitrate from above to 8-bit.

  
* By default quant_min and quant_max will be initialized for 7-bit,
   for avoiding overflow during quantized inference in Pytorch.

            qint8  - quant_min, quant_max = -64, 63
            quint8 - quant_min, quant_max = 0, 127
  To overcome this, look on `avoid_torch_overflow` argument.


## Requirements:
    C++17 must be supported by your compiler! (due to constexpr in code)
    PyTorch >= 1.7.0; 

## Installation:
1. Clone this repo and ```cd LSQFakeQuantize-PyTorch```
2. (optional) If you compile with CUDA, please pass path to nvcc to CUDA_HOME env variable!
3. **Important!** There is bug in PyTorch which can lead to crash during build under CUDA.
   This bug was fixed in PyTorch 1.8. However it easy to fix it in previous versions.
   Run ```python torch_patch.py```(anyway it will automatically run during step 3) to fix it.
   This script change a few lines of code in single C++ header file, however doing this directly in python dist-package folder.
   Please, be sure that you have rights for changing files inside this folder!
   Anyway, you should do it only once for each python environment(PyTorch package).
   (If something will going wrong, please inspect ```torch_patch.py``` first (it very simple) and try to reproduce patch manually.)
4. Run ```python setup.py install``` or ```python setup.py bdist_wheel``` - to install/build package

    
## Using:
**We strictly recommend to study the [Pytorch Quantization](https://pytorch.org/docs/stable/quantization.html) manuals (how to prepare the model and how the observers concept work) before using, because this LSQFakeQuantize implementation is part of it!**

Code example:
    
    from torch.quantization import QConfig
    from torchlsq import LSQFakeQuantize
    lsq_qconfig = QConfig(activation=LSQObserver.with_args(dtype=torch.quint8), weight=LSQObserver.with_args(dtype=torch.qint8))


Additional options for the layer:

    observer(torch.quantization.ObserverBase) - Module for observing statistics on input tensors and calculating scale and zero-point.

    otype(str) - Type specification for which LSQ will used. Can be one of (`weight`, `activation`)

    dtype(torch.dtype) - quint8 or qint8. Default: torch.quint8.

    qscheme(torch.qscheme) - per_tensor_affine, per_tensor_symmetric, per_channel_affine or 
                             per_channel_symmetric schemes. Default: torch.per_tensor_affine.
                             
    quant_min(int) - lower bound of quantized range. Default: None.

    quant_max(int) - upper bound of quantized range. Default: None.

    init_scale(float) - init value for scale, have effect for "activation" otype only. Default: 1.

    init_shift(float) - init value for shift, have effect for "affine" qschemes. Default: 0

    ch_axis(int) - dimension for per channel quantization scheme. Default: None.(1 - for activation, 0 - for weights)

    learn_params(bool) - enable learning via LSQ, otherwise the module will work like FakeQuantizer. Default: True

    init_batches(int) - Number of batches during which parameter initalization will happen,
                        has effect only for "activation" otype. Default: 1000.

    init_mode(str) - Controls initialization mode for parameters: via backprop('learnable') of via 'observer'. Default: 'observer'.

    use_grad_scaling(bool) - Apply scale to scale and zero_point gradient. Default: True.

    grad_scaler(float) - additional grad scaler, helps to compensate the learning rate. Default: 1.

    avoid_torch_overflow(bool) - Reduce 8-bit to 7-bit during default initialization, to avoid overflows in quantized operations. Default: True.

    debug_mode(bool) - Just pass input during forward if enabled. Default: False.
                                                                                  
                                                                                
## VERSION HISTORY
1.0 - First release, has a lot of bugs with zero_point and poor class design.
2.0 - Full refactor of python part, most of known bugs fixed.
2.1 - Replaced scale gradient; minor enhancements and bugfixes

## TO DO:
  1. Add unit tests

