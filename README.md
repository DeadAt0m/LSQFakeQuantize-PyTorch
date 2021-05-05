 Pytorch Observer implementation of Learned Step Size Quantization
 (LSQ  https://arxiv.org/abs/1902.08153 and LSQ+ https://arxiv.org/abs/2004.09576).
 
 Can be considered as replacement of [FakeQuantize Observer in Pytorch](https://pytorch.org/docs/stable/torch.quantization.html#torch.quantization.FakeQuantize) and moreover it fully integrated in [torch.quantization](https://pytorch.org/docs/stable/quantization.html) environment

(**I am not the author** any of mentioned articles, I just implement this for my own purposes)
## !FOR PYTORCH >= 1.7! ##

## Theory

Method emulate the quantization and de-quantization of input and treating `scale` and `zero_point` as learnable parameters.

#### Forward pass:
    
    Recovered input (quantize -> dequantize): 
    
         x_recovered = (round(clamp(x/scale + zero_point,quant_min,quant_max)) - zero_point) * scale
            
#### Backward pass:
                
     Consider x_q = round(clamp(x/scale + zero_point,quant_min,quant_max))

     w.r.t input: 1,  if ((quant_min < x_q) && (x_q < quant_max)),
                  0,  otherwise

     w.r.t scale: round(x/scale + zero_point) - x/scale + zero_point, 
                             if ((quant_min < x_q) && (x_q < quant_max)) 
                  quant_min, if x_q <= quant_min
                  quant_max, if x_q >= quant_max

     w.r.t zero_point:  0,  if ((quant_min < x_q) && (x_q < quant_max))
                        1, otherwise

## Implementation details:

* Based on article, we support scaling of derivatives on `scale` and `zero_point` on `1/sqrt(n_elements(x)*quant_max)`
  (See https://arxiv.org/abs/1902.08153 for details)


* [Pytorch Quantization](https://pytorch.org/docs/stable/quantization.html) assumes using `qint8` for weights quantization and `quint8` for activation.

  Based on that, `qint8` dtype considered as Observer for weights and `quint8` for activation.
  
  *Using the `qint8` type is also restrict quantization schemes to have a symmetric type.*
  
  
* Weight case initialization:
        Is always happens statically following by:
            s_init = max(|µ − 3 ∗ σ|, |µ + 3 ∗ σ|)/2^b−1, 
                     where µ and σ  mean and std of weights tensor correspondingly.
                     
                     
* Activation case initializations:

  **a)** In original article the authors propose to init `scale` and `zero_point` for affine case by minimizing: ||x_r-x||F^2.
     
     Instead of using it as additional loss, we incorporate this natively inside code,
     due to this function supposed to be work inside Pytorch Observer class.
       
  **b)** Alternatively, their is possibility to initialize `scale` and `zero_point` via external observer.
   
     We highly suggest use the [MovingAverage(PerChannel)MinMaxObserver](https://pytorch.org/docs/stable/torch.quantization.html#torch.quantization.MovingAverageMinMaxObserver) to track running min and max values and use them for `scale` and `zero_point` initialization.
     
     *Just pass `AnyObserver.with_args(...)` to `use_exobserver_on_init` argument.*

     *The number of batches during which initialization happens controls by `n_batches`.*
 
 
* LSQ can work with any bitness, however for Pytorch compatibility, we restrict quantization for 7-bit(see below).
  *(If this module intendent to use in custom framework, just overload `_verify_qmin_qmax` method)*
  
  
* By default quant_min and quant_max will be initialized for 7-bit,
   for avoiding overflow during quantized inference in Pytorch.
   
            qint8  - quant_min, quant_max = -64, 63
            quint8 - quant_min, quant_max = 0, 127
  To overcome this, just custom define `quant_min` and `quant_max`  variables! 


## Requirements:
    C++17 must be supported by your compiler! (due to constexpr in code)
    PyTorch >= 1.7.0; 

## Installation:
1. Clone this repo and ```cd LSQ-PyTorch```
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
**We strictly recommend to study the [Pytorch Quantization](https://pytorch.org/docs/stable/quantization.html) manuals (how to prepare the model and how the observers concept work) before using, because this LSQObserver implementation is part of it!**

Code example:
    
    from torch.quantization import QConfig
    from torchlsq import LSQObserver
    lsq_qconfig = QConfig(activation=LSQObserver.with_args(dtype=torch.quint8), weight=LSQObserver.with_args(dtype=torch.qint8))


Additional options for shift layer:

    dtype(torch.dtype) - quint8 or qint8. Default: torch.quint8.
    
    qscheme(torch.qscheme) - per_tensor_affine, per_tensor_symmetric, per_channel_affine or per_channel_symmetric schemes. Default: torch.per_tensor_affine.
    
    quant_min(int) - lower bound of quantized range. Default: None (corresponding 7-bit scheme).
    
    quant_max(int) - upper bound of quantized range. Default: None (corresponding 7-bit scheme).
    
    init_scale(int) - init value for scale, for quint8 dtype(activation only). Default: 1.
    
    init_zero_point(int) - init value for zero point. Default: None(quant_min).
    
    axis(int) - dimension for per channel quantization scheme. Default: None. (1 - for activation, 0 - for weights).
    
    n_batches(int) - Number of batches during which parameter initialization will happen,has effect only when dtype=torch.quint8. Default: 1000.
    
    use_exobserver_on_init(torch.quantization.ObserverBase) - external observer used for `scale` and `zero_point` initialization. Default: None.
    
    use_grad_scaling(bool) - Apply scale to scale and zero_point gradient. Default: True.
    
    grad_scaler(float) - additional grad scaler, helps to compensate the learning rate. Default: 1.
    
    debug_mode(bool) - Turn off the LSQ (Observer will pass the tensor forward without any computations) if True. Default: False.
                                                                                  
                                                                                


## TO DO:
  1. Add unit tests
