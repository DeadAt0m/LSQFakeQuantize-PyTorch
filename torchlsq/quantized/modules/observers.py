import torch
from torch._C import dtype
from torchlsq.functional import lsq
from math import log, ceil
from typing import Tuple

Tensor = torch.Tensor

class LSQObserver(torch.quantization.observer.ObserverBase):
    '''
        Pytorch Observer implementation of Learned Step Size Quantization(LSQ+) (https://arxiv.org/abs/2004.09576)
        
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

        Based on article we support scaling of derivatives on `scale` and `zero_point` on 1/sqrt(n_elements(x)*quant_max)
        (See https://arxiv.org/abs/1902.08153 for details)


        Pytorch Quantization assumes using qint8 for weights quantization and quint8 for activation.
            Based on that, qint8 dtype considered as Observer for weights and quint8 for activation.        .
            Using the qint8 type is also restrict quantization schemes to having symmetric type.


        This logic has strict influence on 'scale' initialization.
            - Weight case initialization:
                Is always happens staticly following by:
                s_init = max(|µ − 3 ∗ σ|, |µ + 3 ∗ σ|)/2^b−1, where µ and σ  mean and std of weights tensor correspondingly.

            - Activation case initializations:
                a) In original article the authors propose to init `scale` and `zero_point` for affine case by minimizing: ||x_r-x||F^2
                Instead of using it as additional loss, we incorporate this natively inside code,
                due to this function supposed to be work inside Pytorch Observer class.
                
                b) Alternatively, their is possibility to initialize `scale` and `zero_point` via external observer.
                   We highly suggest use the MovingAverage(PerChannel)MinMaxObserver to track running min and max values
                   and then use them for `scale` and `zero_point` initialization.
    
                   Just pass `AnyObserver.with_args(...)` to `use_exobserver_on_init` argument.

                The number of batches during which initialization happens controls by `n_batches`.
                

        LSQ can work with any bitrate, however for Pytorch compatiblity, we restrict quantization for 8-bit.
        (If this module intendent to use in custom framework, just overload _verify_qmin_qmax method)

        By default quant_min and quant_max will be initialized for 7-bit,
        for avoiding overflow during quantized inference in Pytorch.
            qint8  - quant_min, quant_max = -64, 63
            quint8 - quant_min, quant_max = 0, 127
        To overcome this, just custom define `quant_min` and `quant_max`  variables!    


        Arguments:
            dtype(torch.dtype) - quint8 or qint8. Default: torch.quint8.
            qscheme(torch.qscheme) - per_tensor_affine, per_tensor_symmetric, per_channel_affine or 
                                     per_channel_symmetric schemes. Default: torch.per_tensor_affine.
            quant_min(int) - lower bound of quantized range. Default: None (corresponding 7-bit scheme)
            quant_max(int) - upper bound of quantized range. Default: None (corresponding 7-bit scheme)
            init_scale(int) - init value for scale, for quint8 dtype(activation only). Default: 1
            init_zero_point(int) - init value for zero point. Default: None(quant_min)
            axis(int) - dimension for per channel quantization scheme. Default: None.(1 - for activation, 0 - for weights)
            n_batches(int) - Number of batches during which parameter initalization will happen,
                              has affect only when dtype=torch.quint8. Default: 1000.
            use_exobserver_on_init(torch.quantization.ObserverBase) - external observer used for `scale` and `zero_point` initialization. Default: None                              
            use_grad_scaling(bool) - Apply scale to scale and zero_point gradient. Default: True.
            grad_scaler(float) - additional grad scaler, helps to compensate the learning rate. Default: 1.
            debug_mode(bool) - Turn off the lsq if on. Default: False.
    ''' 
    def __init__(self, dtype=torch.quint8, qscheme=torch.per_tensor_affine,
                 quant_min=None, quant_max=None, init_scale=1., init_zero_point=None,
                 ch_axis=None, n_batches=1000, use_exobserver_on_init=None, use_grad_scaling=True,
                 grad_scaler=1., debug_mode=False):
        super(LSQObserver, self).__init__(dtype)
        self.qscheme = qscheme
        self.ch_axis = ch_axis
        self.n_batches = n_batches 
        self.use_grad_scaling = use_grad_scaling
        self.grad_scaler = grad_scaler
        self.debug_mode = debug_mode
        self.init_ext_observer = use_exobserver_on_init
        if self.init_ext_observer is not None:
            if not isinstance(self.init_ext_observer, torch.quantization.ObserverBase):
                try:
                    self.init_ext_observer = self.init_ext_observer()
                except:
                    pass
            assert isinstance(self.init_ext_observer, torch.quantization.ObserverBase), 'await uninitialized class derived from torch.quantization.ObserverBase or at least its instance'
        assert self.qscheme in (
            torch.per_tensor_affine,
            torch.per_tensor_symmetric,
            torch.per_channel_affine,
            torch.per_channel_symmetric,
        ), "Default Observer only works for per_tensor_affine, \
                per_tensor_symmetric, per_channel_affine, \
                per_channel_symmetric quantization scheme"
        assert self.dtype in (
            torch.qint8,
            torch.quint8,
        ), "Default Observer only works for qint8, quint8 data type"
        if self.ch_axis is None:
            # 0 - for weights(dtype=qint8), 1 - for activation(dtype=quint8)
            self.ch_axis = int(self.dtype == torch.quint8)
        self._is_perchannel = self.qscheme in (torch.per_channel_affine, torch.per_channel_symmetric)
        self._is_affine = (self.qscheme in (torch.per_tensor_affine, torch.per_channel_affine)) and (dtype==torch.quint8)
        self.init_quant_min = quant_min
        self.init_quant_max = quant_max
        self.quant_min, self.quant_max = self._verify_qmin_qmax(quant_min, quant_max)
        self.init_scale = init_scale
        self.init_zero_point = self.quant_min if init_zero_point is None else init_zero_point  
        self.reset()

    def _verify_qmin_qmax(self, quant_min: int, quant_max: int) -> Tuple[int, int]:
        """ Stealed from PyTorch itself.
            Calculates actual qmin and qmax based on the quantization range,
        observer datatype and if range is reduced.
        """
        self.has_customized_qrange = (quant_min is not None) and (quant_max is not None)
        if self.has_customized_qrange:
            assert quant_min <= 0 <= quant_max, "User-specified quantization range must include 0."
            assert quant_min < quant_max, "qmin must be strictly less than qmax for user-specified quantization range."
        if self.has_customized_qrange:
            initial_quant_min, initial_quant_max = 0, 255
            custom_quant_min, custom_quant_max = quant_min, quant_max
            if custom_quant_min is not None and custom_quant_max is not None:
                initial_quant_min, initial_quant_max = custom_quant_min, custom_quant_max
            qrange_len = initial_quant_max - initial_quant_min + 1
            assert 0 < qrange_len <= 256, \
                "quantization range should be positive and not exceed the maximum bit range (=256)."
            if self.dtype == torch.qint8:
                quant_min, quant_max = -qrange_len // 2, qrange_len // 2 - 1
            else:
                quant_min, quant_max = 0, qrange_len - 1
        else:
            if self.dtype == torch.qint8:
                quant_min, quant_max = -64, 63
                # quant_min, quant_max = -128, 127
            else:
                quant_min, quant_max = 0, 127
                # quant_min, quant_max = 0, 255
        return quant_min, quant_max

    def forward(self, x):
        self._init_weights(x)
        if self.debug_mode:
            return x
        # check param init status
        do_param_init = False
        if self._current_batch.item() < self._init_n_batches:
            do_param_init = True
            self._current_batch += 1
            if self.init_ext_observer is not None:
                _ = self.init_ext_observer(x)
        elif self._current_batch.item() == self._init_n_batches and self.init_ext_observer is not None:
            scale, zp = self.init_ext_observer.calculate_qparams()
            self._initialized = False
            self._init_weights(x, scale, zp)
            do_param_init = False
        else:
            do_param_init = False
        return lsq(x, self.scale, self.zero_point, self.quant_min, self.quant_max,
                   self.ch_axis, self.use_grad_scaling, self.grad_scaler,
                   self._is_affine, self._is_perchannel, bool(do_param_init * (not self._disable_fake_quant)))
        
    def extra_repr(self):
        is_initialized = ''
        if not self._initialized:
            is_initialized = '(Uninitialized!) '
            scale, zp = self.init_scale, self.init_zero_point
        else:
            scale, zp = self.calculate_qparams()
        for_weights = 'weights' if self.dtype == torch.qint8 else 'activation' 
        per_channel = 'No' if not self._is_perchannel else f'Yes, channel axis - {self.ch_axis}'
        if self.debug_mode:
            return 'Debug mode: ON, doing nothing'
        return "{}Observer for {}; Qtype:{}, Affine:{}, PerChannel:{}, Qrange:[{},{}], scale={}, zero_point={}".format(is_initialized,
                                                                                                                       for_weights,
                                                                                                                       self.dtype,
                                                                                                                       self._is_affine,
                                                                                                                       per_channel,
                                                                                                                       self.quant_min,
                                                                                                                       self.quant_max,
                                                                                                                       scale, zp)

    @torch.jit.export
    def reset(self) -> None:
        self.register_parameter('scale', None) 
        self.register_parameter('zero_point', None) 
        self.register_buffer('_current_batch', torch.tensor(0))
        self._do_param_init = False
        self._init_n_batches = -1 if self.dtype == torch.qint8 else self.n_batches
        self._initialized = False
        self._disable_fake_quant = False


    def _init_weights(self, x: Tensor, ex_scale=None, ex_zp=None) -> None:
        if not self._initialized:
            self._initialized = True
            size = x.shape[self.ch_axis] if self._is_perchannel else 1
            size = (size,)
            dev = x.device
            self.scale = torch.nn.Parameter(torch.full(size, self.init_scale, dtype=torch.float32, requires_grad=True).to(dev))
            if self.dtype == torch.qint8:
                reduction_axes = list(range(x.ndim))
                del reduction_axes[self.ch_axis]
                bitness = ceil(log(self.quant_max - self.quant_min)/log(2)) - 1
                mean = x.mean().unsqueeze(0) if size[0] == 1 else torch.mean(x, reduction_axes)
                std = x.std().unsqueeze(0) if size[0] == 1 else torch.std(x, reduction_axes)
                self.scale = torch.nn.Parameter(torch.max(torch.abs(mean-3*std), torch.abs(mean+3*std)) / 2**bitness)
            zp_val = self.init_zero_point if self._is_affine else 0
            self.zero_point = torch.nn.Parameter(torch.full(size, zp_val, dtype=torch.float32).to(dev))
            self.zero_point.requires_grad = self._is_affine
            if ex_scale is not None:
                assert ex_scale.numel() == size[0], f'External Observer return scale with shape {ex_scale.shape} when awaited {size}. Check the config of External Observer properly'
                self.scale = torch.nn.Parameter(ex_scale.view(size).to(torch.float32).to(dev))
            if ex_zp is not None:
                assert ex_zp.numel() == size[0], f'External Observer return zero point with shape {ex_zp.shape} when awaited {size}. Check the config of External Observer properly'
                self.zero_point = torch.nn.Parameter(ex_zp.view(size).to(torch.float32).to(dev))
                self.zero_point.requires_grad = self._is_affine
            
            

    @torch.jit.export
    def enable_fake_quant(self) -> None:
        self._disable_fake_quant = False
        

    @torch.jit.export
    def disable_fake_quant(self) -> None:
        self._disable_fake_quant = True

    @torch.jit.export
    def enable_observer(self) -> None:
        if self.scale is not None:
            self.scale.requires_grad = True
        if self.zero_point is not None:
            self.zero_point.requires_grad = True
       
    @torch.jit.export
    def disable_observer(self) -> None:
        if self.scale is not None:
            self.scale.requires_grad = False
        if self.zero_point is not None:
            self.zero_point.requires_grad = False

    @torch.jit.export
    def calculate_qparams(self) -> Tuple[Tensor, Tensor]:
        if not self._initialized:
            print("Scale and Zero Point are not initialized properly, because  LSQObserver was never called.\
                   You must at least run model on random tensor, before calling convert!\
                   Returned init_scale and init_zero_point")
            return self.init_scale, self.init_zero_point
        scale = torch.max(self.scale.detach().clone().cpu(), torch.tensor(torch.finfo(torch.float32).eps))
        if self.qscheme == torch.per_tensor_symmetric or self.qscheme == torch.per_channel_symmetric:
            zp_val = 0
            if self.dtype == torch.quint8:
                zp_val = (self.quant_min + self.quant_max) // 2 if self.has_customized_qrange else 128
            zero_point = torch.full_like(scale, zp_val, dtype=torch.long) 
        else:
            zero_point = self.zero_point.detach().clone().cpu() if self.zero_point is not None else torch.tensor([self.init_zero_point])
            zero_point = torch.clamp(zero_point, self.quant_min, self.quant_max).to(torch.int64)
        return scale, zero_point