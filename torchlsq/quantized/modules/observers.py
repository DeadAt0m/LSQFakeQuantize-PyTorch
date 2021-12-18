import torch
from torch._C import dtype
from torchlsq.functional import lsq
from math import log, ceil, copysign
from typing import Tuple
import inspect


Tensor = torch.Tensor

#### GLOBAL CONSTANTS #####
OTYPES = {'weight': 0,
          'activation': 1}
TYPES_RANGE_MAPPING = {
                        torch.qint8: {'range': (-128,127), 'bitness': 8, 'unsigned':False},
                        torch.quint8:{'range': (0, 255), 'bitness': 8, "unsigned": True}
                      }
QSCHEMES = (
            torch.per_tensor_affine,
            torch.per_tensor_symmetric,
            torch.per_channel_affine,
            torch.per_channel_symmetric,
           )
def IS_QSCHEME_PER_CHANNEL(qscheme):
    assert qscheme in QSCHEMES, f"Only following schemes supported {QSCHEMES} but recieved {qscheme}"
    return qscheme in (torch.per_channel_affine, torch.per_channel_symmetric)
def IS_QSCHEME_AFFINE(qscheme):
    assert qscheme in QSCHEMES, f"Only following schemes supported {QSCHEMES} but recieved {qscheme}"
    return qscheme in (torch.per_tensor_affine, torch.per_channel_affine)
def IS_QSCHEME_PER_TENSOR(qscheme):
    return not IS_QSCHEME_PER_CHANNEL(qscheme)
def IS_QSCHEME_SYMMETRIC(qscheme):
    return not IS_QSCHEME_AFFINE(qscheme)
##############################


# REDEFINITE THE OBSERVER BASE FOR ITS COMPATIBILITY WITH PICKLE
class _PartialWrapper(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, *args, **keywords):
        return self.p(*args, **keywords)

    def __repr__(self):
        return self.p.__repr__()
   

def _with_args(cls_or_self, **kwargs):
    r"""Wrapper that allows creation of class factories.

    This can be useful when there is a need to create classes with the same
    constructor arguments, but different instances.

    Example::

        >>> Foo.with_args = classmethod(_with_args)
        >>> foo_builder = Foo.with_args(a=3, b=4).with_args(answer=42)
        >>> foo_instance1 = foo_builder()
        >>> foo_instance2 = foo_builder()
        >>> id(foo_instance1) == id(foo_instance2)
        False
    """
    r = _PartialWrapper(partial(cls_or_self, **kwargs))
    r.with_args = _with_args
    return r


class ObserverBase(torch.quantization.observer.ObserverBase):
    with_args = classmethod(_with_args)

class LSQFakeQuantizer(ObserverBase):
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

                w.r.t input: 1,  if ((quant_min < x_q) && (x_q < quant_max))
                             0,  otherwise

                w.r.t scale: (x_r - x)/scale, if ((quant_min < x_q) && (x_q < quant_max)) 
                             quant_min - zero_point,    if x_q <= quant_min
                             quant_max - zero_point,    if x_q >= quant_max

                w.r.t shfit:  0,  if ((quant_min < x_q) && x_q < quant_max))
                              1,  otherwise

        Based on article we support scaling of derivatives on `scale` and `shift` on 1/sqrt(n_elements(x)*quant_max)
        (See https://arxiv.org/abs/1902.08153 for details)


        Pytorch Quantization assumes using qint8 for weights quantization and quint8 for activation.
        Based on that, qint8 dtype considered as Observer for weights and quint8 for activation. 


        This logic has strict influence on 'scale' initialization.
            - Weight case initialization:
                Is always happens staticly following by:
                s_init = max(|µ − 3 ∗ σ|, |µ + 3 ∗ σ|)/2^b−1, where µ and σ  mean and std of weights tensor correspondingly.

            - Activation case initializations:
                a) In original article the authors propose to init `scale` and `shift` for affine case by minimizing: ||x_r-x||F^2
                Instead of using it as additional loss, we incorporate this natively inside code,
                due to this function supposed to be work inside Pytorch Observer class.
                
                b) Alternatively, their is possibility to initialize `scale` and `shift` via  observer.
                   We highly suggest use the MovingAverage(PerChannel)MinMaxObserver to track running min and max values
                
                For control a) and b) use `init_mode` argument during initialization 
                The number of batches during which initialization happens controls by `n_batches`.
                

        LSQ can work with any bitrate, however for Pytorch compatiblity, we limited bitrate from above to 8-bit.

        By default quant_min and quant_max will be initialized for 7-bit,
        for avoiding overflow during quantized inference in Pytorch.
            qint8  - quant_min, quant_max = -64, 63
            quint8 - quant_min, quant_max = 0, 127
        To overcome this, look on `avoid_torch_overflow` argument.


        Arguments:
            observer(torch.quantization.ObserverBase) - Module for observing statistics on input tensors and calculating scale and zero-point.
            otype(str) - Type specification for which LSQ will used. Can be one of ('weight', 'activation')
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
    ''' 
    init_modes = ('learnable', 'observer')
    
    @staticmethod
    def sign(x):
        return copysign(1,x)
    
    def __init__(self, observer, otype,
                 dtype=torch.quint8, 
                 qscheme=torch.per_tensor_affine,
                 quant_min=None, quant_max=None, 
                 init_scale=1., init_shift=0.,
                 ch_axis=None, learn_params=True, 
                 init_batches=1000, init_mode='observer', 
                 use_grad_scaling=True, grad_scaler=1.,
                 avoid_torch_overflow=True, debug_mode=False, **observer_kwargs):
        super().__init__(dtype)
        assert init_mode in ('learnable', 'observer'), f'only following modes available: {("learnable", "observer")}'
        self.activation_post_process = None
        if init_mode == 'observer':
            assert inspect.isclass(observer), 'awaited Observer class not instance or function wrapper'
            # create proper observe_kwargs
            req_keys = set(inspect.signature(observer.__init__).parameters.keys())
            observer_kwargs['reduce_range'] = avoid_torch_overflow 
            src_keys = set(self.__init__.__code__.co_varnames + tuple(observer_kwargs.keys()))
            avail_keys = req_keys.intersection(src_keys)
            avail_keys.remove('self')
            new_observer_kwargs = dict()
            for k in avail_keys:
                new_observer_kwargs[k] = locals()[k] if k in locals() else observer_kwargs[k] 
            self.activation_post_process = observer(**new_observer_kwargs)
     
        assert otype in tuple(OTYPES.keys()), f'otype must be on of {tuple(OTYPES.keys())}, but {self.otype} is given'
        self.otype = OTYPES[otype]

        assert self.dtype in tuple(TYPES_RANGE_MAPPING.keys()), f"Default Observer only works for {tuple(TYPES_RANGE_MAPPING.keys())} data types"

        self.qscheme = qscheme
        self.ch_axis = ch_axis
        if self.ch_axis is None:
            # 0 - for weights, 1 - for activation
            self.ch_axis = int(bool(self.otype))
        
        self.init_mode = init_mode
        self.n_batches = init_batches 
        self.use_grad_scaling = use_grad_scaling
        self.grad_scaler = grad_scaler
        self.debug_mode = debug_mode

        self.is_perchannel = IS_QSCHEME_PER_CHANNEL(self.qscheme)
        self.is_affine =  IS_QSCHEME_AFFINE(self.qscheme)
        self.init_scale = init_scale
        self.init_shift = init_shift
        self.quant_min, self.quant_max = self._verify_qmin_qmax(quant_min, quant_max, lowbit=avoid_torch_overflow)
        self.reset(learn_params=learn_params)

    def _verify_qmin_qmax(self, quant_min: int, quant_max: int, lowbit=True) -> Tuple[int, int]:
        """ Stealed from PyTorch itself.
            Calculates actual qmin and qmax based on the quantization range,
        observer datatype and if range is reduced.
        """
        # Restrictions to work with PyTorch quantized intrinstics
        if self.otype  == 0: # weights
            assert not self.is_affine, 'We support only symmetric scheme for weight'
            assert self.dtype == torch.qint8, 'Pytorch quantized operations implementaion requires `qint8` type for weights'
        elif self.otype == 1: # activation
            assert self.dtype == torch.quint8, 'Pytorch quantized operations implementaion requires `quint8` type for activation'
        ##################
        max_bitness = TYPES_RANGE_MAPPING[self.dtype]['bitness']
        self.has_customized_qrange = (quant_min is not None) and (quant_max is not None)
        if self.has_customized_qrange:
            assert quant_min <= 0 <= quant_max, "User-specified quantization range must include 0."
            assert quant_min < quant_max, "qmin must be strictly less than qmax for user-specified quantization range."
            qrange_len = quant_max - quant_min + 1
            assert 0 < qrange_len <= int(2 ** (max_bitness - int(lowbit)) ), \
                f"quantization range should be positive and not exceed the maximum bit range (=2^{max_bitness - int(lowbit)})."
        else:
            init_range = torch.tensor([0, 2**(max_bitness - int(lowbit)) - 1])
            if not TYPES_RANGE_MAPPING[self.dtype]['unsigned']: # shift range in symmetric case
                init_range = init_range - 2**(max_bitness - 1 - int(lowbit))
            quant_min, quant_max = init_range.tolist()
        # override the init_shift for symmetric case
        if not self.is_affine:
            s = self.sign(quant_min + quant_max)
            self.init_shift = -float(abs(quant_min + quant_max) // 2) * s * self.init_scale
        return quant_min, quant_max

    @torch.jit.export
    def reset(self, learn_params=True) -> None:
        self.n_batches = -1 if self.otype == 0 else self.n_batches  # in case of weights the the LSQ param initialization will happens statically
        self._initialized = False

        self.register_parameter('scale', None) 
        self.register_parameter('shift', None) 
        
        self.register_buffer('fake_quant_enabled', torch.tensor([1], dtype=torch.uint8))
        self.register_buffer('observer_enabled', torch.tensor([1], dtype=torch.uint8))
        self.register_buffer('learning_enabled', torch.tensor([int(learn_params)], dtype=torch.uint8))
        self.register_buffer('current_batch', torch.tensor([0], dtype=torch.int64))

        self.enable_observer() # tricky calling for proper observer enabling (just to avoid code duplications)

    def check_is_init_mode(self):
        is_init = bool(self.learning_enabled[0])
        is_init *= (self.otype != 0)
        is_init *= (self.current_batch[0] <= self.n_batches)
        return is_init
        
    @torch.jit.export
    def enable_observer(self) -> None:
        self.observer_enabled[0] = 1
        if self.learning_enabled[0] == 1 and self.otype == 0:
            self.observer_enabled[0] = 0 # when model weights learned via LSQ we don't need the observer 
        if self.learning_enabled[0] == 1 and self.otype == 1:
            # when model activation learned via lsq there is some cases where observer is not needed
            if self.init_mode == 'learnable': 
                 self.observer_enabled[0] = 0 # params learned via backprop, we don't need the observer
            elif self.init_mode == 'observer' and self.current_batch[0] > self.n_batches:
                # observer is also doesn't needed when initialization already happens via observer
                # and someone whant to turn it on (just increase n_batches in new runs )
                self.observer_enabled[0] = 0

    @torch.jit.export
    def disable_observer(self) -> None:
        self.observer_enabled[0] = 0

    @torch.jit.export
    def enable_fake_quant(self) -> None:
        self.fake_quant_enabled[0] = 1

    @torch.jit.export
    def disable_fake_quant(self) -> None:
        self.fake_quant_enabled[0] = 0

    @torch.jit.export
    def enable_param_learning(self):
        """
            Enables learning of quantization parameters and
            disables static observer estimates. Forward path returns fake quantized X.
            (partly taken from PyTorch code)
        """
        self.learning_enabled[0] = 1
        self.disable_observer()
        self.n_batches = -1 # assumed that params where statically observed some amount of time, hence init is not needed

    @torch.jit.export
    def enable_static_estimate(self):
        """
            Enables static observer estimates and disbales learning of
            quantization parameters. Forward path returns fake quantized X.
            (partly taken from PyTorch code)
        """
        self.learning_enabled[0] = 0
        self.enable_observer()



    def _init_weights(self, x: Tensor, _init_device=torch.device('cpu')) -> None:
        '''
            Dynamically initialized weights for first time, required the example of input tensor.

            !!! Please take to account, due to this function you need to pass params to optimizator, only after 1st forward pass !!!

            Arguments:
                x(torch.Tensor) - input tensor, which used for dynamic parameter initialization
                _init_device(torch.device) - device for technical initialization . Default: CPU
        '''
        self._initialized = True
        size = x.shape[self.ch_axis] if (self.is_perchannel and x is not None) else 1
        size = (size,)
        device = x.device if x is not None else _init_device
        scale = torch.full(size, self.init_scale, dtype=torch.float32).to(device)
        if self.otype == 0 and x is not None: # init scale properly for weights
            x = x.detach()
            reduction_axes = list(range(x.ndim))
            del reduction_axes[self.ch_axis]
            bitness = ceil(log(self.quant_max - self.quant_min)/log(2)) - 1
            with torch.no_grad():
                mean = x.mean().unsqueeze(0) if size[0] == 1 else torch.mean(x, reduction_axes)
                std = x.std().unsqueeze(0) if size[0] == 1 else torch.std(x, reduction_axes)
                scale = (torch.max(torch.abs(mean-3*std), torch.abs(mean+3*std)).to(device) / 2**bitness).to(torch.float32)
        shift = torch.full(size, self.init_shift, dtype=torch.float32).to(device)
        self.scale = torch.nn.Parameter(scale)
        self.shift = torch.nn.Parameter(shift)
        self.scale.requires_grad = bool(self.learning_enabled[0])
        self.shift.requires_grad = bool(self.learning_enabled[0]) and self.is_affine



    def _set_weights(self, scale=None, shift=None, zero_point=None, _init_device=torch.device('cpu')):
        '''
            Safely set LSQ params such scale and shift.

            Arguments:
                scale(torch.tensor) - scale tensor, which data will be copied to inner scale parameter
                shift(torch.tensor) - shift tensor, which data will be copied to inner shift parameter
                zero_point(torch.tensor) - will be converted to shift first, and than copied to inner shift parameter
                _init_device(torch.device) - device for technical initialization (read comments below). Default: CPU
        '''
        if self.scale is None:
            # technical init if required (happens in situation when set_wights call instantly after class initialization)
            # work in per_tensor mode only!
            self._init_weights(None, _init_device=_init_device)
        
        if scale is not None:
            with torch.no_grad():
                scale = scale.to(self.scale.device).to(self.scale.dtype)
                scale.resize_(self.scale.shape)
            self.scale.data.copy_(scale)
        if zero_point is not None:
            with torch.no_grad(): # conversion from zero point to shift
                shift = -zero_point.to(self.scale.device)*self.scale.detach()
        if shift is not None:
            with torch.no_grad():
                shift = shift.to(self.shift.device).to(self.shift.dtype)
                shift.resize_(self.shift.shape)
            self.shift.data.copy_(shift)
   
    def set_weights(self, scale, zero_point=None, _init_device=torch.device('cpu')):  
        self._set_weights(scale, shift=None, zero_point=zero_point, _init_device=_init_device)    

    @staticmethod
    def convert_shift_to_zp(shift, scale, dtype):
        '''
            PyTorch use the following formula for quantization x_q = clamp(x/scale + zero_point, qmin, qmax),
            where zero_point - is usually has same type as x_q (i.e qint8 or quint8)

            But, for LSQ used another quantization approach: x_q = clamp((x - shift)/scale, qmin, qmax),
            where shift is float.

            This function is properly converts shift to zero_point (zero_point = round(-shift/scale)).

            Arguments:
                shift(torch.tensor) - quantization shift used in LSQ
                scale(torch.tensor) - quantization scale
                dtype(torch.dtype) - dtype to infer type numerical bounds.

            Returns:
                zero_point(torch.tensor) - zero_point, which has integer type.
        '''
        tmin, tmax = TYPES_RANGE_MAPPING[dtype]['range']
        with torch.no_grad():
            zero_point = -shift/scale
            zero_point.round_().clamp_(min=tmin, max=tmax)
            return zero_point.to(torch.int64)

    @torch.jit.export
    def calculate_qparams(self, verbose=True, need_shift=False) -> Tuple[Tensor, Tensor]:
        if not self._initialized:
            if verbose:
                print("Scale and Zero Point are not initialized properly, because  LSQObserver was never called.\
                       You must at least run model on random tensor, before calling convert!\
                       Returned init_scale and init_zero_point")
            # hack for zero_point fast init
            zp = self.convert_shift_to_zp(torch.tensor(self.init_shift),
                                          torch.tensor(self.init_scale), 
                                          self.dtype).item()
            if need_shift:
                return self.init_scale, self.init_shift, zp
            return self.init_scale, zp
        scale = torch.max(self.scale.detach().clone().cpu(), torch.tensor(torch.finfo(torch.float32).eps))
        shift = self.shift.detach().clone().cpu()
        zero_point = self.convert_shift_to_zp(shift, scale, self.dtype)
        if need_shift:
            return scale, shift, zero_point
        return scale, zero_point

    def forward(self, x):
        if self.debug_mode: # do nothing in debug mode
            return x
        if not self._initialized:
            self._init_weights(x)
            return x # instantly return x during weights initialization initialization
        do_backprop_init = False
        do_full_lsq = bool(self.learning_enabled[0])
        # initialization conditions 
        if self.current_batch[0] <= self.n_batches and\
           self.training and\
           self.learning_enabled[0] == 1: # big IF for big deals
            if self.init_mode == 'observer': # force LSQ work like FakeQuantizer in this mode
                do_full_lsq = False 
                if self.current_batch[0]  == self.n_batches:
                    do_full_lsq = True
                    self.disable_observer()
            elif self.init_mode == 'learnable': # force LSQ work in param init via backprob mode
                self.disable_observer() # turn off the observer because it not needed in this mode
                do_backprop_init = self.current_batch[0]  != self.n_batches
            self.current_batch[0] += 1
                
        if self.observer_enabled[0] == 1:
            self.activation_post_process(x.detach())
            scale, zero_point = self.activation_post_process.calculate_qparams()
            self._set_weights(scale=scale, zero_point=zero_point)
            
        if self.fake_quant_enabled[0] == 1:
            do_backprop_init *= do_full_lsq # protection for parameter initalization via backward
            tmin, tmax = TYPES_RANGE_MAPPING[self.dtype]['range']
            # do not need to learn params when work like FakeQuantizer
            self.scale.requires_grad = do_full_lsq
            self.shift.requires_grad = do_full_lsq and self.is_affine

            return lsq(x, self.scale, self.shift, self.quant_min, self.quant_max, tmin, tmax,
                           self.ch_axis, self.use_grad_scaling, self.grad_scaler,
                           self.is_affine, self.is_perchannel,
                           eval_mode=(not do_full_lsq), init_mode=bool(do_backprop_init))
        return x    
    
    @torch.jit.export
    def extra_repr(self):
        is_initialized = '' if self._initialized else '(Uninitialized!) '
        scale, shift, zp = self.calculate_qparams(verbose=False, need_shift=True)
        for_weights = 'weights' if self.otype == 0 else 'activation' 
        per_channel = 'No' if not self.is_perchannel else f'Yes, channel axis - {self.ch_axis}'
        is_initialized += f'(Observer in parameter init mode: {self.init_mode}; {self.current_batch[0]}/{self.n_batches} batches left) ' if self.check_is_init_mode() else ''
        if self.debug_mode:
            return 'Debug mode: ON, doing nothing.'
        info_str = "{}Observer for {}; Learnable:{}; Observer:{}; FakeQuant:{}; "\
                   "Qtype:{}, Affine:{}, PerChannel:{}, Qrange:[{},{}], scale={}, zero_point={} (shift={})."
        torch.set_printoptions(threshold=8)
        info_str = info_str.format(is_initialized, for_weights, bool(self.learning_enabled[0]),
                                   bool(self.observer_enabled[0]), bool(self.fake_quant_enabled[0]),
                                   self.dtype, self.is_affine, per_channel,
                                   self.quant_min, self.quant_max, scale, zp, shift)
        if hasattr(self, 'recalibrated'):
            info_str += '\nModule was recalibrated!'
        torch.set_printoptions(threshold=1000)
        return info_str
