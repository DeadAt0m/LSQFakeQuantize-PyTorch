import torch
from .modules.observers import LSQFakeQuantizer


def disable_fake_quant(mod):
    if isinstance(mod, torch.quantization.FakeQuantize) or isinstance(mod, LSQFakeQuantize):
        mod.disable_fake_quant()

def enable_fake_quant(mod):
    if isinstance(mod, torch.quantization.FakeQuantize) or isinstance(mod, LSQFakeQuantize):
        mod.enable_fake_quant()

def disable_observer(mod):
    if isinstance(mod, torch.quantization.FakeQuantize) or isinstance(mod, LSQFakeQuantize):
        mod.disable_observer()

def enable_observer(mod):
    if isinstance(mod, torch.quantization.FakeQuantize) or isinstance(mod, LSQFakeQuantize):
        mod.enable_observer()

def disable_fake_quant_on_act(mod):
    if isinstance(mod, torch.quantization.FakeQuantize) or isinstance(mod, LSQFakeQuantize) and mod.dtype==torch.quint8:
        mod.disable_fake_quant()

def enable_fake_quant_on_act(mod):
    if isinstance(mod, torch.quantization.FakeQuantize) or isinstance(mod, LSQFakeQuantize) and mod.dtype==torch.quint8:
        mod.enable_fake_quant()

def disable_observer_on_weights(mod):
    if isinstance(mod, torch.quantization.FakeQuantize) or isinstance(mod, LSQFakeQuantize) and mod.dtype==torch.qint8:
        mod.disable_observer()

def enable_observer_on_weights(mod):
    if isinstance(mod, torch.quantization.FakeQuantize) or isinstance(mod, LSQFakeQuantize) and mod.dtype==torch.qint8:
        mod.enable_observer()

