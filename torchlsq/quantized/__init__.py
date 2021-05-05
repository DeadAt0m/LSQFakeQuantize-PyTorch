import torch
from .modules.observers import LSQObserver


def disable_fake_quant(mod):
    if (type(mod) == torch.quantization.FakeQuantize or type(mod) == LSQObserver):
        mod.disable_fake_quant()

def enable_fake_quant(mod):
    if (type(mod) == torch.quantization.FakeQuantize or type(mod) == LSQObserver):
        mod.enable_fake_quant()

def disable_observer(mod):
    if (type(mod) == torch.quantization.FakeQuantize or type(mod) == LSQObserver):
        mod.disable_observer()

def enable_observer(mod):
    if (type(mod) == torch.quantization.FakeQuantize or type(mod) == LSQObserver):
        mod.enable_observer()

def disable_fake_quant_on_act(mod):
    if (type(mod) == torch.quantization.FakeQuantize or type(mod) == LSQObserver) and mod.dtype==torch.quint8:
        mod.disable_fake_quant()

def enable_fake_quant_on_act(mod):
    if (type(mod) == torch.quantization.FakeQuantize or type(mod) == LSQObserver) and mod.dtype==torch.quint8:
        mod.enable_fake_quant()

def disable_observer_on_weights(mod):
    if (type(mod) == torch.quantization.FakeQuantize or type(mod) == LSQObserver) and mod.dtype==torch.qint8:
        mod.disable_observer()

def enable_observer_on_weights(mod):
    if (type(mod) == torch.quantization.FakeQuantize or type(mod) == LSQObserver) and mod.dtype==torch.qint8:
        mod.enable_observer()

