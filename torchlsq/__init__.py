import warnings
from pathlib import Path
from .extension import _HAS_OPS

try:
    from .version import __version__
except ImportError:
    pass

# Check if torchlsq is being imported within the root folder
if (not _HAS_OPS and Path(__file__).parent.resolve() == (Path.cwd() / 'torchlsq')):
    message = (f'You are importing torchlsq within its own root folder ({Path.cwd() / "torchlsq"}). '
               'This is not expected to work and may give errors. Please exit the '
               'torchlsq project source and relaunch your python interpreter.')
    warnings.warn(message)

from torchlsq.quantized import *
