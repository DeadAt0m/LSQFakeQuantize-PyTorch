'''
    This file was partly taken from torchvision
'''

_HAS_OPS = False
error_str = ''


def _has_ops():
    return False

def _register_extensions():
    from pathlib import Path 
    import sys
    import os
    import importlib
    import torch

    # load the custom_op_library and register the custom ops
    lib_dir = Path(__file__).resolve().parent
    if os.name == 'nt':
        # Register the main torchlsq library location on the default DLL path
        import ctypes
        kernel32 = ctypes.WinDLL('kernel32.dll', use_last_error=True)
        with_load_library_flags = hasattr(kernel32, 'AddDllDirectory')
        prev_error_mode = kernel32.SetErrorMode(0x0001)
        if with_load_library_flags:
            kernel32.AddDllDirectory.restype = ctypes.c_void_p
        if sys.version_info >= (3, 8):
            os.add_dll_directory(str(lib_dir))
        elif with_load_library_flags:
            res = kernel32.AddDllDirectory(str(lib_dir))
            if res is None:
                err = ctypes.WinError(ctypes.get_last_error())
                err.strerror += f' Error adding "{str(lib_dir)}" to the DLL directories.'
                raise err
        kernel32.SetErrorMode(prev_error_mode)

    loader_details = (importlib.machinery.ExtensionFileLoader,
                      importlib.machinery.EXTENSION_SUFFIXES)
    extfinder = importlib.machinery.FileFinder(str(lib_dir), loader_details)
    ext_specs = extfinder.find_spec("_C")
    if ext_specs is None:
        raise ImportError
    torch.ops.load_library(ext_specs.origin)


try:
    _register_extensions()
    _HAS_OPS = True

    def _has_ops():
        return True
except (ImportError, OSError) as e:
    error_str = str(e)
    pass


def _assert_has_ops():
    if not _has_ops():
        raise RuntimeError(
            "Couldn't load custom C++ ops. This can happen if your PyTorch and "
            "torchlsq versions are incompatible, or if you had errors while compiling "
            "torchlsq from source. For further information on the compatible versions, check "
            "https://github.com/DeadAt0m/LSQ-PyTorch/blob/master/README.md for the compatibility matrix. "
            "Please check your PyTorch version with torch.__version__ and verify if it is compatible, and if not "
            "please reinstall your PyTorch."
            f"\n\nImport error details:\n\t{error_str}"
        )

def _check_cuda_version():
    """
    Make sure that CUDA versions match between the pytorch install and torchlsq install
    """
    if not _HAS_OPS:
        return -1
    import torch
    _version = torch.ops.torchlsq._cuda_version()
    if _version != -1 and torch.version.cuda is not None:
        ts_version = str(_version)
        if int(ts_version) < 10000:
            ts_major = int(ts_version[0])
            ts_minor = int(ts_version[2])
        else:
            ts_major = int(ts_version[0:2])
            ts_minor = int(ts_version[3])
        t_version = torch.version.cuda
        t_version = t_version.split('.')
        t_major = int(t_version[0])
        t_minor = int(t_version[1])
        if t_major != ts_major or t_minor > ts_minor:
            raise RuntimeError("Detected that PyTorch and torchlsq were compiled with different CUDA versions. "
                               "PyTorch has CUDA Version={}.{} and torchlsq has CUDA Version={}.{}. "
                               "Please reinstall the torchlsq that matches your PyTorch install."
                               .format(t_major, t_minor, ts_major, ts_minor))
    return _version

_check_cuda_version()
