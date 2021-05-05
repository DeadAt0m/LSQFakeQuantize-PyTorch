#include "torchlsq.h"

#include <Python.h>
#include <pybind11/pybind11.h>
#include <torch/library.h>

#include "ops/ops.h"

#ifdef WITH_CUDA
    #include <cuda.h>
#endif



#ifdef _WIN32
    #if PY_MAJOR_VERSION < 3
        PyMODINIT_FUNC init_C(void) {return NULL;}
    #else
        PyMODINIT_FUNC PyInit__C(void) {return NULL;}
    #endif
#endif

namespace quantops {
    
    int64_t cuda_version() {
        #ifdef WITH_CUDA
            return CUDA_VERSION;
        #else
            return -1;
        #endif
    }

}

TS_TORCH_LIBRARY_FRAGMENT(torchlsq, m) {
        m.def("_cuda_version", &quantops::cuda_version);
        m.def("lsq", &quantops::ops::lsq);   
        
}

