#ifndef TORCHLSQ
#define TORCHLSQ


#include <cstdint>
#include "macros.h"

namespace quantops {
    API_EXPORT int64_t cuda_version();

namespace detail {
        //(Taken from torchvision)
        int64_t _cuda_version = cuda_version();

} 
} 

#endif 
