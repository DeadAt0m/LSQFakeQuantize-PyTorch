#pragma once
 

#if defined(TORCH18)
    #define TS_TORCH_LIBRARY_FRAGMENT(ns,m) TORCH_LIBRARY_FRAGMENT(ns, m)
#elif defined(TORCH17)
    #define TS_TORCH_LIBRARY_FRAGMENT(ns,m) TORCH_LIBRARY_FRAGMENT_THIS_API_IS_FOR_PER_OP_REGISTRATION_ONLY(ns, m)
#else
   #define TS_TORCH_LIBRARY_FRAGMENT(ns,m) TORCH_LIBRARY_FRAGMENT(ns, m)
#endif

#ifdef _WIN32
#if defined(TORCHSRROPS_EXPORTS)
#define API_EXPORT __declspec(dllexport)
#else
#define API_EXPORT __declspec(dllimport)
#endif
#else
#define API_EXPORT
#endif


