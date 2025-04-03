#ifndef _CUDA_UTILS_H
#define _CUDA_UTILS_H

#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <float.h>

#define THREADS_PER_BLOCK 1024
#define DIVUP(m, n) (((m) + (n) - 1) / (n))
#define NU 256000  
#define BLOCKS DIVUP(NU, THREADS_PER_BLOCK)  
#define TOTAL_THREADS (BLOCKS * THREADS_PER_BLOCK)  
#define SHARED_MEM_SIZE (256 * sizeof(float))


inline int opt_n_threads(int work_size) {
    const int pow_2 = std::log(static_cast<double>(work_size)) / std::log(2.0);

    return max(min(1 << pow_2, TOTAL_THREADS), 1);
}

#define CUDA_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line) {
    if (code != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s %s:%d\n",
                cudaGetErrorString(code), file, line);
        exit(code);
    }
}



#define CUBLAS_CHECK(call)                                                          \
    {                                                                               \
        cublasStatus_t status = call;                                               \
        if (status != CUBLAS_STATUS_SUCCESS) {                                      \
            std::cerr << "cuBLAS error: " << status << " at "                       \
                      << __FILE__ << ":" << __LINE__ << std::endl;                  \
            exit(EXIT_FAILURE);                                                     \
        }                                                                           \
    }


#define CUSOLVER_CHECK(call)                                                        \
    {                                                                               \
        cusolverStatus_t status = call;                                             \
        if (status != CUSOLVER_STATUS_SUCCESS) {                                    \
            std::cerr << "cuSolver error: " << status << " at "                     \
                      << __FILE__ << ":" << __LINE__ << std::endl;                  \
            exit(EXIT_FAILURE);                                                     \
        }                                                                           \
    }

#endif
