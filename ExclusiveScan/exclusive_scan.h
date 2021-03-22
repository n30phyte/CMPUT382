#ifndef EXCLUSIVE_SCAN_H
#define EXCLUSIVE_SCAN_H

#include <cstdio>

#define BLOCK_SIZE 512

#define wbCheck(ans) gpuAssert((ans), __FILE__, __LINE__)

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

__global__ void exclusiveScan(const float* input, float* output, float* S, int N);

__global__ void auxMerge(const float* offsets, float* input, int N);

void recursiveScan(float* input, float* output, int numInputs);

#endif
