#ifndef EXCLUSIVE_SCAN_H
#define EXCLUSIVE_SCAN_H

#define BLOCK_SIZE 512

#define wbCheck(ans) gpuAssert((ans), __FILE__, __LINE__)

__global__ void exclusiveScan(float* input, float* output, int N);

#endif