#include <iostream>
#include <cuda.h>

#include "libwb/wb.h"

__device__ int binarySearch(const int value, const int *A, const int N) {
    int start = 0;
    int end = N - 1;

    int location = N;

    while (start <= end) {
        int middle = (start + end) / 2;

        // Target is larger than middle, move start.
        if (A[middle] <= value) {
            start = middle + 1;
        } else {
            location = middle;
            end = middle - 1;
        }
    }
    return location;
}

__device__ int linearSearch(const int value, const int *A, const int N) {
    int i;

    for (i = 0; i < N; i++) {
        if (A[i] > value) {
            return i + 1;
        }
    }

    return i;
}

__global__ void merge(int *C, const int *A, const int *B, const int N) {
    int threadId = blockDim.x * blockIdx.x + threadIdx.x;
    // Do A
    int j = binarySearch(A[threadId], B, N);
    C[threadId + j] = A[threadId];
    // Do B
    int i = binarySearch(B[threadId], A, N);
    C[threadId + i] = B[threadId];
}

int main(int argc, char **argv) {
    wbArg_t args;
    int N;
    int *A;
    int *B;
    int *C;
    int *deviceA;
    int *deviceB;
    int *deviceC;

    args = wbArg_read(argc, argv);

    wbTime_start(Generic, "Importing data and creating memory on host");
    A = (int *) wbImport(wbArg_getInputFile(args, 0), &N, NULL, "Integer");
    B = (int *) wbImport(wbArg_getInputFile(args, 1), &N, NULL, "Integer");
    C = (int *) malloc(2 * N * sizeof(int));
    wbTime_stop(Generic, "Importing data and creating memory on host");

    wbLog(TRACE, "The input length is ", N);

    int threads = 256;
    int blocks = N / threads + ((N % threads == 0) ? 0 : 1);

    wbTime_start(GPU, "Allocating GPU memory.");
    cudaMalloc((void **) &deviceA, N * sizeof(int));
    cudaMalloc((void **) &deviceB, N * sizeof(int));
    cudaMalloc((void **) &deviceC, 2 * N * sizeof(int));
    wbTime_stop(GPU, "Allocating GPU memory.");


    wbTime_start(GPU, "Copying input memory to the GPU.");
    cudaMemcpy(deviceA, A, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceB, B, N * sizeof(int), cudaMemcpyHostToDevice);
    wbTime_stop(GPU, "Copying input memory to the GPU.");

    // Perform on CUDA.
    const dim3 blockSize(threads, 1, 1);
    const dim3 gridSize(blocks, 1, 1);

    wbTime_start(Compute, "Performing CUDA computation");
    merge <<<gridSize, blockSize>>>(deviceC, deviceA, deviceB, N);
    cudaDeviceSynchronize();
    wbTime_stop(Compute, "Performing CUDA computation");

    wbTime_start(Copy, "Copying output memory to the CPU");
    cudaMemcpy(C, deviceC, 2 * N * sizeof(int), cudaMemcpyDeviceToHost);
    wbTime_stop(Copy, "Copying output memory to the CPU");

    wbTime_start(GPU, "Freeing GPU Memory");
    cudaFree(deviceA);
    cudaFree(deviceB);
    cudaFree(deviceC);
    wbTime_stop(GPU, "Freeing GPU Memory");

    wbSolution(args, C, 2 * N);

    free(A);
    free(B);
    free(C);

#if LAB_DEBUG
    system("pause");
#endif

    return 0;
}
