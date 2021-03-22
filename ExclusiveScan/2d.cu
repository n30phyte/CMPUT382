#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "wb.h"
#include "exclusive_scan.h"

#define TRANSPOSE_TILE_DIM 32
#define TRANSPOSE_BLOCK_ROWS 8

__global__ void kernelPrint(float *array, int cols, int rows) {
    for (int j = 0; j < rows; j++) {
        for (int i = 0; i < cols; i++) {
            printf("%f ", array[j * cols + i]);
        }
        printf("\n");
    }
    printf("\n");
}

__global__ void transposeMatrix(float *input, float *output, int width, int height) {
    __shared__ float tile[TRANSPOSE_TILE_DIM][TRANSPOSE_TILE_DIM];

    int x = blockIdx.x * TRANSPOSE_TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TRANSPOSE_TILE_DIM + threadIdx.y;
    int inputIdx = x + y * width;
    if (x < width && y < height) {
        for (int i = 0; i < TRANSPOSE_TILE_DIM; i += TRANSPOSE_BLOCK_ROWS) {
            tile[threadIdx.y + i][threadIdx.x] = input[inputIdx + i * width];
        }
    }
    __syncthreads();

    x = blockIdx.y * TRANSPOSE_TILE_DIM + threadIdx.x;
    y = blockIdx.x * TRANSPOSE_TILE_DIM + threadIdx.y;
    int outputIdx = x + y * height;

    if (y < width && x < height) {

        for (int i = 0; i < TRANSPOSE_TILE_DIM; i += TRANSPOSE_BLOCK_ROWS) {
            output[outputIdx + i * height] = tile[threadIdx.x][threadIdx.y + i];
        }
    }
}

int main(int argc, char **argv) {

    wbArg_t args;
    float *hostInput;  // The input 1D list
    float *hostOutput; // The output list
    float *deviceInput;  // device input
    float *deviceTmpOutput;  // temporary output
    float *deviceOutput;  // output
    int numInputRows, numInputCols; // dimensions of the array

    args = wbArg_read(argc, argv);

    wbTime_start(Generic, "Importing data and creating memory on host");
    hostInput = (float *) wbImport(wbArg_getInputFile(args, 0), &numInputRows, &numInputCols);
    cudaHostAlloc(&hostOutput, numInputRows * numInputCols * sizeof(float), cudaHostAllocDefault);
    wbTime_stop(Generic, "Importing data and creating memory on host");

    wbLog(TRACE, "The dimensions of input are ",
          numInputRows, "x", numInputCols);

    wbTime_start(GPU, "Allocating GPU memory.");
    cudaMalloc((void **) &deviceInput, numInputRows * numInputCols * sizeof(float));
    cudaMalloc((void **) &deviceOutput, numInputRows * numInputCols * sizeof(float));
    cudaMalloc((void **) &deviceTmpOutput, numInputRows * numInputCols * sizeof(float));
    wbTime_stop(GPU, "Allocating GPU memory.");

    wbTime_start(GPU, "Clearing output memory.");
    cudaMemset(deviceOutput, 0, numInputRows * numInputCols * sizeof(float));
    wbTime_stop(GPU, "Clearing output memory.");

    wbTime_start(GPU, "Copying input memory to the GPU.");
    cudaMemcpy(deviceInput, hostInput, numInputRows * numInputCols * sizeof(float), cudaMemcpyHostToDevice);
    wbTime_stop(GPU, "Copying input memory to the GPU.");

    wbTime_start(Compute, "Performing CUDA computation");

    for (int i = 0; i < numInputRows; ++i) {
        recursiveScan(&deviceInput[numInputCols * i], &deviceTmpOutput[numInputCols * i], numInputCols);
    }

//    kernelPrint<<<1, 1>>>(deviceTmpOutput, numInputCols, numInputRows);
//    wbCheck(cudaDeviceSynchronize());

    dim3 transposeBlockDim(TRANSPOSE_TILE_DIM, TRANSPOSE_BLOCK_ROWS);
    dim3 transposeGridDim(ceil(numInputCols / (float) TRANSPOSE_TILE_DIM), ceil(numInputRows / (float) TRANSPOSE_TILE_DIM));
    transposeMatrix<<<transposeGridDim, transposeBlockDim>>>(deviceTmpOutput, deviceInput, numInputCols, numInputRows);
    wbCheck(cudaDeviceSynchronize());

//    kernelPrint<<<1, 1>>>(deviceInput, numInputCols, numInputRows);
//    wbCheck(cudaDeviceSynchronize());

    for (int i = 0; i < numInputCols; ++i) {
        recursiveScan(&deviceInput[numInputRows * i], &deviceTmpOutput[numInputRows * i], numInputRows);
    }

    // You can change TranposeBlockDim and TranposeGridDim, but if you use kernel suggested in the manual file, these should be the correct ones
    transposeBlockDim = dim3(TRANSPOSE_TILE_DIM, TRANSPOSE_BLOCK_ROWS);
    transposeGridDim = dim3(ceil(numInputRows / (float) TRANSPOSE_TILE_DIM), ceil(numInputCols / (float) TRANSPOSE_TILE_DIM));
    transposeMatrix<<<transposeGridDim, transposeBlockDim>>>(deviceTmpOutput, deviceOutput, numInputRows, numInputCols);
    wbCheck(cudaDeviceSynchronize());

    wbTime_stop(Compute, "Performing CUDA computation");

    wbTime_start(Copy, "Copying output memory to the CPU");
    cudaMemcpy(hostOutput, deviceOutput, numInputRows * numInputCols * sizeof(float), cudaMemcpyDeviceToHost);
    wbTime_stop(Copy, "Copying output memory to the CPU");

    wbTime_start(GPU, "Freeing GPU Memory");
    cudaFree(deviceInput);
    cudaFree(deviceTmpOutput);
    cudaFree(deviceOutput);
    wbTime_stop(GPU, "Freeing GPU Memory");

    wbSolution(args, hostOutput, numInputRows, numInputCols);

    free(hostInput);
    cudaFreeHost(hostOutput);

    cudaDeviceSynchronize();

#if LAB_DEBUG
    system("pause");
#endif

    return 0;
}
