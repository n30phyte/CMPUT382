#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "wb.h"

__global__ void matrixMultiply(float *A, float *B, float *C, int numARows,
                               int numAColumns, int numBRows, int numBColumns) {
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < numARows && col < numBColumns) {
        float sum = 0;

        for (auto k = 0; k < numAColumns; k++) {
            float a = A[row * numAColumns + k];
            float b = B[k * numBColumns + col];
            sum += (a * b);
        }

        C[row * numBColumns + col] = sum;
    }
}

int main(int argc, char **argv) {
    wbArg_t args;
    float *hostA; // The output C matrix
    float *hostB; // The output C matrix
    float *hostC; // The output C matrix
    float *deviceA;
    float *deviceB;
    float *deviceC;
    int numARows;    // number of rows in the matrix A
    int numAColumns; // number of columns in the matrix A
    int numBRows;    // number of rows in the matrix B
    int numBColumns; // number of columns in the matrix B
    int numCRows;
    int numCColumns;

    args = wbArg_read(argc, argv);

#if LAB_DEBUG
    std::cout << "Running GPU Matrix Multiplicaion ..." << std::endl;
#endif

    wbTime_start(Generic, "Importing data and creating memory on host");
    hostA = (float *) wbImport(wbArg_getInputFile(args, 0), &numARows,
                               &numAColumns);
    hostB = (float *) wbImport(wbArg_getInputFile(args, 1), &numBRows,
                               &numBColumns);
    hostC = (float *) malloc(numARows * numBColumns * sizeof(float));
    wbTime_stop(Generic, "Importing data and creating memory on host");

    numCRows = numARows;
    numCColumns = numBColumns;

    wbLog(TRACE, "The dimensions of A are ", numARows, " x ", numAColumns);
    wbLog(TRACE, "The dimensions of B are ", numBRows, " x ", numBColumns);
    wbLog(TRACE, "The dimensions of C are ", numCRows, " x ", numCColumns);

    wbTime_start(GPU, "Allocating GPU memory.");
    cudaMalloc((void **) &deviceA, numARows * numAColumns * sizeof(float));
    cudaMalloc((void **) &deviceB, numBRows * numBColumns * sizeof(float));
    cudaMalloc((void **) &deviceC, numCRows * numCColumns * sizeof(float));
    wbTime_stop(GPU, "Allocating GPU memory.");

    wbTime_start(GPU, "Copying input memory to the GPU.");
    cudaMemcpy(deviceA, hostA, numARows * numAColumns * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceB, hostB, numBRows * numBColumns * sizeof(float), cudaMemcpyHostToDevice);
    wbTime_stop(GPU, "Copying input memory to the GPU.");

    unsigned int threads = 64;
    unsigned int blocksX = (numCColumns + threads - 1) / threads;
    unsigned int blocksY = (numCRows + threads - 1) / threads;

    dim3 gridSize(threads, threads, 1);
    dim3 blockSize(blocksX, blocksY, 1);

    wbLog(TRACE, "The block dimensions are ", gridSize.x, " x ", gridSize.y);
    wbLog(TRACE, "The grid dimensions are ", blockSize.x, " x ", blockSize.y);

    wbTime_start(Compute, "Performing CUDA computation");
    matrixMultiply<<<gridSize, blockSize>>>(deviceA, deviceB, deviceC, numARows, numAColumns, numBRows, numBColumns);

    cudaDeviceSynchronize();
    wbTime_stop(Compute, "Performing CUDA computation");

    wbTime_start(Copy, "Copying output memory to the CPU");
    cudaMemcpy(hostC, deviceC, numCRows * numCColumns * sizeof(float), cudaMemcpyDeviceToHost);
    wbTime_stop(Copy, "Copying output memory to the CPU");

    wbTime_start(GPU, "Freeing GPU Memory");
    cudaFree(deviceA);
    cudaFree(deviceB);
    cudaFree(deviceC);
    wbTime_stop(GPU, "Freeing GPU Memory");

    wbSolution(args, hostC, numCRows, numCColumns);

    free(hostA);
    free(hostB);
    free(hostC);

#if LAB_DEBUG
    system("pause");
#endif

    return 0;
}