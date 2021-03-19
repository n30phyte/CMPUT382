#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "wb.h"

#define MASK_WIDTH 5
#define MASK_RADIUS MASK_WIDTH / 2
#define TILE_WIDTH 16
#define SHARED_WIDTH (TILE_WIDTH + MASK_WIDTH - 1)

__constant__ float M[MASK_WIDTH][MASK_WIDTH];

__global__ void convolution(const float *__restrict__ I, float *__restrict__ P, const int channels,
                            const int width, const int height) {
    const unsigned int tx = threadIdx.x;
    const unsigned int ty = threadIdx.y;

    const unsigned int imgX = tx + TILE_WIDTH * blockIdx.x;
    const unsigned int imgY = ty + TILE_WIDTH * blockIdx.y;

    const int shared_x = imgX - MASK_RADIUS;
    const int shared_y = imgY - MASK_RADIUS;

    for (unsigned int c = 0; c < channels; c++) {
        __shared__ float I_ds[SHARED_WIDTH][SHARED_WIDTH];

        if ((shared_x >= 0 && shared_x < width)
            && (shared_y >= 0 && shared_y < height)) {
            const unsigned int imgXY = (shared_x + shared_y * width) * channels + c;
            I_ds[ty][tx] = I[imgXY];
        } else {
            I_ds[ty][tx] = 0;
        }
        __syncthreads();

        if (tx < TILE_WIDTH && ty < TILE_WIDTH) {
            float accumulator = 0;
            for (unsigned int y = 0; y <= MASK_WIDTH; y++) {
                for (unsigned int x = 0; x <= MASK_WIDTH; x++) {
                    accumulator += M[y][x] * I_ds[ty + y][tx + x];
                }
            }

            if (imgX < width && imgY < height) {
                P[(imgX + imgY * width) * channels + c] = __saturatef(accumulator);
            }
        }
    }
}

int main(int argc, char *argv[]) {
    wbArg_t arg;
    int maskRows;
    int maskColumns;
    int imageChannels;
    int imageWidth;
    int imageHeight;
    char *inputImageFile;
    char *inputMaskFile;
    wbImage_t inputImage;
    wbImage_t outputImage;
    float *hostInputImageData;
    float *hostOutputImageData;
    float *hostMaskData;
    float *deviceInputImageData;
    float *deviceOutputImageData;

    arg = wbArg_read(argc, argv); /* parse the input arguments */

    inputImageFile = wbArg_getInputFile(arg, 0);
    inputMaskFile = wbArg_getInputFile(arg, 1);

    inputImage = wbImport(inputImageFile);
    hostMaskData = (float *) wbImport(inputMaskFile, &maskRows, &maskColumns);

    assert(maskRows == 5);    /* mask height is fixed to 5 in this mp */
    assert(maskColumns == 5); /* mask width is fixed to 5 in this mp */

    imageWidth = wbImage_getWidth(inputImage);
    imageHeight = wbImage_getHeight(inputImage);
    imageChannels = wbImage_getChannels(inputImage);

    outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);

    hostInputImageData = wbImage_getData(inputImage);
    hostOutputImageData = wbImage_getData(outputImage);

    wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

    wbTime_start(GPU, "Doing GPU memory allocation");
    const size_t imageSize = imageWidth * imageHeight * imageChannels * sizeof(float);
    const size_t maskSize = maskRows * maskColumns * sizeof(float);

    cudaMalloc((void **) &deviceInputImageData, imageSize);
    cudaMalloc((void **) &deviceOutputImageData, imageSize);

    wbTime_stop(GPU, "Doing GPU memory allocation");

    wbTime_start(Copy, "Copying data to the GPU");
    cudaMemcpy(deviceInputImageData, hostInputImageData, imageSize, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(M, hostMaskData, maskSize);
    wbTime_stop(Copy, "Copying data to the GPU");

    const int threads = SHARED_WIDTH;
    const dim3 blockSize(threads, threads, 1);
    const int gridX = (imageWidth + TILE_WIDTH - 1) / TILE_WIDTH;
    const int gridY = (imageHeight + TILE_WIDTH - 1) / TILE_WIDTH;
    const dim3 gridSize(gridX, gridY, 1);

    wbTime_start(Compute, "Doing the computation on the GPU");
    convolution<<<gridSize, blockSize>>>(deviceInputImageData, deviceOutputImageData, imageChannels, imageWidth, imageHeight);
    cudaDeviceSynchronize();
    wbTime_stop(Compute, "Doing the computation on the GPU");

    wbTime_start(Copy, "Copying data from the GPU");
    cudaMemcpy(hostOutputImageData, deviceOutputImageData, imageSize, cudaMemcpyDeviceToHost);

    wbTime_stop(Copy, "Copying data from the GPU");

    wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

    wbSolution(arg, outputImage);

    cudaFree(deviceInputImageData);
    cudaFree(deviceOutputImageData);

    free(hostMaskData);
    wbImage_delete(outputImage);
    wbImage_delete(inputImage);

#if LAB_DEBUG
    system("pause");
#endif

    return 0;
}