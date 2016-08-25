// Add lib we need
#pragma comment(lib,"lib/cximage.lib")
#pragma comment(lib,"lib/jasper.lib")
#pragma comment(lib,"lib/jbig.lib")
#pragma comment(lib,"lib/Jpeg.lib")
#pragma comment(lib,"lib/libdcr.lib")
#pragma comment(lib,"lib/libpsd.lib")
#pragma comment(lib,"lib/mng.lib")
#pragma comment(lib,"lib/png.lib")
#pragma comment(lib,"lib/Tiff.lib")
#pragma comment(lib,"lib/zlib.lib")

//we have to put this before ximage.h

#ifndef UNICODE
#define UNICODE
#endif

#ifndef _UNICODE
#define _UNICODE
#endif


#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"
#include "ximage.h"
#include <math.h>
#include <stdio.h>
#include <iostream>
#include <string>



__device__ void maxReduce(float inputArray[]) {
    for(size_t s = 512; s > 0; s >>= 1) {
        int idx = threadIdx.y * blockDim.x + threadIdx.x;
        if(idx < s) {
            inputArray[idx] = inputArray[idx] > inputArray[s + idx] ? inputArray[idx] : inputArray[idx + s];
        }
        __syncthreads();
    }
}

__device__ void sumReduce(float inputArray[]) {
    for (size_t s = 512; s > 0 ; s >>= 1) {
        int idx = threadIdx.y * blockDim.x + threadIdx.x;
        if(idx < s) {
            inputArray[idx] += inputArray[s + idx];
        }
        __syncthreads();
    }
}

__device__ void paddingEdgeAndCorner(float** im) {
    /*   int ix = threadIdx.x;
       int iy = threadIdx.y;
       if(ix >= 5 && iy < 5)
           im[iy][ix] = im[5][ix];
       if(ix >= 5 && iy > 36)
           im[iy][ix] = im[36][ix];
       __syncthreads();
       if(ix < 5)
           im[iy][ix] = im[iy][5];
       if(ix > 36)
           im[iy][ix] = im[iy][36];
       __syncthreads();*/
}
//二维数组的参数传递是个大问题
__device__ void gpuImfilter(float im[][42], float om[][32], float filter[][11], size_t fx, size_t fy) {
    om[threadIdx.y][threadIdx.x] = 0;
    __syncthreads();
    for(size_t i = 0; i < fy; i++) {
        for(size_t j = 0; j < fx; j++) {
            om[threadIdx.y][threadIdx.x] += filter[i][j] * im[threadIdx.y + i][threadIdx.x + j];
        }
    }
    __syncthreads();
}



//这个内核的运行时间太长，还是得修改在注册表修改下TDR
//详情请看网址 http://stackoverflow.com/questions/497685/cuda-apps-time-out-fail-after-several-seconds-how-to-work-around-this
__global__ void sasanMethod(const float		*inputImage,
                            float			*outputImage,
                            size_t			width,
                            size_t			height,
                            size_t			pitch,
                            const float		*filter,
                            size_t			filterPitch,
                            size_t			filterSize,
                            size_t          phase) {

    __shared__  float paddingImage[42][42];
    __shared__  float reduceArray[1024];
    __shared__  float blockImage[32][32];
    __shared__  float tmpImage[32][32];
    __shared__  float gausFilter[11][11];

    //初始化
    paddingImage[threadIdx.y][threadIdx.x] = 0;
    if (threadIdx.y + blockDim.y < 42 && threadIdx.x + blockDim.x < 42) {
        paddingImage[threadIdx.y + blockDim.y][threadIdx.x + blockDim.x] = 0;
    }
    if (threadIdx.y + blockDim.y < 42) {
        paddingImage[threadIdx.y + blockDim.y][threadIdx.x] = 0;
    }
    if (threadIdx.x + blockDim.x < 42) {
        paddingImage[threadIdx.y][threadIdx.x + blockDim.x] = 0;
    }


    //copy filter from global memory to shared memory
    if(threadIdx.y < 11 && threadIdx.x < 11) {
        gausFilter[threadIdx.y][threadIdx.x] = *((float *)((char *)filter + threadIdx.y * filterPitch) + threadIdx.x) ;
    }

    size_t bx, by;
    switch(phase) {
    case 0:
        bx = blockIdx.x * 2;
        by = blockIdx.y * 2;
        break;
    case 1:
        bx = blockIdx.x * 2 + 1;
        by = blockIdx.y * 2;
        break;
    case 2:
        bx = blockIdx.x * 2;
        by = blockIdx.y * 2 + 1;
        break;
    case 3:
        bx = blockIdx.x * 2 + 1;
        by = blockIdx.y * 2 + 1;
        break;
    }
    //将相应输入的二维图像的像素点拷贝入共享内存

    blockImage[threadIdx.y][threadIdx.x] = *((float *)((char*)inputImage + (by * blockDim.y + threadIdx.y) * pitch) + bx * blockDim.x + threadIdx.x);
    __syncthreads();

    size_t i = threadIdx.y * blockDim.x + threadIdx.x;
    reduceArray[i] = blockImage[threadIdx.y][threadIdx.x];
    __syncthreads();
    sumReduce(reduceArray);
    size_t ndots = reduceArray[0];
    //这里必须要加一个同步才行
     __syncthreads();

    for(size_t m = 0; m < ndots; m++) {

        size_t i = threadIdx.y * blockDim.x + threadIdx.x;
        reduceArray[i] = blockImage[threadIdx.y][threadIdx.x];

        __syncthreads();
        maxReduce(reduceArray);
        if (blockImage[threadIdx.y][threadIdx.x] == reduceArray[0]) {
            paddingImage[threadIdx.y + 5][threadIdx.x + 5] = 1.0f;
			//如果有同时存在相同大小的点呢？
        }
        __syncthreads();
        //paddingEdgeAndCorner((float**)paddingImage);
        gpuImfilter(paddingImage, tmpImage, gausFilter, 11, 11);

        blockImage[threadIdx.y][threadIdx.x] -= tmpImage[threadIdx.y][threadIdx.x];//+ paddingImage[threadIdx.y+5][threadIdx.x+5];

        __syncthreads();
    }

    *((float *)((char*)outputImage + (by * blockDim.y + threadIdx.y) * pitch) + bx * 32 + threadIdx.x) = paddingImage[threadIdx.y + 5][threadIdx.x + 5];

    //__syncthreads();
}


//为了能够消除边界的缺陷，整个并行算法要
//要分为四步来进行

cudaError_t applySasanMethod(const float   *inputImage ,
                             float         *outputImage ,
                             size_t        width ,
                             size_t        height ,
                             size_t        pitch ,
                             const float   *filter ,
                             size_t        filterPitch ,
                             size_t        filterSize ,
                             dim3          dimGrid ,
                             dim3          dimBlock) {
    cudaError_t cudaStatus;
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }
    sasanMethod <<< dimGrid, dimBlock >>>(inputImage, outputImage, width, height, pitch, filter, filterPitch, filterSize, 0);
    //cudaDeviceSynchronize等待kernel执行完成再进行再一个kernel的调用
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "phase0 cudaDeviceSynchronize returned error code %d after launching Kernel!\n", cudaStatus);
        std::string a = cudaGetErrorString(cudaGetLastError());
        std::cout << "problem: " << a << std::endl;
        getchar();
        goto Error;
    }
    sasanMethod <<< dimGrid, dimBlock >>>(inputImage, outputImage, width, height, pitch, filter, filterPitch, filterSize, 1);
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "phase1 cudaDeviceSynchronize returned error code %d after launching Kernel!\n", cudaStatus);
        std::string a = cudaGetErrorString(cudaGetLastError());
        std::cout << "problem: " << a << std::endl;
        getchar();
        goto Error;
    }
    sasanMethod <<< dimGrid, dimBlock >>>(inputImage, outputImage, width, height, pitch, filter, filterPitch, filterSize, 2);
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "phase2 cudaDeviceSynchronize returned error code %d after launching Kernel!\n", cudaStatus);
        std::string a = cudaGetErrorString(cudaGetLastError());
        std::cout << "problem: " << a << std::endl;
        getchar();
        goto Error;
    }
    sasanMethod <<< dimGrid, dimBlock >>>(inputImage, outputImage, width, height, pitch, filter, filterPitch, filterSize, 3);
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "phase3 cudaDeviceSynchronize returned error code %d after launching Kernel!\n", cudaStatus);
        std::string a = cudaGetErrorString(cudaGetLastError());
        std::cout << "problem: " << a << std::endl;
        getchar();
        goto Error;
    }
Error:
    return cudaStatus;
}

template<typename T>
void new2d(size_t cx, size_t cy, T*** rev, T initVal) {
    T** f = new T*[cy];
    for (size_t i = 0; i < cy; i++) {
        f[i] = new T[cx];
        //为了提高速度不得不这么做，应该有更好的办法
        memset(f[i], initVal, sizeof(T)*cx);
    }
    *rev = f;
}

template<typename T>
void del2d(size_t cx, size_t cy, T** array) {
    for (size_t i = 0; i < cy; i++) {
        delete[] array[i];
    }
    delete[] array;
}

float** gaussianFilterCreator(float sigma, int size) {
    float sum = 0;
    float **f;
    int mid = (size + 1) / 2;
    new2d(size, size, &f, 0.0f);
    for (int y = 0; y < size; y++) {
        for (int x = 0; x < size; x++) {
            sum += f[y][x] = exp(-(pow((x + 1 - mid), 2) + pow((y + 1 - mid), 2)) / (2 * pow(sigma, 2)));
        }
    }
    for (int y = 0; y < size; y++) {
        for (int x = 0; x < size; x++) {
            f[y][x] /= sum;
        }
    }
    return f;
}

float** padarray(size_t cx, size_t cy, float** im, size_t px, size_t py) {
    float** om;
    //0.0f 是为了让函数模板类型推断正确通过
    new2d((cx + 2 * px), (cy + 2 * py), &om, 0.0f);
    for (size_t i = 0; i < cy; i++) {
        for (size_t j = 0; j < cx; j++) {
            om[i + py][j + px] = im[i][j];
        }
    }
    //padding edge
    for (size_t i = 1; i < cy - 1; i++) {
        for (size_t j = 0; j < px; j++) {
            om[py + i][j] = om[py + i][px];
            om[py + i][cx + px + j] = om[py + i][cx + px - 1];
        }
    }
    for (size_t i = 1; i < cx - 1; i++) {
        for (size_t j = 0; j < py; j++) {
            om[j][px + i] = om[py][px + i];
            om[cy + py + j][px + i] = om[cy + py - 1][py + i];
        }
    }
    //padding corner
    for (size_t i = 0; i <= py; i++) {
        for (size_t j = 0; j <= px; j++) {
            om[i][j] = om[py][px];
            om[i][j + px + cx - 1] = om[py][cx + px - 1];
            om[i + py + cy - 1][j] = om[py + cy - 1][px];
            om[i + py + cy - 1][j + px + cx - 1] = om[py + cy - 1][px + cx - 1];
        }
    }
    return om;

}

float** imfilter(float** im, size_t ix, size_t iy, float** filter, size_t fx, size_t fy) {
    float **pim = padarray(ix, iy, im, fx, fy);
    float **om;
    new2d(ix, iy, &om, 0.0f);
    size_t mx = (fx - 1) / 2;
    size_t my = (fy - 1) / 2;
    for (size_t i = 0; i < iy; i++) {
        for (size_t j = 0; j < ix; j++) {
            om[i][j] = 0;
            for (size_t m = 0; m < fy; m++) {
                for (size_t n = 0; n < fx; n++) {
                    om[i][j] += pim[m + i + my + 1][n + j + mx + 1] * filter[m][n];
                }
            }
        }
    }
    del2d(ix + fx, iy + fy, pim);
    return om;
}

int main() {
    CxImage image;
    size_t filterSize = 11;
    cudaError_t cudaStatus;
    float **gausFilter = gaussianFilterCreator(1.3f, filterSize);
    if (image.Load(_T("lena.jpg"), CXIMAGE_SUPPORT_JPG)) { // _T() is necessary if we need UNICODE support
        image.GrayScale();                                 // To simplify the problem ,convert the image to 8 bits gray scale
        size_t width = image.GetWidth();
        size_t height = image.GetHeight();
        float** image_array;
        new2d(width, height, &image_array, 0.0f);
        for (size_t y = 0; y < height; y++) {
            uint8_t *iSrc = image.GetBits(y);
            for (size_t x = 0; x < width; x++) {
                image_array[y][x] = iSrc[x] / 255.0;
            }
        }
        //对灰度图进行高斯滤波
        float **h_image = imfilter(image_array, width, height, gausFilter , filterSize, filterSize);
        //分配二维数组，注意pitch的单位是字节数
        float* gpuInputImage;
        size_t pitch;
        cudaStatus = cudaMallocPitch(&gpuInputImage, &pitch, width * sizeof(float), height);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMallocPitch return error code %d !\n", cudaStatus);
            std::string a = cudaGetErrorString(cudaGetLastError());
            std::cout << "problem: " << a << std::endl;
            getchar();
        }
        for (size_t y = 0; y < height; y++) {
            cudaMemcpy((float*)((char*)gpuInputImage + y * pitch), h_image[y], width * sizeof(float), cudaMemcpyHostToDevice);
            if (cudaStatus != cudaSuccess) {
                fprintf(stderr, "cudaMemcpy return error code %d !\n", cudaStatus);
                std::string a = cudaGetErrorString(cudaGetLastError());
                std::cout << "problem: " << a << std::endl;
                getchar();
            }
        }
        //输出图像
        float* gpuOutputImage;
        cudaMallocPitch(&gpuOutputImage, &pitch, width * sizeof(float), height);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMallocPitch return error code %d !\n", cudaStatus);
            std::string a = cudaGetErrorString(cudaGetLastError());
            std::cout << "problem: " << a << std::endl;
            getchar();
        }


        float* gpuFilter;
        size_t filterPitch;
        /////////////////careful!
        //gausFilter[5][5] = 1.0f;
        /////////////////
        cudaMallocPitch(&gpuFilter, &filterPitch, filterSize * sizeof(float), filterSize);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMallocPitch return error code %d !\n", cudaStatus);
            std::string a = cudaGetErrorString(cudaGetLastError());
            std::cout << "problem: " << a << std::endl;
            getchar();
        }
        for (size_t y = 0; y < filterSize; y++) {
            cudaMemcpy((float*)((char*)gpuFilter + y * filterPitch), gausFilter[y], filterSize * sizeof(float), cudaMemcpyHostToDevice);
            if (cudaStatus != cudaSuccess) {
                fprintf(stderr, "cudaMemcpy return error code %d !\n", cudaStatus);
                std::string a = cudaGetErrorString(cudaGetLastError());
                std::cout << "problem: " << a << std::endl;
                getchar();
            }

        }

        //GT940M MaxThreadsPerBlock=1024
        //////////////////////////
        size_t blockSize = 32;
        //////////////////////////
        dim3 dimBlock(blockSize, blockSize);
        dim3 dimGrid(height / blockSize / 2, width / blockSize / 2);

        cudaStatus = applySasanMethod(gpuInputImage, gpuOutputImage, width, height, pitch, gpuFilter, filterPitch, filterSize, dimGrid, dimBlock);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "applySasanMethod failed!");
            return 1;
        }
        for (size_t y = 0; y < height; y++) {
            cudaMemcpy(h_image[y], (float*)((char*)gpuOutputImage + y * pitch), width * sizeof(float), cudaMemcpyDeviceToHost);
            if (cudaStatus != cudaSuccess) {
                fprintf(stderr, "cudaMemcpy return error code %d !\n", cudaStatus);
                std::string a = cudaGetErrorString(cudaGetLastError());
                std::cout << "problem: " << a << std::endl;
                getchar();
            }
        }
        //生成加网后的图像

        CxImage outputImg;
        uint8_t **imageByte;
        new2d(width, height, &imageByte, uint8_t(0));
        for (size_t i = 0; i < height; i++) {
            for (size_t j = 0; j < width; j++) {
                if (h_image[i][j] == 1.0f)
                    imageByte[i][j] = 255;

            }
        }
        outputImg.CreateFromMatrix(imageByte, width, height, 8, width, NULL);

        outputImg.Save(_T("ht_lena.bmp"), CXIMAGE_SUPPORT_BMP);
        //free host memory
        del2d(width, height, h_image);
        del2d(width, height, image_array);
        del2d(filterSize, filterSize, gausFilter);
        del2d(width, height, imageByte);
        //NOTE: we need to free gpu memory
        cudaFree(gpuInputImage);
        cudaFree(gpuOutputImage);
        cudaFree(gpuFilter);
        std::cout << "done\n";
        // cudaDeviceReset must be called before exiting in order for profiling and
        // tracing tools such as Nsight and Visual Profiler to show complete traces.
        cudaStatus = cudaDeviceReset();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaDeviceReset failed!");
            return 1;
        }
    }
    getchar();
    return 0;
}




