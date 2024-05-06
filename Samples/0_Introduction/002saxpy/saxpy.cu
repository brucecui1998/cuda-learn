
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <chrono>
#include <iostream>

// 使用 std::chrono 来记录时间
float myCPUTimer() {
    static auto last_time = std::chrono::high_resolution_clock::now();
    auto current_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> duration = current_time - last_time;
    last_time = current_time;
    return duration.count();
}

__global__
void saxpy(int n, float a, float* x, float* y)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a * x[i] + y[i];
}

int main(void)
{
    int N = 1 << 20;
    float* x, * y, * d_x, * d_y;
    x = (float*)malloc(N * sizeof(float));
    y = (float*)malloc(N * sizeof(float));

    cudaMalloc(&d_x, N * sizeof(float));
    cudaMalloc(&d_y, N * sizeof(float));

    for (int i = 0; i < N; i++) {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaMemcpy(d_x, x, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, N * sizeof(float), cudaMemcpyHostToDevice);

     //Perform SAXPY on 1M elements
    /*float t1 = myCPUTimer();
    saxpy << <(N + 255) / 256, 256 >> > (N, 2.0f, d_x, d_y);
    cudaDeviceSynchronize();
    float t2 = myCPUTimer();
    std::cout << "Kernel execution time: " << t2 - t1 << " ms" << std::endl;*/

    cudaEventRecord(start);
    // Perform SAXPY on 1M elements
    saxpy << <(N + 511) / 512, 512 >> > (N, 2.0f, d_x, d_y);
    cudaEventRecord(stop);

    cudaMemcpy(y, d_y, N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Kernel execution time: " << milliseconds << " ms" << std::endl;
    printf("Effective Bandwidth (GB/s): %f\n", N * sizeof(float) * 3 / milliseconds / 1e6);
    

    float maxError = 0.0f;
    for (int i = 0; i < N; i++)
        maxError = fmax(maxError, abs(y[i] - 4.0f));
    printf("Max error: %f\n", maxError);

    cudaFree(d_x);
    cudaFree(d_y);
    free(x);
    free(y);
}