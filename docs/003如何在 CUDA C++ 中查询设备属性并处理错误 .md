# 如何在 CUDA C++ 中查询设备属性并处理错误 

ref:

- [gpu Compute Capability](https://developer.nvidia.com/cuda-gpus)
- [官方Compute Capability详细情况](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capabilities)
- [1050理论带宽](https://www.techpowerup.com/gpu-specs/geforce-gtx-1050-ti.c2885)


## 计算能力

```
nvcc --gpu-architecture=compute_61 --gpu-code=sm_61 add.cu
```

## 错误处理

```c++
cudaError_t err = cudaGetDeviceCount(&nDevices);
  if (err != cudaSuccess) printf("%s\n", cudaGetErrorString(err));

saxpy<<<(N+255)/256, 256>>>(N, 2.0, d_x, d_y);
cudaError_t errSync  = cudaGetLastError();
cudaError_t errAsync = cudaDeviceSynchronize();
if (errSync != cudaSuccess) 
  printf("Sync kernel error: %s\n", cudaGetErrorString(errSync));
if (errAsync != cudaSuccess)
  printf("Async kernel error: %s\n", cudaGetErrorString(errAsync));
```

函数cudaPeekAtLastError()返回该变量的值

函数 cudaGetLastError()返回此变量的值并将其重置为 cudaSuccess

 cudaDeviceSynchronize()，它会阻塞主机线程，直到所有先前发出的命令完成为止。 任何异步错误都由 cudaDeviceSynchronize()