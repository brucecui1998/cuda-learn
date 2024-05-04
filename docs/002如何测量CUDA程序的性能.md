# 002如何测量CUDA程序的性能

ref：https://developer.nvidia.com/blog/how-implement-performance-metrics-cuda-cc/


## 主机-设备同步

cudaMemcpy()是 同步 （或 阻塞 ）传输，在同步传输完成之前，后续的 CUDA 调用无法开始

gpu kernel内核启动是异步的。 一旦内核启动，控制权立即返回到 CPU，而不等待内核完成


## 测量计时

### 使用 CPU 定时器对内核执行进行计时 

```c++ 
// 使用 std::chrono 来记录时间
float myCPUTimer() {
    static auto last_time = std::chrono::high_resolution_clock::now();
    auto current_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> duration = current_time - last_time;
    last_time = current_time;
    return duration.count();
}
        ....
        float t1 = myCPUTimer();
        saxpy << <(N + 255) / 256, 256 >> > (N, 2.0f, d_x, d_y);
        cudaDeviceSynchronize();
        float t2 = myCPUTimer();
        std::cout << "Kernel execution time: " << t2 - t1 << " ms" << std::endl;
        ....
```

### 使用 CUDA 事件计时

```c++
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

cudaMemcpy(d_x, x, N*sizeof(float), cudaMemcpyHostToDevice);
cudaMemcpy(d_y, y, N*sizeof(float), cudaMemcpyHostToDevice);

cudaEventRecord(start);
saxpy<<<(N+255)/256, 256>>>(N, 2.0f, d_x, d_y);
cudaEventRecord(stop);

cudaMemcpy(y, d_y, N*sizeof(float), cudaMemcpyDeviceToHost);

cudaEventSynchronize(stop);
float milliseconds = 0;
cudaEventElapsedTime(&milliseconds, start, stop);
```

TODO：cudaMemcpy本身是阻塞传输，为什么还需要cudaEventSynchronize来阻塞呢？
TODO: cudaEventRecord与nvprof相比，好像是nvprof更加精确。

## 内存带宽

这种区分主要源于两种不同的计数系统：

    十进制前缀（如 KB, MB, GB）：这是基于十进制，每个步骤乘以1000（103103）。这种表示方式在大多数非技术领域以及硬盘驱动器、USB存储设备和其他一些存储产品中更为常见。

    二进制前缀（如 KiB, MiB, GiB）：这是基于二进制，每个步骤乘以1024（210210）。这种表示方法在操作系统和编程语言中用于精确表示内存和存储的大小，因为计算机系统基于二进制。

由于历史上的混用和标准化的需求，国际单位系统（SI）和国际电工委员会（IEC）推出了这两种不同的定义，以帮助消除混淆并更清晰地表达数据大小。在通信和某些类型的数据存储中使用 109109 字节定义，可以确保标准的一致性和跨行业的通信一致性。

```c++
__global__
void saxpy(int n, float a, float* x, float* y)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a * x[i] + y[i];
}
```
需要读一次x[i]，读一次y[i]，写一次y[i]，总共是3次读写

计算公式：BWEffective = (RB + WB) / (t * 10^9) 单位是GB/s

```c++
printf("Effective Bandwidth (GB/s): %f\n", N * sizeof(float) * 3 / milliseconds / 1e6);
```

## 测量计算吞吐量

更常见的是使用分析工具来了解计算吞吐量是否是瓶颈