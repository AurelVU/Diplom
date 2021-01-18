
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include "mainSolver.h"
/*
ToDo:
Основная проблема сейчас в том, что у нас алгоритм последовательо проходится от i=0 до максимального, аналогично для j. И рассчет следующего выполняется на основе предыдущего. Надо что-то придумать с этим
*/
cudaError_t solver(float dx, float Tb, float Tb0, int imax, int jmax, float dh, float** T, int k, int FaceArea, float delt);


__global__ void solverKernel(float dx, float Tb, float Tb0, int imax, int jmax, float dh, float* T, float* T2, int k, int FaceArea, float delt)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= 0 && i < imax && j >= 0 && j < jmax)
    {
        float Tc, Te, Tw, Tn, Ts;
        float FluxC, FluxE, FluxW, FluxN, FluxS;

        Tc = T[i * jmax + j];
        dx = dh;
        
        if (i == imax - 1) { Te = Tb0; dx = dx / 2; }
        else
            Te = T[i * jmax + j];
        FluxE = (-k * FaceArea) / dx;

        if (i == 0) { Tw = Tb0; dx = dx / 2; }
        else
            Tw = T[i * jmax + j];
        FluxW = (-k * FaceArea) / dx;

        if (j == jmax - 1) { Tn = Tb0; dx = dx / 2; }
        else
            Tn = T[i * jmax + j + 1];
        FluxN = (-k * FaceArea) / dx;

        if (j == 0) { Ts = Tb; dx = dx / 2; }
        else
            Ts = T[i * jmax + j - 1];
        FluxS = (-k * FaceArea) / dx;

        FluxC = FluxE + FluxW + FluxN + FluxS;

        T2[i * jmax + j] = Tc + delt * (FluxC * Tc - (FluxE * Te + FluxW * Tw + FluxN * Tn + FluxS * Ts));
    }
}

int main()
{
    mainSolver* m = new mainSolver();
    m->create("dfsfdsdf");
    m->RunPhysic();
    

    float dx = 0;
    float Tb = 240;
    float Tb0 = 0;
    float delt = 0.2;
    int FaceArea = 1;
    int k = 1;
    int dh = 1;
    int imax = 40;
    int jmax = 40;
    float** T = new float* [imax];
    for (int i = 0; i < imax; i++)
    {
        T[i] = new float[jmax];
        for (int j = 0; j < jmax; j++)
        {
            //if (j == 0)
            T[i][j] = 0;
        }
    }

    for (int i = 0; i < 49; i++)
        m->RunPhysic();


    for (int i = 0; i < imax; i++)
    {
        for (int j = 0; j < jmax; j++)
            printf("%.2f ", m->T[i][j]);
        printf("\n");
    }
    printf("###############################################################################\n");
    

    cudaError_t cudaStatus = solver(dx, Tb, Tb0, imax, jmax, dh, T, k, FaceArea, delt);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}



cudaError_t solver(float dx, float Tb, float Tb0, int imax, int jmax, float dh, float** T, int k, int FaceArea, float delt)
{
    float* dev_T = 0;
    float* dev_T2 = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_T, imax * jmax * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_T2, imax * jmax * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    dim3 gridSize = dim3((imax + 31) / 32, (jmax + 31) / 32, 1);    //Размер используемого грида
    dim3 blockSize = dim3(32, 32, 1); //Размер используемого блока


    // Launch a kernel on the GPU with one thread for each element.
    for (int i = 0; i < 50; i++)
    {
        solverKernel << <gridSize, blockSize >> > (dx, Tb, Tb0, imax, jmax, dh, dev_T, dev_T2, k, FaceArea, delt);
        float* a = dev_T;
        dev_T = dev_T2;
        dev_T2 = a;
    }
    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    for (int i = 0; i < imax; i++)
    {
        cudaStatus = cudaMemcpy(T[i], dev_T + i * jmax, jmax * sizeof(float), cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed!");
            goto Error;
        }
    }
    for (int i = 0; i < imax; i++)
    {
        for (int j = 0; j < jmax; j++)
            printf("%.2f ", T[i][j]);
        printf("\n");
    }
    
Error:
    cudaFree(dev_T);
    cudaFree(dev_T2);

    return cudaStatus;
}