#include<stdio.h>
#include<stdlib.h>
#include<cuda_runtime.h>

__global__ void hola(int n, float a, float *x) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    printf(" i %d blockDim.x %d gridDim.x %d\n", i, blockDim.x, gridDim.x);
    if (i < n) x[i] = a*x[i];
}

int main(void) {
    int N = 1 << 4;
    unsigned int threads = 2;
    float *hx, *dx;
    hx = (float*)malloc(N * sizeof(float));

    cudaMalloc(&dx, N * sizeof(float));

    for (int i = 0; i < N; i++)
    {
        hx[i] = (float)(N -i);
    }

    cudaMemcpy(dx, hx, N * sizeof(float), cudaMemcpyHostToDevice);

    //unsigned int blocks = (N + 255) / threads;
    unsigned int blocks = ceil(N/threads);

    float a = 2.0f;

    void *args[] = { &N, &a, &dx };
    cudaError_t res = cudaLaunchKernel((void*)hola, dim3(blocks,1,1), dim3(threads,1,1), args, 0, NULL);
    if (res != cudaSuccess) {
        printf ("error en kernel launch: %s \n", cudaGetErrorString(res));
        return -1;
    }
    cudaDeviceSynchronize();


    cudaMemcpy(hx, dx, N * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; i++) {
	printf(" hx %f\n",hx[i]);
    }

    cudaFree(dx);
    free(hx);

    return 0;
}

