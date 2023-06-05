#include<stdio.h>
#include<stdlib.h>
#include<cuda_runtime.h>

__global__ void hola(int n, float a, float *x) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = a*x[i];
}

int main(void) {
    int N = 1 << 4;
    unsigned int threads = 2;
    float *hdx;

    cudaMallocManaged(&hdx, N * sizeof(float));

    for (int i = 0; i < N; i++)
    {
        hdx[i] = (float)(N -i);
    }
    unsigned int blocks = ceil(N/threads);

    float a = 2.0f;

    void *args[] = { &N, &a, &hdx };
    cudaError_t res = cudaLaunchKernel((void*)hola, dim3(blocks,1,1), dim3(threads,1,1), args, 0, NULL);
    if (res != cudaSuccess) {
        printf ("error en kernel launch: %s \n", cudaGetErrorString(res));
        return -1;
    }
    cudaDeviceSynchronize();


    for (int i = 0; i < N; i++) {
	printf(" hdx %f\n",hdx[i]);
    }

    cudaFree(hdx);

    return 0;
}

