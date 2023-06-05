#include<stdio.h>
#include<stdlib.h>
#include<cuda_runtime.h>

__global__ void doble(float *x, int offset) {
    int i = offset + blockIdx.x * blockDim.x + threadIdx.x;
    x[i] = 2*x[i];
}

int main(void) {
    int N = 1 << 4;
    unsigned int threads = 2;
    float *hx, *dx;

    cudaMallocHost(&hx, N * sizeof(float));
    cudaMalloc(&dx, N * sizeof(float));

    for (int i = 0; i < N; i++)
    {
        hx[i] = (float)(N -i);
    }
    int nStreams = 2;
    int streamSize = N/nStreams;
    int streamSizeBytes = streamSize*sizeof(int);
    int gdstream = streamSize/threads;
    cudaStream_t stream[nStreams];
    for (int i = 0; i < nStreams; i ++) {
        cudaStreamCreate(&stream[i]);
    }
    printf(" N %d blocks %d streamSize %d gdstream %d\n", N, threads, streamSize, gdstream);
    for (int i = 0; i < nStreams; i ++) {
        int offset = i * streamSize;
        cudaMemcpyAsync(&dx[offset], &hx[offset], streamSizeBytes, cudaMemcpyHostToDevice, stream[i]);
        doble<<<gdstream, threads, 0, stream[i]>>>(dx, offset);
        cudaMemcpyAsync(&hx[offset], &dx[offset], streamSizeBytes, cudaMemcpyDeviceToHost, stream[i]);
    }
    cudaDeviceSynchronize();

    for (int i = 0; i < N; i++) {
	printf(" hx %f\n",hx[i]);
    }

    cudaFree(dx);
    cudaFreeHost(hx);
    for (int i = 0; i < nStreams; i ++) {
        cudaStreamDestroy(stream[i]);
    }

    return 0;
}

