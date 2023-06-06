#include <stdio.h>
#include <iostream>
#include <vector>
#include <math.h> 
#include <cooperative_groups.h>
#include <cuda_runtime.h>

using namespace std;
using namespace cooperative_groups;


// Kernel para encontrar el maximo
__device__ int reduce_max(thread_group g, int *temp, int val) {
    int tid = g.thread_rank();

    for (int i = blockDim.x / 2; i > 0; i /= 2){
        temp[tid] = val;
		g.sync();

        if(tid<i) val += temp[tid + i];
		g.sync();
    }

    return val; // note: only thread 0 will return full sum
}


// Kernel inicio, cooperative groups
__global__ void reduce(int *sum, int *input, int n){
    extern __shared__ int temp[];
    int id = blockIdx.x*blockDim.x + threadIdx.x;

    thread_group g = this_thread_block();

    int block_sum = reduce_sum(g,temp, input[id]);

    if (threadIdx.x == 0) atomicAdd(sum, block_sum);
}


int main(int argc, char *argv[]) {
	// arreglos, tamaño
	int n = 10, k = 20;
	for(int i=0; i<argc; i++){
		if( !strcmp(argv[i], "-n" ) ) n = atoi(argv[i+1]);
		if( !strcmp(argv[i], "-k" ) ) k = atoi(argv[i+1]);
	}

	// memoria
	float *arreglosDst[n], *arreglosSrc[n];
	for(int i=0; i<n; i++){
		cudaMallocHost(&arreglosDst[i], n * sizeof(float));
		cudaMalloc(&arreglosSrc[i], n * sizeof(float));
	}

	// creacion arreglos
	for(int i=0; i<n; i++) for(int j=0; j<k; j++) arreglosDst[i][j] = j;

	
	/* ejemplo cudaMallocManaged
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
	*/














	
	// cudaMallocManaged

	for (int i = 0; i < N; i++){
		hx[i] = (float)(N-i);
	}

	int nStreams = 2;
	int streamSize = N/nStreams;
	int streamSizeBytes = streamSize*sizeof(int);
	int gdstream = streamSize/threads;
	cudaStream_t stream[nStreams];

	for (int i = 0; i < nStreams; i ++) cudaStreamCreate(&stream[i]);

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

