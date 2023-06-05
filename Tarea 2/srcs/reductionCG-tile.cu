#include <stdio.h>
#include <iostream>
#include <cuda_runtime.h>
#include <chrono>
#include <cooperative_groups.h>


using namespace std;
using namespace cooperative_groups;

__device__ int reduce_sum(thread_group g, int *temp, int val) {
    int tid = g.thread_rank();
    for (int i = blockDim.x / 2; i > 0; i /= 2)
    {
        temp[tid] = val;
	g.sync();
        if(tid<i) val += temp[tid + i];
	g.sync();
    }
    return val; // note: only thread 0 will return full sum
}

__global__ void reduce(int *sum, int *input, int *parcial)
{
    extern __shared__ int temp[];
    grid_group grid = this_grid();
    int id = blockIdx.x*blockDim.x + threadIdx.x;
    thread_group g = this_thread_block();

    int block_sum = reduce_sum(g,temp, input[id]);

    if (threadIdx.x == 0) parcial[blockIdx.x] = block_sum;
    grid.sync();
    int sum_total = 0;
    thread_group tile32 = tiled_partition(g,32);
    if(blockIdx.x == 0 && threadIdx.x<32){
	sum_total = reduce_sum(tile32, temp, parcial[id]);
    }
    if(blockIdx.x == 0 && threadIdx.x == 0){
	*sum = sum_total;
    }

}

void print(int *in, int N){
        for(int i=0; i<N; i++)
                printf("%d ", in[i]);
        printf("\n");
}


int main(int argc, char *argv[]){
	
	if(argc != 3){
		cout<<" USO "<<argv[0]<<" Nbits K (hebras por bloque) \n";
		return 1;
	}
	int midev;
        cudaGetDevice(&midev);
	int bits = atoi(argv[1]);
	int tpb = atoi(argv[2]);
     	int n = 1<<bits;
     	int blockSize = tpb;
     	int nBlocks = (n + blockSize - 1) / blockSize;
	cout<<" nBlocks = "<<nBlocks<<" ceil(n/blockSize) = "<<ceil(n/blockSize)<<endl;
     	int sharedBytes = blockSize * sizeof(int);

     	int *sum, *data, *parcial;
     	cudaMallocManaged(&sum, sizeof(int));
     	cudaMallocManaged(&data, n * sizeof(int));
     	cudaMallocManaged(&parcial, nBlocks * sizeof(int));
     	std::fill_n(data, n, 2); // initialize data
     	cudaMemset(sum, 0, sizeof(int));
	int dev = 0;
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, dev);
	// initialize, then launch
	//cudaLaunchCooperativeKernel((void*)my_kernel, deviceProp.multiProcessorCount, numThreads, args);
	cout<<" deviceProp.multiProcessorCount "<<deviceProp.multiProcessorCount<<endl;
	void *args[] = { &sum, &data, &parcial };
	dim3 gd = dim3(nBlocks,1,1);
	dim3 bd = dim3(blockSize,1,1);
        auto st_time = std::chrono::high_resolution_clock::now();
        cudaError_t res = cudaLaunchCooperativeKernel((void *)reduce, gd, bd, args, sharedBytes);
	if (res != cudaSuccess) {
        	printf ("error en kernel launch: %s \n", cudaGetErrorString(res));
        	return -1;
    	}
	//reduce<<<nBlocks, blockSize, sharedBytes>>>(sum, data,parcial);
	cudaDeviceSynchronize();
        auto e_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> ttime = e_time - st_time;
        std::cout<<" tiempo kernel "<<ttime.count()<<"s"<<std::endl;
	cout<<" sum "<<*sum<<endl;	
	cudaFree(sum);
	cudaFree(data);
	cudaFree(parcial);


}
