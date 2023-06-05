#include <stdio.h>
#include <iostream>
#include <cuda_runtime.h>
#include <chrono>
#include <cooperative_groups.h>


using namespace std;

#define blocksz 256

__device__ int reduce_sum(int *temp, int val) {
    int tid = threadIdx.x;
    for (int i = blockDim.x / 2; i > 0; i /= 2)
    {
        temp[tid] = val;
	__syncthreads();
        if(tid<i) val += temp[tid + i];
	__syncthreads();
    }
    return val; // note: only thread 0 will return full sum
}

__global__ void reduce(int *sum, int *input, int n)
{
    extern __shared__ int temp[];
    int id = blockIdx.x*blockDim.x + threadIdx.x;
    int block_sum = reduce_sum(temp, input[id]);

    if (threadIdx.x == 0) atomicAdd(sum, block_sum);
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
     	//int n = 1<<24;
     	//int blockSize = 256;
     	int n = 1<<bits;
     	int blockSize = tpb;
     	int nBlocks = (n + blockSize - 1) / blockSize;
	cout<<" nBlocks = "<<nBlocks<<" ceil(n/blockSize) = "<<ceil(n/blockSize)<<endl;
     	int sharedBytes = blockSize * sizeof(int);

     	int *sum, *data;
     	cudaMallocManaged(&sum, sizeof(int));
     	cudaMallocManaged(&data, n * sizeof(int));
     	std::fill_n(data, n, 2); // initialize data
     	cudaMemset(sum, 0, sizeof(int));
        int *oner = (int *)malloc(sizeof(int)); // puntero a resultado host

        auto st_time = std::chrono::high_resolution_clock::now();
	reduce<<<nBlocks, blockSize, sharedBytes>>>(sum, data, n);
        auto e_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> ttime = e_time - st_time;
        std::cout<<" tiempo kernel "<<ttime.count()<<"s"<<std::endl;


	cudaMemcpy(oner, sum, sizeof(int), cudaMemcpyDeviceToHost);
	cout<<" oner "<<*oner<<endl;	

	cudaFree(sum);
	cudaFree(data);

	free(oner);

}
