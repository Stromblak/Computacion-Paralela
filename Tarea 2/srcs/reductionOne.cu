#include <stdio.h>
#include <iostream>
#include <cuda_runtime.h>
#include <chrono>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>


using namespace std;
//using namespace cooperative_groups;
namespace cg = cooperative_groups;

__device__ void reduce_sum(cg::thread_block g, int *temp) {
    int tid = g.thread_rank();
    cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(g);
    temp[tid] = cg::reduce(tile32,temp[tid], cg::plus<int>());
    g.sync();

    int beta = 0;
    if( g.thread_rank() == 0){
	beta = 0;
	for(int i=0; i<blockDim.x; i+=tile32.size()){
		beta += temp[i];
	}	
	temp[0] = beta;
    }
    g.sync();

}

__global__ void reduceG(int *sum, int *input, int *odata, int *Total)
{
    int n = *Total;
    extern __shared__ int temp[];
    cg::grid_group grid = cg::this_grid();
    int id = blockIdx.x*blockDim.x + threadIdx.x;
    cg::thread_block g = cg::this_thread_block();
    temp[g.thread_rank()] = 0;

    for(int i=grid.thread_rank() ; i<n; i+=grid.size()){
	temp[g.thread_rank()] += input[i];
    }
    grid.sync();
    reduce_sum(g,temp);
    if(g.thread_rank() == 0){
	odata[blockIdx.x] = temp[0];
    }
    grid.sync();
    if(grid.thread_rank() == 0){
	for(int b=0; b<gridDim.x; b++){
		*sum += odata[b];
	}
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

     	int *sum, *data, *parcial, *Total;
     	cudaMallocManaged(&sum, sizeof(int));
     	cudaMallocManaged(&Total, sizeof(int));
	*Total = n;
     	cudaMallocManaged(&data, n * sizeof(int));
     	cudaMallocManaged(&parcial, nBlocks * sizeof(int));
     	std::fill_n(data, n, 2); // initialize data
     	cudaMemset(sum, 0, sizeof(int));
	int dev = 0;
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, dev);
	// initialize, then launch
	//cudaLaunchCooperativeKernel((void*)my_kernel, deviceProp.multiProcessorCount, numThreads, args);
	int maxThreads = deviceProp.maxThreadsPerBlock;
	int numBlocksPerSm = 0;
	int numThreads = 0;
	int numSms = deviceProp.multiProcessorCount;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, reduceG, blockSize, blockSize*sizeof(int));

        if (nBlocks > numBlocksPerSm * numSms) {
           nBlocks = numBlocksPerSm * numSms;
        }
	if (blockSize > maxThreads)
		blockSize = maxThreads;
     	int sharedBytes = blockSize * sizeof(int);
	cout<<" nBlocks = "<<nBlocks<<" ceil(n/blockSize) = "<<ceil(n/blockSize)<<endl;
	
        printf("numThreads: %d\n", blockSize);
        printf("numBlocks: %d\n", nBlocks);


	cout<<" deviceProp.multiProcessorCount "<<deviceProp.multiProcessorCount<<endl;
	void *args[] = { &sum, &data, &parcial, &Total };
	dim3 gd = dim3(nBlocks,1,1);
	dim3 bd = dim3(blockSize,1,1);
        auto st_time = std::chrono::high_resolution_clock::now();
        cudaError_t res = cudaLaunchCooperativeKernel((void *)reduceG, gd, bd, args, sharedBytes);
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
