#include <stdio.h>
#include <iostream>
#include <cuda_runtime.h>
#include <chrono>

using namespace std;

__global__ void reduce_a(int *gdata, int *out, int N){
   __shared__ int sdata[32];
   int tid = threadIdx.x;
   sdata[tid] = 0;
   size_t idx = threadIdx.x+blockDim.x*blockIdx.x;
   while (idx < N) { // grid stride loop to load data
      sdata[tid] += gdata[idx];
      idx += gridDim.x*blockDim.x;
   }
   for (unsigned int s=blockDim.x/2; s>0; s>>=1) {
      __syncthreads();
      if (tid < s) // parallel sweep reduction
         sdata[tid] += sdata[tid + s];
   }
   if (tid == 0) atomicAdd(out, sdata[0]);
}

__global__ void red1(int *gdata, int *out, int N){
        //__shared__ int sdata[BLOCK_SIZE];
	extern __shared__ int sdata[];
	int tid = threadIdx.x;
	sdata[tid] = 0;
	size_t idx = threadIdx.x+blockDim.x*blockIdx.x;
	while (idx < N) { // grid stride loop to load data
		sdata[tid] += gdata[idx];
		idx += gridDim.x*blockDim.x;
	}
	for (unsigned int s=blockDim.x/2; s>0; s>>=1) {
		__syncthreads();
		if (tid < s) // parallel sweep reduction
			sdata[tid] += sdata[tid + s];
	}
	if (tid == 0) out[blockIdx.x] = sdata[0];
}

__global__ void red(int *in, int *out) {
	int id = blockDim.x*blockIdx.x + threadIdx.x;
	int thid = threadIdx.x;
	int bdim = blockDim.x;
	int bid = blockIdx.x;
	//printf(" bdim %d bid %d tid %d unicotid %d blockIdx.y %d\n", bdim, bid, thid, id, blockIdx.y);

	extern __shared__ int sharray[];
	sharray[thid] = in[id]; 

	// todas las hebras en su bloque escriben en mem compartida
	// luego hay que esperar que todas terminn
	__syncthreads();

	//for(int i=bdim/2; i > 0; i >>= 1){
	for(int i=bdim/2; i > 0; i/=2){
		//printf(" i %d thid %d bid %d\n", i, thid, bid);
		if(thid < i){
			//printf(" thid %d bid %d process in index1 %d e index2 %d\n", thid, bid, thid, thid+i);
			sharray[thid] += sharray[thid + i];
		}
		__syncthreads();
	}

	if(thid == 0){
		out[bid] = sharray[0];
	}

}

void initA(int *in, int N){
        for(int i=0; i<N; i++)
                in[i] = i%10;
}

void print(int *in, int N){
        printf(" print Array\n");
        for(int i=0; i<N; i++)
                printf("%d ", in[i]);
        printf("\n");
}


int main(int argc, char *argv[]){
	
	if(argc != 3){
		cout<<" USO "<<argv[0]<<" N K (hebras por bloque) \n";
		return 1;
	}
	int pot = atoi(argv[1]);
        int K = atoi(argv[2]);
	int N = 1<<pot;
        
	int midev;
        cudaGetDevice(&midev);

	cout<<" N "<<N<<" K "<<K<<endl;
        int nb = (N+1)/K;
	cout<<" K "<<K<<" nb "<<nb<<" nb * sizeof(int) "<<nb*sizeof(int)<<endl;

        int size = N*sizeof(int); // num de bytes para in
        int *in = (int *)malloc(size); // puntero a datos entrada en host 
        int *oner = (int *)malloc(sizeof(int)); // puntero a resultado host
  	*oner = 0;

        initA(in,N);
        print(in,N);

	int *d_in, *d_out, *d_one;
	cudaMalloc(&d_in, size); // espacio para datos entrada en gpu
	cudaMalloc(&d_out, nb*sizeof(int)); // espacio para datos intermedios en gpu  
	cudaMalloc(&d_one, sizeof(int)); // espacio en gpu para resultado final
	cudaMemcpy(d_in, in, size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_one, oner , sizeof(int),  cudaMemcpyHostToDevice);
	
        int tpb = K;
	dim3 dimBlock(K,K);
        int sharebytes = K*sizeof(int); // espacio datos compartidos
        auto st_time = std::chrono::high_resolution_clock::now();
	reduce_a<<<nb,tpb,sharebytes>>>(d_in,d_one,N); //num bloques = N/K  
        auto e_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> ttime = e_time - st_time;
        std::cout<<" tiempo kernel paso 1 "<<ttime.count()<<"s"<<std::endl;

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) 
                std::cout<<"Error: "<<cudaGetErrorString(err)<<std::endl;

        cudaDeviceSynchronize();

	cudaMemcpy(oner, d_one, sizeof(int), cudaMemcpyDeviceToHost);

	cudaFree(d_in);
	cudaFree(d_out);
	cudaFree(d_one);

	cout<<" out\n";
	cout<<" oner "<<*oner<<endl;

	free(in);
	free(oner);

	cout<<" fin \n";
}
