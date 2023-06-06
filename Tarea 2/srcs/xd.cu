#include <stdio.h>
#include <iostream>
#include <cuda_runtime.h>
#include <chrono>
#include <cooperative_groups.h>

using namespace std;
namespace cg = cooperative_groups;

#define blocksz 256

__device__ int reduce_max(cg::thread_group g, int *temp, int val)
{
    int tid = g.thread_rank();
    for (int i = blockDim.x / 2; i > 0; i /= 2)
    {
        temp[tid] = val;
        g.sync();
        if (tid < i)
            val = max(val, temp[tid + i]);
        g.sync();
    }
    return val; // note: only thread 0 will return full sum
}

__global__ void reduce(int *maximos, int *input, int n)
{
    extern __shared__ int temp[];
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    cg::thread_group g = cg::this_thread_block();
    int block_max = reduce_max(g, temp, input[id]);

    if (threadIdx.x == 0)
        atomicMax(maximos, block_max);
}

int main(int argc, char *argv[])
{

    if (argc != 3)
    {
        cout << " USO " << argv[0] << " Nbits K (hebras por bloque) \n";
        return 1;
    }
    int midev;
    cudaGetDevice(&midev);
    int bits = atoi(argv[1]);
    int tpb = atoi(argv[2]);
    // int n = 1<<24;
    // int blockSize = 256;
    int n = 1 << bits;
    int blockSize = tpb;
    int nBlocks = (n + blockSize - 1) / blockSize;
    cout << " nBlocks = " << nBlocks << " ceil(n/blockSize) = " << ceil(n / blockSize) << endl;
    int sharedBytes = blockSize * sizeof(int);

    int *maximos, *data;
    cudaMallocManaged(&maximos, sizeof(int));
    cudaMallocManaged(&data, n * sizeof(int));
    std::fill_n(data, n, 2); // initialize data

    for (int i = 0; i < n; ++i)
        cout << data[i] << " ";

    //cudaStream_t stream[nBlocks]; no funciona
    cudaStream_t *streams = new cudaStream_t[nBlocks];
    for (int i = 0; i < nBlocks; ++i)
        cudaStreamCreate(&streams[i]);

    for (int i = 0; i < nBlocks; ++i)
        reduce<<<1, blockSize, sharedBytes, streams[i]>>>(maximos, data + i * blockSize, blockSize);

    cudaDeviceSynchronize();

    for (int i = 0; i < nBlocks; ++i)
        cout << "Max A" << i + 1 << ": " << maximos[i] << endl;

    for (int i = 0; i < nBlocks; ++i)
        cudaStreamDestroy(streams[i]);
    delete[] streams;
    cudaFree(maximos);
    cudaFree(data);
    return 0;
}
