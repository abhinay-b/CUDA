// Problem: Apply 1D stencil to given 1D array of elements
// Result: Each output element is the sum of input elements within a radius
#include <iostream>
#include <cuda_runtime.h>

#define BLOCKSIZE 512
#define RADIUS 4

using namespace std;

// kernel function
__global__ void stencil1D(float *out, float *in, int len)
{
    // We'll use a shared block to reduce the global memory accesses  
    // calculate the global and local index to be assigned to the thread where
    // global index accesses the global memory (i.e the input array) and 
    // the local index accesses the share memory
    int gidx = threadIdx.x + blockIdx.x * blockDim.x;
    
    int lidx = threadIdx.x + RADIUS;
    // create a shared memory for the thread block (border elements (leftmost & rightmost) need halo elements)
    __shared__ float tile[BLOCKSIZE + 2 * RADIUS];
    
    // Each thread in the block loads an element from global to local memory  
    tile[lidx] = in[gidx];
    
    // Each of the first ('RADIUS' number of) threads load one of the haloed element to the left and right
    if (threadIdx.x < RADIUS)
    {
        tile[lidx - RADIUS] = in[gidx - RADIUS];
        tile[lidx + BLOCKSIZE] = in[gidx + BLOCKSIZE];
    }
    
    // Synchronise so that all the threads wait for the shared memory to be loaded completely 
    __synchthreads();
    float result = 0;
    for (int i = -RADIUS; i <= RADIUS; i++)
        result += tile[lidx + i];
    out[gidx] = result;
    
}

// A function to design the dimensions of the cuda grid 
void stencil(float *out, float *in, int len)
{
    dim3 threadsPerBlock(BLOCKSIZE), blocksPerGrid(ceil(len / BLOCKSIZE));
    stencil1D<<<blocksPerGrid, threadsPerBlock>>>(out, in, len);
}

// A class designed to abstract out CUDA operations
template <class T>
class d_vector
{
    T *start;
    T *end;
    void allocate(size_t size)
    {
        cudaError_t success = cudaMalloc((void **)&start, size * sizeof(T));
        if (success != cudaSuccess)
        {
            std::runtime_error << "couldn't allocate memory in device\n";
        }
        end = start + size;
    }
    void free()
    {
        cudaFree(start);
        start = end = 0;
    }

public:
    d_vector() = default;
    d_vector(size_t size)
    {
        allocate(size);
    }
    ~d_vector()
    {
        free();
    }
    T *getData()
    {
        return start;
    }
    void set(T *in, size_t size)
    {
        cudaError_t success = cudaMemcpy(start, in, size * sizeof(T), cudaMemcpyHostToDevice);
    }
    void get(T *out, size_t size)
    {
        cudaError_t success = cudaMemcpy(out, start, size * sizeof(T), cudaMemcpyDeviceToHost);
    }
};

int main()
{
    vector<float> A = {1.1,2.7,3,4.3,5.5,6}, B;
    d_vector<float> d_A(A.size()), d_B(A.size());
    d_A.set(&A[0], A.size());
    stencil(d_B.getData(), d_A.getData(), A.size());
    cudaDeviceSynchronize();
    d_B.get(&B[0], A.size());
}