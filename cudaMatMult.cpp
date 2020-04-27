// Problem: Matrix multiplication in CUDA without shared memory
#include <bits/stdc++.h>
#include <cuda_runtime.h>

#define BLOCKSIZE 16
using namespace std;

__global__ void matrixMult(float *d_A, float *d_B, float *d_C, int m, int p, int n)
{
    // each thread works on one element in the block
    int rowID = blockIdx.y * BLOCKSIZE + threadIdx.y;
    int colID = blockIdx.x * BLOCKSIZE + threadIdx.x;
    int sum = 0;

    if (rowID < m and colID < n)
    {
        for (int i = 0; i < p; i++)
            sum += d_A[rowID * m + i] * d_B[colID * p + i];
    }

    d_C[rowID * m + colID] += sum;
}

void matrixMult(const d_vector<float> d_A, const d_vector<float> d_B, d_vector<float> d_C, int m, int p, int n)
{
    dim3 blocksPerGrid(ceil(n / BLOCKSIZE), ceil(p / BLOCKSIZE));
    dim3 threadsPerBlock(BLOCKSIZE, BLOCKSIZE);
    matrixMult<<<(blocksPerGrid, threadsPerBlock)>>>(d_A, d_B, d_C, m, p, n);
}

template <class T>
class d_vector
{
    T *start, *end;
    void allocate(size_t size)
    {
        size_t len = sizeof(T) * size;
        cudaError_t success = cudaMalloc((void **)&start, len);
        if (success != cudaSuccess)
        {
            start = end = 0;
            runtime_error("failed to allocate device memory");
        }
        end = start + size;
    }

public:
    d_vector() : *start{0}, *end{0} {}
    d_vector(size_t size)
    {
        allocate(size);
    }
    ~d_vector()
    {
        cudaFree(start);
        start = end = 0;
    }

    void set(const T *src, size_t size)
    {
        cudaError_t success = cudaMemcpy(start, src, size * sizeof(T), cudaMemcpyHostToDevice);
        if (success != cudaSuccess)
        {
            runtime_error("failed to copy host to device");
        }
    }
    void get(T *dst, size_t size)
    {
        cudaError_t success = cudaMemcpy(dst, start, size * sizeof(T), cudaMemcpyDeviceToHost);
        if (success != cudaSuccess)
        {
            runtime_error("failed to copy device to host");
        }
    }
    T *getData()
    {
        return start;
    }
};

int main()
{
    int m, p, n;
    m = 16;
    p = 16;
    q = 16;
    vector<float> h_A(m * p);
    vector<float> h_B(p * q);
    vector<float> h_C(m * q);
    d_vector<float> d_A(m * p);
    d_vector<float> d_B(p * q);
    d_vector<float> d_C(m * q);

    d_A.set(&h_A[0], m * p);
    d_B.set(&h_B[0], p * q);
    matrixMult(d_A.getData(), d_B.getData(), d_C.getData(), m, p, q);
    // wait for the cuda execution to complete before going to next step
    cudaDeviceSynchronize();
    d_C.get(&h_C[0], m * q);
    // wait for the copying to complete before going to next step
    cudaDeviceSynchronize();
}