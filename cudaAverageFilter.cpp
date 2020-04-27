// Problem: Apply average filter on a 2D matrix using CUDA 
#include <iostream>
#include <cuda_runtime>

#define BLOCKSIZE 256
#define RADIUS 3
using namespace std;

__global__ void averageKernel(float *out, float *in, int rows, int cols)
{
    //calculate the output's element coordinates to work on
    int gcol = threadIdx.x + blockIdx.x * blockDim.x;
    int grow = threadIdx.y + blockIdx.y * blockDim.y;

    //calculate the shared memory's coordinates to fetch into
    // (need to offset by RADIUS to accomodate halo elements)
    int lcol = threadIdx.x + RADIUS;
    int lrow = threadIdx.y + RADIUS;

    //create a shared memory accomodating the halo elements for the first and the last rows / cols
    __shared__ float tile[BLOCKSIZE + 2 * RADIUS][BLOCKSIZE + 2 * RADIUS];

    // fetch from input: in[grow][gcol] into shared memory: tile[lrow][lcol]
    tile[lrow][lcol] = in[grow * len + gcol];

    // let the first (RADIUS x RADIUS) threads fetch the corner halo elements
    if (threadIdx.y < RADIUS && threadIdx.x < RADIUS)
    {
        // Copy up-left, up-right, down-left, down-right elements
        tile[lrow - RADIUS][lcol - RADIUS] = in[(grow - RADIUS) * col + (gcol - RADIUS)];
        tile[lrow - RADIUS][lcol + RADIUS] = in[(grow - RADIUS) * col + (gcol + RADIUS)];
        tile[lrow + BLOCKSIZE][lcol - RADIUS] = in[(grow + BLOCKSIZE) * col + (gcol - RADIUS)];
        tile[lrow + BLOCKSIZE][lcol + RADIUS] = in[(grow + BLOCKSIZE) * col + (gcol + RADIUS)];
    }
    // Copy up and down elements: to be done by first RADIUS number of row threads
    if (threadIdx.y < RADIUS)
    {
        tile[lrow - RADIUS][lcol] = in[(grow - RADIUS) * col + gcol];
        tile[lrow + RADIUS][lcol] = in[(lrow + RADIUS) * col + gcol];
    }
    // copy left and right elements: to be done by first RADIUS number of col threads
    if (threadIdx.x < RADIUS)
    {
        tile[lrow][lcol - RADIUS] = in[(grow)*col + (gcol - RADIUS)];
        tile[lrow][lcol + RADIUS] = in[(lrow)*col + (lcol + RADIUS)];
    }

    // wait for all the threads to completely fill the shared memory
    __synchthreads();

    // Find the average in (2*RADIUS+1) x (2*RADIUS+1) nbhd
    float result = 0;
    for (int i = -RADIUS; i <= RADIUS : i++)
        for (int j = -RADIUS; j <= RADIUS; j++)
        {
            result += tile[lrow + i][lcol + j];
        }
    out[grow * col + gcol] = result / ((2 * RADIUS + 1) * (2 * RADIUS + 1));
}

void average(float *out, float *in, int len)
{
    dim3 threadsPerBlock(BLOCKSIZE), blocksPerGrid(ceil(len / BLOCKSIZE));
    averageKernel<<<blocksPerGrid, threadsPerBlock>>>(out, in, len);
}

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
    int m, n;
    vector<float> A(m * n), B(m * n);
    d_vector<float> d_A(m * n), d_B(m * n);
    d_A.set(&A[0], m * n);
    average(d_B.getData(), d_A.getData(), m * n);
    cudaDeviceSynchronize();
    d_B.get(&B[0], m * n);
}