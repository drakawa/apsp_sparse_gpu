#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <cusparse.h>         // cusparseSpGEMM
#include <stdio.h>            // printf
#include <stdint.h>

#define UINT64_BITS             64
#define BLOCKS   (28*16)
#define THREADS  (64*16)  /* Must be 2^n */
#define POPCNT(a) __popcll(a)

#define CHECK_CUDA(func)                                              \
{                                                                     \
    cudaError_t status = (func);                                      \
    if (status != cudaSuccess) {                                      \
        printf("CUDA API failed at line %d with error: %s (%d)\n",    \
               __LINE__, cudaGetErrorString(status), status);         \
        return EXIT_FAILURE;                                          \
    }                                                                 \
}

#define CHECK_CUSPARSE(func)                                          \
{                                                                     \
    cusparseStatus_t status = (func);                                 \
    if (status != CUSPARSE_STATUS_SUCCESS) {                          \
        printf("CUSPARSE API failed at line %d with error: %s (%d)\n",\
               __LINE__, cusparseGetErrorString(status), status);     \
        return EXIT_FAILURE;                                          \
    }                                                                 \
}

extern "C"
{
// __global__ void hoge_print()
// {
//     printf("hoge\n");
// }
    
__global__ void clear_buffers_dev(uint64_t* __restrict__ A, uint64_t* __restrict__ B, const int length)
{
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  while (tid<length) {
    A[tid] = B[tid] = 0;
    tid += blockDim.x * gridDim.x;
  }
}
__global__ void init_dev(uint64_t* __restrict__ A, uint64_t* __restrict__ B,
			 const int nodes, const unsigned int elements)
{
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  while (tid < nodes) {
    unsigned int offset = tid*elements+tid/UINT64_BITS;
    A[offset] = B[offset] = (0x1ULL << (tid%UINT64_BITS));
    // A[offset] = B[offset] = (0x1ULL << (UINT64_BITS-tid%UINT64_BITS-1));
    tid += blockDim.x * gridDim.x;
  }
}
//__global__ static void matrix_op_dev(const uint64_t* __restrict__ A, uint64_t* __restrict__ B, const int* __restrict__ adjacency,
__global__ void matrix_op_dev(const uint64_t* __restrict__ A, uint64_t* __restrict__ B, const int* __restrict__ adjacency,
				     const int* __restrict__ num_degrees, const int nodes, const int degree, const unsigned int elements)
{
  int tid = threadIdx.x + blockIdx.x * blockDim.x;

  while (tid < nodes*elements) {
    int i = tid / elements;
    int k = tid % elements;
    uint64_t tmp = B[tid];
    for(int j=0;j<num_degrees[i];j++){
      int n = *(adjacency + i * degree + j);  // int n = adjacency[i][j];
      tmp |= A[n*elements+k];
    }
    B[tid] = tmp;
    tid += blockDim.x * gridDim.x;
  }
}
    
__global__ void matrix_op_dev2(const uint64_t* __restrict__ A, uint64_t* __restrict__ B, const int* __restrict__ adjacency,
				     const int* __restrict__ num_degrees, const int nodes, const int degree, const unsigned int elements)
{
  int tid = threadIdx.x + blockIdx.x * blockDim.x;

  while (tid < nodes*elements*degree) {
      int i = tid / (elements*degree);
      int j = (tid % (elements*degree)) / elements;
      int k = (tid % (elements*degree)) % elements;

      int n = *(adjacency + i * degree + j);  // int n = adjacency[i][j];
      atomicOr((unsigned long long*)(B + i * elements + k), A[n * elements + k]); // B[i][k] = A[n][k]
      
      tid += blockDim.x * gridDim.x;
  }
}
    
__global__ void matrix_op_dev_sp(uint64_t* __restrict__ B, const int* __restrict__ adjacency,
				 const int* __restrict__ num_degrees, const int nodes, const int degree, const unsigned int elements, uint32_t* __restrict__ B_sp, const unsigned int sp_start, const unsigned int sp_end, const unsigned int max_k)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    while (tid < nodes * degree * max_k) {
	int i = tid / (degree * max_k);
	int j = (tid % (degree * max_k)) / max_k;
	int k = (tid % (degree * max_k)) % max_k;
	
	int n = *(adjacency + i * degree + j); // int n = adjacency[i][j]
	int nxt = *(B_sp + (sp_start+k) * nodes + n); // int nxt = B_sp[sp_start+k][n]
	*(B_sp + (sp_end + k * degree + j) * nodes + i) = nxt; // B_sp[sp_end+k*num_degrees[i]+j][i] = nxt
	// *(B + i * elements + nxt / UINT64_BITS) |= 0x1ULL << nxt % UINT64_BITS; // B[i][nxt/64] = 1<<nxt%64
	
	// if (atomicAnd((unsigned long long*)(B + i * elements + nxt / UINT64_BITS), 0x1ULL << nxt % UINT64_BITS) == 0x0ULL)
	//     atomicOr((unsigned long long*)(B + i * elements + nxt / UINT64_BITS), 0x1ULL << nxt % UINT64_BITS); // B[i][nxt/64] = 1<<nxt%64
	atomicOr((unsigned long long*)(B + i * elements + nxt / UINT64_BITS), 0x1ULL << nxt % UINT64_BITS); // B[i][nxt/64] = 1<<nxt%64
	
	// int tmp_degree = num_degrees[i];
	// for(int j=0;j<tmp_degree;j++){
	//     int n = *(adjacency + i * degree + j); // int n = adjacency[i][j]
	//     for (int k=0; k<max_k; k++) {
	// 	int nxt = *(B_sp + (sp_start+k) * nodes + n); // int nxt = B_sp[sp_start+k][n]
	// 	*(B_sp + (sp_end + k * tmp_degree + j) * nodes + i) = nxt; // B_sp[sp_end+k*num_degrees[i]+j][i] = nxt
	// 	*(B + i * elements + nxt / UINT64_BITS) |= 0x1ULL << nxt % UINT64_BITS; // B[i][nxt/64] = 1<<nxt%64
	//     }
	// }
	tid += blockDim.x * gridDim.x;
    }
}
//__global__ static void popcnt_dev(const uint64_t* __restrict__ B, const int nodes, 
__global__ void popcnt_dev(const uint64_t* __restrict__ B, const int nodes, 
				  const unsigned int elements, uint64_t* __restrict__ result)
{
  __shared__ uint64_t cache[THREADS];
  int cacheIndex = threadIdx.x;
  int tid = threadIdx.x + blockIdx.x * blockDim.x;

  uint64_t num = 0;
  while (tid < elements*nodes) {
    num += POPCNT(B[tid]);
    tid += blockDim.x * gridDim.x;
  }
  cache[cacheIndex] = num;
  __syncthreads();

  int i = blockDim.x/2;
  while (i != 0){
    if (cacheIndex < i)
      cache[cacheIndex] += cache[cacheIndex+i];
    __syncthreads();
    i /= 2;
  }

  if(cacheIndex == 0)
    result[blockIdx.x] = cache[0];
}
}
