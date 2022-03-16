/*
 * Copyright (C) 2020 and later bymnakao (https://github.com/mnakao).
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 * http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*
 * Modifications copyright (C) 2021 and later by drakawa (https://github.com/drakawa).
 */

#include <stdint.h>

#define UINT64_BITS             64
#define BLOCKS   (28*16)
#define THREADS  (64*16)  /* Must be 2^n */
#define POPCNT(a) __popcll(a)

extern "C"
{
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
//    A[offset] = B[offset] = (0x1ULL << (tid%UINT64_BITS));
    A[offset] = B[offset] = (0x1ULL << (UINT64_BITS-tid%UINT64_BITS-1));
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
