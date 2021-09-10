#!/usr/bin/env python3

import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

import networkx as nx
from optparse import OptionParser
import numpy as np

from time import perf_counter

import math

UINT64_BITS = 8 * np.dtype(np.uint64).itemsize
SIZEOF_UINT64_T = np.dtype(np.uint64).itemsize
SIZEOF_UINT = np.dtype(np.uint32).itemsize
SIZEOF_UINT32 = np.dtype(np.uint32).itemsize

BLOCK_X = 28
BLOCK_Y = 16
BLOCKS = BLOCK_X * BLOCK_Y
DIM3_BLOCKS = (BLOCKS, 1, 1)

THREAD_X = 64
THREAD_Y = 16
THREADS = THREAD_X * THREAD_Y
DIM3_THREADS = (THREADS, 1, 1)
MAX_SPLEN = 1024 ** 3 # 1GB

cu_source = "matrix_op_gemm.cu"
with open(cu_source) as f:
    cu_source_txt = f.read()
mod = SourceModule(cu_source_txt, no_extern_c=True)
# mod = SourceModule(cu_source_txt)

def read_edges(inf):
    G = nx.read_edgelist(inf, nodetype=np.uint32)
    return G

def bool_to_uint32x4(arr):
    result_arr_len = int(round(math.ceil(arr.shape[0] / ((SIZEOF_UINT*8)*4)))) * 4
    # print(result_arr_len)
    result_arr = np.zeros(result_arr_len, dtype=np.uint32)
    for i, a in enumerate(arr):
        if a:
           result_arr[i // (SIZEOF_UINT*8)] |= 1<<((SIZEOF_UINT*8)-(i % (SIZEOF_UINT*8))-1)
    # for ra in result_arr:
        # print(format(ra, "032b"))
    return result_arr
    # print(result_arr)
    
def gen_myadj(adjacency, degree, nodes):
    # result_adj = np.zeros((8,0), dtype=np.uint32)
    result_adj = np.zeros((0, math.ceil((degree*8)/128)*4), dtype=np.uint32)
    result_idx = np.zeros((0, math.ceil((degree*8)/128)*128), dtype=np.uint32)
    print(adjacency.shape)
    for i in range(0, adjacency.shape[0], 8):
        tmp_adj = adjacency[i:i+8]
        # print(tmp_adj)
        tmp_sorted = np.sort(tmp_adj.flatten())
        tmp_sorted = np.pad(tmp_sorted, [0, math.ceil((degree*8)/128)*128-tmp_sorted.shape[0]], "constant", constant_values=(0,nodes))
        result_idx = np.vstack([result_idx, tmp_sorted])
        # print(tmp_sorted)
        # print(result_idx)

        tmp_dict = {k:v for (v,k) in enumerate(tmp_sorted)}
        # print(tmp_dict)
        tmp_bool = np.zeros((8, degree*8), dtype=bool)
        # print(tmp_bool)

        # print(tmp_adj[1][0])
        for j in range(min(8, tmp_adj.shape[0])):
            for k in range(degree):
                # print(i, tmp_dict[tmp_adj[i][j]])
                # print(j, k, tmp_adj)
                # print(j, k, tmp_adj[j][k], tmp_dict[tmp_adj[j][k]])
                tmp_bool[j][tmp_dict[tmp_adj[j][k]]] = True

        tmp_result_adj = np.zeros((0, math.ceil((degree*8)/128)*4), dtype=np.int32)
        # print(tmp_bool)
        for j in range(8):
            tmp_uint32x4 = bool_to_uint32x4(tmp_bool[j])
            # print(tmp_uint32x4)
            tmp_result_adj = np.vstack([tmp_result_adj, tmp_uint32x4])
            # print(tmp_result_adj)
        # print(result_adj)
        result_adj = np.vstack([result_adj, tmp_result_adj])
        print(i, result_adj.shape, result_idx.shape)
        # if (i > 16):
        #     exit(1)

def gen_inputs(inf):
    G = read_edges(inf)
    nodes = sorted(G.nodes())
    degree = max(dict(G.degree()).values())
    adjacency = np.array([list(sorted(dict(G[i]).keys())) for i in nodes], dtype=np.uint32)
    num_degrees = np.array([G.degree()[i] for i in nodes], dtype=np.uint32)

    # gen_myadj(adjacency, degree, len(nodes))
    # exit(1)

    return nodes, degree, adjacency, num_degrees, G

def calc_splen(nodes, degree): # sparse matrix no saidai chou wo keisan suru

    print("SIZEOF_UINT:", SIZEOF_UINT)
    # print(nodes, degree)
    nnd = int(round(nodes ** 2 * degree / 64))
    # ndk = nodes
    ndk = nodes * 5
    splen = ndk * SIZEOF_UINT
    num_iter = 0
    prev_num_iter, prev_splen = num_iter, splen
    while ndk < nnd and splen < MAX_SPLEN:
    # while ndk < nnd:
        print("num_iter: {}, nnd: {:,}, ndk: {:,} splen: {:,}".format(num_iter, nnd, ndk, splen))
        prev_num_iter, prev_splen = num_iter, splen
        num_iter += 1
        ndk *= degree
        splen += ndk * SIZEOF_UINT
    print("num_iter: {}, nnd: {:,}, ndk: {:,} splen: {:,}".format(num_iter, nnd, ndk, splen))
    return prev_num_iter, prev_splen

def init_matrix_dev(nodes, degree, num_degrees):
    elements = (nodes + UINT64_BITS - 1) // UINT64_BITS
    s = elements
    s *= nodes * SIZEOF_UINT64_T

    a_dev = cuda.mem_alloc(s)
    b_dev = cuda.mem_alloc(s)
    result = np.zeros(BLOCK_X*BLOCK_Y, dtype=np.int64)
    result_dev = cuda.mem_alloc(SIZEOF_UINT64_T * BLOCKS)
    ### hack by kawano ###
    adjacency_dev = cuda.mem_alloc(SIZEOF_UINT * nodes * degree)
    # adjacency_dev = cuda.mem_alloc(SIZEOF_UINT * nodes * (degree+1))
    num_degrees_dev = cuda.mem_alloc(SIZEOF_UINT * nodes)
    # print(type(num_degrees_dev), type(num_degrees))
    # print(num_degrees, nodes, SIZEOF_UINT)
    cuda.memcpy_htod(num_degrees_dev, num_degrees)

    sp_num_iter, splen = calc_splen(nodes, degree)
    # print("({:,}, {:,})".format(sp_num_iter, splen))

    #### TODO: B_SP wo gyaku ni suru ####
    print("allocated:", s * 2 + SIZEOF_UINT64_T * BLOCKS + SIZEOF_UINT * nodes * degree + SIZEOF_UINT * nodes)
    print("splen:", splen)
    b_sp_dev = cuda.mem_alloc(splen)

    b_sp = np.zeros((splen // (nodes*SIZEOF_UINT32), nodes), dtype=np.uint32)
    # print("b_sp:", b_sp.shape)
    # print(b_sp)
    b_sp[0,:]=np.array(range(nodes), dtype=np.uint32)
    # print(b_sp)

    cuda.memcpy_htod(b_sp_dev, b_sp)

    sp_row = splen // (nodes*SIZEOF_UINT32)
    # print("sp_row:", sp_row)
    # exit(1)
    return a_dev, b_dev, result, result_dev, adjacency_dev, num_degrees_dev, b_sp_dev, sp_row, sp_num_iter

def matrix_op(nodes, degree, adjacency, num_degrees_dev, adjacency_dev, a_dev, b_dev, b_sp_dev, sp_row, sp_num_iter, result_dev, result):
    
    sum = nodes * (nodes - 1)
    diameter = 1
    
    cuda.memcpy_htod(adjacency_dev, adjacency)
    # print("adj[0]:", adjacency[0])
    elements = (nodes + UINT64_BITS-1) // UINT64_BITS
    func = mod.get_function("clear_buffers_dev")
    func(a_dev, b_dev, np.uint32(nodes*elements), grid=DIM3_BLOCKS, block=DIM3_THREADS)
    func = mod.get_function("init_dev")
    func(a_dev, b_dev, np.uint32(nodes), np.uint32(elements), grid=DIM3_BLOCKS, block=DIM3_THREADS)

    ### for debug ###
    # sp_num_iter -= 1
    # sp_num_iter = 4
    # print("sp_num_iter:", sp_num_iter)
    
    timer_prev = perf_counter()
    
    sp_start = 0
    print("sp_num_iter:", sp_num_iter)
    for kk in range(nodes):
    # for kk in range(10):
        if kk < sp_num_iter:
            func = mod.get_function("matrix_op_dev_sp")
            func(b_dev, adjacency_dev, num_degrees_dev, np.uint32(nodes), np.uint32(degree), np.uint32(elements), b_sp_dev, np.uint32(sp_start), np.uint32(sp_start+degree**kk), np.uint32(degree**kk), grid=DIM3_BLOCKS, block=DIM3_THREADS)
            sp_start += degree ** kk
            # print("hoge")
        else:
            func = mod.get_function("matrix_op_dev")
            # func = mod.get_function("matrix_op_dev2")
            func(a_dev, b_dev, adjacency_dev, num_degrees_dev, np.uint32(nodes), np.uint32(degree), np.uint32(elements), grid=DIM3_BLOCKS, block=DIM3_THREADS)
        func = mod.get_function("popcnt_dev")
        func(b_dev, np.uint32(nodes), np.uint32(elements), result_dev, grid=DIM3_BLOCKS, block=DIM3_THREADS)
        cuda.memcpy_dtoh(result, result_dev)
        #print(result)

        num = 0
        for i in range(BLOCKS):
            num += result[i]
        # print("num:{:,}".format(num))
        timer_tmp = perf_counter()
        print("iter: %d" % kk, "num: %d" % num, "time: %f" % (timer_tmp - timer_prev))
        timer_prev = timer_tmp
        if num == nodes * nodes:
            break
        
        if kk == sp_num_iter - 1:
            a_dev, b_dev = b_dev, a_dev
        elif kk < sp_num_iter:
            pass
        else:
            a_dev, b_dev = b_dev, a_dev
            
            
        sum += nodes * nodes - num
        diameter += 1
        # if kk == 10:
        #     exit(1)
        #     popcnt_dev <<< BLOCKS, THREADS >>> (B_dev, nodes, elements, result_dev);

    diameter = max((diameter, kk+1))
    if diameter > nodes:
        print("This graph is not connected graph.")
        exit(1)
        
    ASPL = sum / (nodes * (nodes - 1))
    sum /= 2

    return ASPL, diameter, sum

# mod = SourceModule("""
# #include <stdint.h>

# #define UINT64_BITS             64
# #define BLOCKS   (28*16)
# #define THREADS  (64*16)  /* Must be 2^n */
# #define POPCNT(a) __popcll(a)

# extern "C"{
# __global__ void clear_buffers_dev(uint64_t* __restrict__ A, uint64_t* __restrict__ B, const int length)
# {
#   int tid = threadIdx.x + blockIdx.x * blockDim.x;
#   while (tid<length) {
#     A[tid] = B[tid] = 0;
#     tid += blockDim.x * gridDim.x;
#   }
# }
# __global__ void init_dev(uint64_t* __restrict__ A, uint64_t* __restrict__ B,
# 			 const int nodes, const unsigned int elements)
# {
#   int tid = threadIdx.x + blockIdx.x * blockDim.x;
#   while (tid < nodes) {
#     unsigned int offset = tid*elements+tid/UINT64_BITS;
# //    A[offset] = B[offset] = (0x1ULL << (tid%UINT64_BITS));
#     A[offset] = B[offset] = (0x1ULL << (UINT64_BITS-tid%UINT64_BITS-1));
#     tid += blockDim.x * gridDim.x;
#   }
# }
# //__global__ static void matrix_op_dev(const uint64_t* __restrict__ A, uint64_t* __restrict__ B, const int* __restrict__ adjacency,
# __global__ void matrix_op_dev(const uint64_t* __restrict__ A, uint64_t* __restrict__ B, const int* __restrict__ adjacency,
# 				     const int* __restrict__ num_degrees, const int nodes, const int degree, const unsigned int elements)
# {
#   int tid = threadIdx.x + blockIdx.x * blockDim.x;

#   while (tid < nodes*elements) {
#     int i = tid / elements;
#     int k = tid % elements;
#     uint64_t tmp = B[tid];
#     for(int j=0;j<num_degrees[i];j++){
#       int n = *(adjacency + i * degree + j);  // int n = adjacency[i][j];
#       tmp |= A[n*elements+k];
#     }
#     B[tid] = tmp;
#     tid += blockDim.x * gridDim.x;
#   }
# }
# //__global__ static void popcnt_dev(const uint64_t* __restrict__ B, const int nodes, 
# __global__ void popcnt_dev(const uint64_t* __restrict__ B, const int nodes, 
# 				  const unsigned int elements, uint64_t* __restrict__ result)
# {
#   __shared__ uint64_t cache[THREADS];
#   int cacheIndex = threadIdx.x;
#   int tid = threadIdx.x + blockIdx.x * blockDim.x;

#   uint64_t num = 0;
#   while (tid < elements*nodes) {
#     num += POPCNT(B[tid]);
#     tid += blockDim.x * gridDim.x;
#   }
#   cache[cacheIndex] = num;
#   __syncthreads();

#   int i = blockDim.x/2;
#   while (i != 0){
#     if (cacheIndex < i)
#       cache[cacheIndex] += cache[cacheIndex+i];
#     __syncthreads();
#     i /= 2;
#   }

#   if(cacheIndex == 0)
#     result[blockIdx.x] = cache[0];
# }
# }
# """)

if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("-f", "--file", dest="filename",
                      help="write report to FILE", metavar="FILE")
    (options, args) = parser.parse_args()
    # G = read_edges(options.filename)
    # print(G.nodes())
    # nodes = sorted(G.nodes())
    # degree = max(dict(G.degree()).values())
    # print(nodes)
    # print(degree)
    # print(G[0])
    # print(list(sorted(dict(G[0]).keys())))
    # adjacency = np.array([list(sorted(dict(G[i]).keys())) for i in nodes], dtype=np.int)
    # print(adjacency)
    # num_degrees = np.array([G.degree()[i] for i in nodes], dtype=np.int)
    # print(num_degrees)

    nodes_list, degree, adjacency, num_degrees, G = gen_inputs(options.filename)
    nodes = len(nodes_list)
    # print(nodes)
    # print(degree)
    # print(adjacency)
    # print(num_degrees)
    # print(nx.average_shortest_path_length(G))
    # print(nx.diameter(G))

    (a_dev, b_dev, result, result_dev, adjacency_dev, num_degrees_dev, b_sp_dev, sp_row, sp_num_iter) = init_matrix_dev(nodes, degree, num_degrees)

    start = perf_counter()
    ASPL, diameter, sum = matrix_op(nodes, degree, adjacency, num_degrees_dev, adjacency_dev, a_dev, b_dev, b_sp_dev, sp_row, sp_num_iter, result_dev, result)
    end = perf_counter()

    print(ASPL, diameter, sum)
    print("total_time:", end - start)

    
