# apsp_sparse_gpu

Measuring the average shortest path length (APSP) for low-degree unweighted regular graphs \[1\].
This modified method is based on the APSP parallelization method in unweighted graphs \[2\].

\[1\] Ryuta Kawano, Hiroki Matsutani, Michihiro Koibuchi, Hideharu Amano, "GPU Parallelization of All-Pairs-Shortest-Path Algorithm in Low-Degree Unweighted Regular Graph", Proc. of the 8th ACIS International Virtual Conference on Applied Computing & Information Technology (ACIT 2021), pp.xx–xx, Jun 2021.
\[2\] M. Nakao, H. Murai, and M. Sato, “Parallelization of All-Pairs-Shortest-Path Algorithms in Unweighted Graph,” in Proceedings of the International Conference on High Performance Computing in Asia-Pacific Region, Fukuoka, Japan, Jan. 2020, pp. 63–72. doi: 10.1145/3368474.3368478.

# Required
* cuda
* python3,pycuda,networkx,numpy

# Usage
    $ python3 matrix_op_gemm.py -f 6_65536.edges # Modified method
    $ python3 matrix_op.py -f 6_65536.edges # Original method
    
