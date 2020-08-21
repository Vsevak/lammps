/*
 * lal_sort.cpp
 *
 *  Created on: Aug 15, 2020
 *      Author: vsevak
 */

#include <cmath>
#include "lal_scan.h"
#if defined(USE_OPENCL)
#include "scan_cl.h"
#elif defined(USE_CUDART)
const char *scan=0;
#else
#include "scan_cubin.h"
#endif

namespace LAMMPS_AL {

// This value must be consistent with BLOCK in lal_scan.cu
// BLOCK = Scan::block_size
const int Scan::block_size = 256;

Scan::Scan(UCL_Device &d, std::string param) : gpu(d), ocl_param(param) {
  block_res.reserve(10);
  block_res_out.reserve(10);
  compile_kernels();
}

void Scan::compile_kernels() {
  UCL_Program dev_program(gpu);
  dev_program.load_string(::scan, ocl_param.c_str());
  k_scan.set_function(dev_program,"k_scan");
  k_sum.set_function(dev_program, "k_add");
}

void Scan::scan(UCL_D_Vec<unsigned int> &input,
    UCL_D_Vec<unsigned int> &output, const int n, const int iter) {

  int t = static_cast<int>(std::ceil(static_cast<double>(n)/block_size));
  while (block_res.size() < iter+1) {
    block_res.emplace_back(UCL_D_Vec<unsigned int>());
    block_res.back().alloc( std::max(t, 8), gpu);
  }
  block_res[iter].resize_ib(t);

  k_scan.clear_args();
  k_scan.set_size(t, block_size);
  gpu.sync();
  k_scan.run(&input, &output, &n, &(block_res[iter]));
  gpu.sync();
  if (t > 1) {
    while (block_res_out.size() < iter+1) {
      block_res_out.emplace_back(UCL_D_Vec<unsigned int>());
      block_res_out.back().alloc( std::max(t, 8), gpu);
    }
    block_res_out[iter].resize_ib(t);
    scan(block_res[iter], block_res_out[iter], t, iter+1);
    k_sum.set_size(t, block_size);
    k_sum.run(&block_res_out[iter], &output, &n);
  }
}

}




