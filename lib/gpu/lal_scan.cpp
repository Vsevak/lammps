/*
 * lal_sort.cpp
 *
 *  Created on: Aug 15, 2020
 *      Author: vsevak
 */

#include <cmath>
#include "lal_scan.h"

namespace LAMMPS_AL {

// This value must be consistent with BLOCK in lal_scan.cu
// BLOCK = Scan::block_size
const int Scan::block_size = 256;

Scan::Scan(UCL_Device &d) : gpu(d) {
  //block_res.alloc(8, *gpu);
  compile_kernels();
}

void Scan::compile_kernels() {
  UCL_Program dev_program(gpu);
  std::string flags = ""; //"-D"+std::string(OCL_VENDOR);
  dev_program.load_string(::scan, flags.c_str());
  k_scan.set_function(dev_program,"k_scan");
  k_sum.set_function(dev_program, "k_add");
}

void Scan::scan(UCL_D_Vec<unsigned int> &input,
    UCL_D_Vec<unsigned int> &output, const int n) {

  int t = static_cast<int>(std::ceil(static_cast<double>(n)/block_size));
  UCL_D_Vec<unsigned int> block_res;
  block_res.alloc( std::max(t, 8), gpu);

  k_scan.set_size(t, block_size);
  k_scan.run(&input, &output, &n, &block_res);

  if (t > 1) {
    UCL_D_Vec<unsigned int> block_res_out;
    block_res_out.alloc( std::max(t, 8), gpu);
    scan(block_res, block_res_out, t);
    k_sum.set_size(t, block_size);
    k_sum.run(&block_res, &output, &n);
  }
}

}




