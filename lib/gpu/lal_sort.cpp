/*
 * lal_sort.cpp
 *
 *  Created on: Aug 15, 2020
 *      Author: vsevak
 */

#include <string>
#include <cmath>
#include "lal_sort.h"

namespace LAMMPS_AL {
// This value must be consistent with BLOCK in lal_sort.cu
const int RadixSort::block_size = 256;


RadixSort::RadixSort(UCL_Device &d) : gpu(d), scanner(d) {
  k_out.alloc(16, gpu);
  v_out.alloc(16, gpu);
  prefix.alloc(16, gpu);
  block.alloc(8, gpu);
  scan_block.alloc(8, gpu);
  compile_kernels();
}

void RadixSort::sort(
    UCL_D_Vec<unsigned int> &key, UCL_D_Vec<int> &value, const int n) {

  int t = static_cast<int>(std::ceil(static_cast<double>(n)/block_size));
  if (key.cols() < n) {
    printf("\nWarning: RadixSort key is too short.\n");
    key.resize(n);
  }
  if (value.cols() < n) {
    printf("\nWarning: RadixSort value is too short.\n");
    value.resize(n);
  }
  k_out.resize_ib(n);
  v_out.resize_ib(n);
  prefix.resize_ib(n);
  prefix.zero(n);
  block.resize_ib(4*t);
  block.zero(4*t);
  scan_block.resize_ib(4*t);
  scan_block.zero(4*t);
  for (int b = 0; b <= 30; b+=2) {
    k_local.set_size(t, block_size);
    k_local.run(&key, &value, &k_out, &v_out, &n, &prefix, &block, &b);
#if RADIX_PRINT
    if (b==0) {
      printf("\n\n========KEY%d===============\n", b);
      ucl_print(k_out, block_size);
      printf("\n==========PREFIX/MERGED==================\n");
      ucl_print(prefix, block_size);
      printf("\n==========BLOCK==================\n");
      ucl_print(block, 4*t);
      printf("\n========OUT===============\n");
    }
#endif
    scanner.scan(block, scan_block, 4*t);
#if RADIX_PRINT
    if (b==0) {
      printf("\n========SCAN BLOCK===============\n");
      ucl_print(scan_block, 4*t);
      printf("\n========OUT===============\n\n");
    }
#endif
    k_global_scatter.set_size(t, block_size);
    k_global_scatter.run(&key, &value, &k_out, &v_out, &n, &prefix, &scan_block, &b);
  }
}


void RadixSort::compile_kernels() {
  UCL_Program dev_program(gpu);
  std::string flags = ""; //"-D"+std::string(OCL_VENDOR);
  dev_program.load_string(::sort, flags.c_str());
  k_local.set_function(dev_program,"k_local");
  k_global_scatter.set_function(dev_program, "k_global_scatter");
}
}




