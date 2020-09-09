/*
 * lal_radixsort.cpp
 *
 *  Created on: Aug 15, 2020
 *      Author: Vsevolod Nikolskiy
 *       Email: thevsevak@gmail.com
 */

#include <string>
#include <cmath>
#include "lal_precision.h"
#include "lal_radixsort.h"
#if defined(USE_OPENCL)
#include "sort_cl.h"
#elif defined(USE_CUDART)
const char *sort=0;
#else
#include "sort_cubin.h"
#endif

namespace LAMMPS_AL {

RadixSort::RadixSort(UCL_Device &d, std::string param) :
    gpu(d), ocl_param(param), scanner(d, param) {
  printf("\nGPU Radix Sort enabled\n");
  alloc(0);
  compile_kernels();
}

bool RadixSort::alloc(const int _n) {
  if (_allocated) {
    return resize(_n);
  }
  int n = std::max(_n, 16);
  int rc = k_out.alloc(n, gpu);
  if (rc == UCL_SUCCESS)
    rc = v_out.alloc(n, gpu);
  if (rc == UCL_SUCCESS)
    rc = prefix.alloc(n, gpu);
  int t = static_cast<int>(std::ceil(static_cast<double>(n)/block_size));
  t = std::max(4*t, 8);
  if (rc == UCL_SUCCESS)
    block.alloc(t, gpu);
  if (rc == UCL_SUCCESS)
    scan_block.alloc(t, gpu);
  if (rc == UCL_SUCCESS)
    _allocated = true;
  return rc;
}

void RadixSort::clear() {
  if (!_allocated) return;
  _allocated = false;
  k_out.clear();
  v_out.clear();
  prefix.clear();
  block.clear();
  scan_block.clear();
  scanner.clear();
}

bool RadixSort::resize(const int n) {
  if (!_allocated) return alloc(n);
  k_out.resize_ib(n);
  v_out.resize_ib(n);
  prefix.resize_ib(n);
  int t = static_cast<int>(std::ceil(static_cast<double>(n)/block_size));
  block.resize_ib(4*t);
  scan_block.resize_ib(4*t);
  return UCL_SUCCESS;
}

void RadixSort::sort(UCL_D_Vec<unsigned int> &key,
    UCL_D_Vec<int> &value, const int n) {

  int t = static_cast<int>(std::ceil(static_cast<double>(n)/block_size));
  if (key.cols() < n) {
    printf("\nError: RadixSort key len=%lu is less than n=%d required.\n",
        key.cols(),  n);
    return;
  }
  if (value.cols() < n) {
    printf("\nError: RadixSort key len=%lu is less than n=%d required.\n",
        value.cols(), n);
    return;
  }
//  auto check_size = [n] (auto& vec) { return (int)vec.cols() < n; };
//  if ( check_size(k_out) || check_size(v_out) || check_size(prefix)) {
//    printf("WARNING: resize\n");
//    resize(n);
//  }
  prefix.zero(n);
  block.zero(4*t);
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
    scanner.scan(block, scan_block, 4*t, 0);
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
  dev_program.load_string(::sort, ocl_param.c_str());
  k_local.set_function(dev_program,"k_local");
  k_global_scatter.set_function(dev_program, "k_global_scatter");
  k_check.set_function(dev_program, "k_check");
}
}




