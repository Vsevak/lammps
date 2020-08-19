/*
 * sort.h
 *
 *  Created on: Aug 14, 2020
 *      Author: vsevak
 */

#ifndef LAL_SORT_H_
#define LAL_SORT_H_

#include "lal_precision.h"
#if defined(USE_OPENCL)
#include "sort_cl.h"
#elif defined(USE_CUDART)
const char *sort=0;
#else
#include "sort_cubin.h"
#endif
#include "lal_scan.h"

namespace LAMMPS_AL {

#define RADIX_PRINT 0

class RadixSort {
public:
  RadixSort(UCL_Device &gpu);
  ~RadixSort() = default;
  void sort(UCL_D_Vec<unsigned int> &k, UCL_D_Vec<int> &v, const int n);
private:
  bool is_sorted(UCL_D_Vec<unsigned int> &input, const int n);
  void compile_kernels();

  UCL_Device &gpu;
  UCL_Kernel k_local, k_global_scatter;

  Scan scanner;
  UCL_D_Vec<unsigned int> k_out;
  UCL_D_Vec<int> v_out;
  UCL_D_Vec<unsigned int> prefix;
  UCL_D_Vec<unsigned int> block;
  UCL_D_Vec<unsigned int> scan_block;

  static const int block_size;
};

}

#endif /* LAL_SORT_H_ */
