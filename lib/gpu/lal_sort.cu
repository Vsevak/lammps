/*
 * lal_sort.cu
 *
 *  Created on: Aug 14, 2020
 *      Author: vsevak
 */

#if defined(NV_KERNEL) || defined(USE_HIP)
#include "lal_aux_fun1.h"
#endif

__kernel void k_local(__global int *restrict x, const int n){
  int tid = BLOCK_ID_X*BLOCK_SIZE_X + THREAD_ID_X;
  if (tid < n){
    x[tid] *= 5;
  }
}

__kernel void k_global(__global int *restrict x, const int n){
  int tid = BLOCK_ID_X*BLOCK_SIZE_X + THREAD_ID_X;
  if (tid < n){
    x[tid] *= 100;
  }
}

