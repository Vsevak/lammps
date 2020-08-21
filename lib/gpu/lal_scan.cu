/*
 * lal_scan.cu
 *
 *  Created on: Aug 17, 2020
 *      Author: vsevak
 */

#if defined(NV_KERNEL) || defined(USE_HIP)
#include "lal_aux_fun1.h"
#endif

#ifdef __CDT_PARSER__
#include "lal_precision.h"
#include "lal_preprocessor.h"
#include "lal_aux_fun1.h"
#endif
// This value must be consistent with lal_scan.h
// BLOCK = Scan::block_size
#define BLOCK 256

__kernel void k_scan (
    __global unsigned int *input,
    __global unsigned int *output,
    const int n,
    __global unsigned int *block) {

  __local unsigned int l_tmp[BLOCK+1];
  int tid = THREAD_ID_X;
  int gid = GLOBAL_ID_X;

  if (gid < n) {
    l_tmp[tid] = input[gid];
  } else {
    l_tmp[tid] = 0;
  }
  __syncthreads();
  int target = 0;
  unsigned int sum = 0;
  int total = (int) log2((float)BLOCK);
  for (unsigned int step = 0; step < total; ++step) {
    target = tid - (1 << step);
    if (target >= 0) {
      sum = l_tmp[tid] + l_tmp[target];
    } else {
      sum = l_tmp[tid];
    }
    __syncthreads();
    l_tmp[tid] = sum;
    __syncthreads();
  }


  // Shift
  if (tid == 0) {
    sum = 0;
    block[BLOCK_ID_X] = l_tmp[BLOCK-1];
    //      l_mask_sums[i] = tsum;
    //      // gridDim.x expressed in terms of the Geryon
  } else {
    sum = l_tmp[tid - 1];
  }
  if (gid < n) {
    output[gid] = sum;
  }
}

__kernel void k_add (
    __global unsigned int *plus,
    __global unsigned int *result,
    const int n) {

  int tid = THREAD_ID_X;
  int gid = GLOBAL_ID_X;
  int block = BLOCK_ID_X;
  __local unsigned int block_inc;
  if (tid == 0) {
    block_inc = plus[block];
  }
  __syncthreads();
  if (gid < n) {
    result[gid] += block_inc;
  }
}

