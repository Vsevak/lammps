/*
 * lal_sort.cu
 *
 *  Created on: Aug 14, 2020
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

#define BLOCK 256

__kernel void k_local(__global unsigned *restrict k,
    __global int *restrict v,
    __global unsigned *restrict out,
    const int n,
    __global unsigned *restrict prefix,
    __global unsigned *restrict block,
    const int b) {

  __local unsigned int l_input[BLOCK];
  __local unsigned int l_mask[BLOCK+1];
  __local unsigned int l_merged[BLOCK];
  __local unsigned int l_mask_sums[4];
  __local unsigned int l_scan_mask_sums[4];

  int tid = THREAD_ID_X;
  int gid = GLOBAL_ID_X;
  // Read key from global to shared
  if (gid < n) {
    l_input[tid] = k[gid];
  } else {
    l_input[tid] = 0;
  }
  __syncthreads();
  unsigned int thread_input = l_input[tid];
  // Get two LSB
  unsigned int get_two_bits = (thread_input >> b ) & 3;

  // 4-way radix
  for(int i=0; i<4; ++i) {
    l_mask[tid] = 0;
    if (tid == 0) {
      l_mask[BLOCK] = 0;
    }
    __syncthreads();

    bool out_eq_in = false;
    if (gid < n) {
      out_eq_in = (get_two_bits == i);
      l_mask[tid] = out_eq_in;
    }
    __syncthreads();

    int target = 0;
    unsigned int sum = 0;
    int total = (int) log2f(BLOCK);
    // Prefix-sum masks
    for (int step = 0; step < total; ++step) {
      target = tid - (1 << step);
      sum = l_mask[tid] + ((target >= 0) ? l_mask[target] : 0);
      __syncthreads();
      l_mask[tid] = sum;
      __syncthreads();
    }

    // Shift
    unsigned int buffer = l_mask[tid];
    __syncthreads();
    l_mask[tid + 1] = buffer;
    __syncthreads();
    if (tid ==0) {
      l_mask[0] = 0;
      sum = l_mask[BLOCK];
      l_mask_sums[i] = sum;
      int grid = GLOBAL_SIZE_X;
      grid /= BLOCK_SIZE_X;
      block[i * grid  + BLOCK_ID_X]  = sum;
    }
    __syncthreads();

    if (out_eq_in && (gid < n)) {
      l_merged[tid] = l_mask[tid];
    }
    __syncthreads();
  }

  // Serial scan resulting masks
  if (tid == 0) {
    unsigned int csum = 0;
    for (int i = 0; i < 4; ++ i){
      l_scan_mask_sums[i] = csum;
      csum += l_mask_sums[i];
    }
  }
  __syncthreads();

  if (gid < n) {
    unsigned int merged = l_merged[tid];
    unsigned int pos = merged + l_scan_mask_sums[get_two_bits];
    __syncthreads();
    l_input[pos] = thread_input;
    l_merged[pos] = merged;
    __syncthreads();

    // Global output
    prefix[gid] = l_merged[tid];
    out[gid] = l_input[tid];
  }
}

__kernel void k_global(__global int *restrict x, const int n){
  int tid = BLOCK_ID_X*BLOCK_SIZE_X + THREAD_ID_X;
  if (tid < n){
    //x[tid] *= 100;
  }
}

