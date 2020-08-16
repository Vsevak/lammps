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

__kernel void k_local(
    __global unsigned *k,
    __global int *v,
    __global unsigned *k_out,
    __global unsigned *v_out,
    const int n,
    __global unsigned *prefix,
    __global unsigned *block,
    const int b) {

  __local unsigned int l_input[BLOCK];
  __local          int l_input_value[BLOCK];
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
  unsigned int input_key = l_input[tid];
  // Read value from global to register
  int input_value;
  if (gid < n) {
    input_value = v[gid];
  } else {
    input_value = 0;
  }
  // Get two LSB
  unsigned int get_two_bits = (input_key >> b ) & 3;

  // 4-way radix
  for(unsigned int i = 0; i < 4; ++i) {
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
    for (unsigned int step = 0; step < total; ++step) {
      target = tid - (1 << step);
      if (target >= 0) {
        sum = l_mask[tid] + l_mask[target];
      } else {
        sum = l_mask[tid];
      }
      __syncthreads();
      l_mask[tid] = sum;
      __syncthreads();
    }

    // Shift
    unsigned int buffer = l_mask[tid];
    __syncthreads();
    l_mask[tid + 1] = buffer;
    __syncthreads();
    if (tid == 0) {
      l_mask[0] = 0;
      unsigned int tsum = l_mask[BLOCK];
      l_mask_sums[i] = tsum;
      int grid = GLOBAL_SIZE_X;
      grid /= BLOCK_SIZE_X;
      block[i * grid  + BLOCK_ID_X]  = tsum;
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
    int pos = merged + l_scan_mask_sums[get_two_bits];
    __syncthreads();
    l_input[pos] = input_key;
    l_input_value[pos] = input_value;
    l_merged[pos] = merged;
    __syncthreads();

    // Global output
    prefix[gid] = l_merged[tid];
    k_out[gid] = l_input[tid];
    v_out[gid] = l_input_value[tid];
  }
}

__kernel void k_global(
    __global unsigned int *key_out,
    __global int *restrict value_out,
    __global unsigned int *key_in,
    __global int *restrict value_in,
    const int n,
    __global unsigned int *prefix,
    __global unsigned int *scan_block,
    const int b) {

  int gid = GLOBAL_ID_X;
  int grid = GLOBAL_SIZE_X; grid /= BLOCK_SIZE_X;

  if (gid < n) {
    unsigned int k_in = key_in[gid];
    unsigned int get_two_bits = (k_in >> b) & 3;
    int v_in = value_in[gid];
    unsigned int pref = prefix[gid];
    unsigned int pos = scan_block[get_two_bits * grid + BLOCK_ID_X] + pref;
    __syncthreads();
    key_out[pos] = k_in;
    value_out[pos] = v_in;
  }

}

