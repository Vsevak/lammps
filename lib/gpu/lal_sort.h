/*
 * sort.h
 *
 *  Created on: Aug 14, 2020
 *      Author: vsevak
 */

#ifndef LAL_SORT_H_
#define LAL_SORT_H_

#include "lal_device.h"
#include "lal_precision.h"
#if defined(USE_OPENCL)
#include "sort_cl.h"
#elif defined(USE_CUDART)
const char *sort=0;
#else
#include "sort_cubin.h"
#endif

namespace LAMMPS_AL {

extern Device<PRECISION,ACC_PRECISION> global_device;

#define TSort RadixSort<ktyp,vtyp>;
#define PRINT 0

template<typename ktyp, typename vtyp>
class RadixSort {
public:
  RadixSort(UCL_D_Vec<ktyp> *k, UCL_D_Vec<vtyp> *v);
  ~RadixSort() {};
  void sort(const int n);
  void sort(UCL_D_Vec<ktyp> *k, UCL_D_Vec<vtyp> *v, const int n);

private:
  void sum_scan(const int n);
  void compile_kernels();

  UCL_Kernel k_local, k_global;
  UCL_D_Vec<ktyp> *key;
  UCL_D_Vec<vtyp> *value;

  UCL_D_Vec<ktyp> k_out;
  UCL_D_Vec<vtyp> v_out;
  UCL_D_Vec<ktyp> prefix;
  UCL_D_Vec<ktyp> block;
  UCL_D_Vec<ktyp> scan_block;

  static const int block_size;
};

template<class ktyp, class vtyp>
const int RadixSort<ktyp, vtyp>::block_size = 256;


template<class ktyp, class vtyp>
RadixSort<ktyp, vtyp>::RadixSort(UCL_D_Vec<ktyp> *k, UCL_D_Vec<vtyp> *v)
    : key(k), value(v) {

  Device<PRECISION,ACC_PRECISION> *dev = &global_device;
  UCL_Device *d = dev->gpu;
  k_out.alloc(16, *d);
  v_out.alloc(16, *d);
  prefix.alloc(16, *d);
  block.alloc(16, *d);
  scan_block.alloc(8, *d);
  compile_kernels();
}

template<class ktyp, class vtyp>
void RadixSort<ktyp, vtyp>::sort(
    UCL_D_Vec<ktyp> *key, UCL_D_Vec<vtyp> *value, const int n) {

  Device<PRECISION,ACC_PRECISION> *d = &global_device;
  int t = static_cast<int>(std::ceil(static_cast<double>(n)/block_size));
  if (key->cols() < n) {
    printf("\nWarning: RadixSort key is too short.\n");
    key->resize(n);
  }
  if (value->cols() < n) {
    printf("\nWarning: RadixSort value is too short.\n");
    value->resize(n);
  }
  k_out.resize_ib(n);
  v_out.resize_ib(n);
  prefix.resize_ib(n);
  prefix.zero(n);
  block.resize_ib(4*t);
  block.zero(4*t);
  scan_block.resize_ib(4*t);
  scan_block.zero(4*t);
  CUDPPHandle scan_plan;
  CUDPPConfiguration scan_config;
  scan_config.op = CUDPP_ADD;
  scan_config.datatype = CUDPP_UINT;
  scan_config.algorithm = CUDPP_SCAN;
  scan_config.options = CUDPP_OPTION_EXCLUSIVE;
  CUDPPResult result = cudppPlan(&scan_plan, scan_config, 4*t, 1, 0);
  for (int b = 0; b <= 30; b+=2) {
    k_local.set_size(t, block_size);
    k_local.run(key, value, &k_out, &v_out, &n, &prefix, &block, &b);
    if (PRINT && b==0) {
      printf("\n\n========KEY%d===============\n", b);
      ucl_print(k_out, block_size);
      printf("\n==========PREFIX/MERGED==================\n");
      ucl_print(prefix, block_size);
      printf("\n==========BLOCK==================\n");
      ucl_print(block, 4*t);
      printf("\n========OUT===============\n");
    }
    //sum_scan(n);
    CUDPPResult result = cudppScan(scan_plan, (unsigned*) scan_block.begin(),
        (unsigned*) block.begin(), 4*t);
    if (PRINT && b==0) {
      printf("\n========SCAN BLOCK===============\n");
      ucl_print(scan_block, 4*t);
      printf("\n\n========OUT===============\n");
    }
    k_global.set_size(t, block_size);
    k_global.run(key, value, &k_out, &v_out, &n, &prefix, &scan_block, &b);
  }
}

template<class ktyp, class vtyp>
void RadixSort<ktyp, vtyp>::sort(const int n) {
  sort(key, value, n);
}

template<class ktyp, class vtyp>
void RadixSort<ktyp, vtyp>::sum_scan(const int n) {
  key->zero(n);
  int scan_block_sz = block_size / 2;
  int grid = scan_block_sz * 2;
  int t = static_cast<int>(std::ceil(static_cast<double>(n)/grid));
  scan_block.resize_ib(4*t);
  scan_block.zero(4*t);
}

template<class ktyp, class vtyp>
void RadixSort<ktyp, vtyp>::compile_kernels() {
  Device<PRECISION,ACC_PRECISION> *d = &global_device;
  UCL_Program dev_program(*(d->gpu));

  int success = dev_program.load_string(::sort, d->compile_string().c_str());
//  if(!success) {
//    printf("RadixSort program was not loaded.\n");
//    return;
//  }
  k_local.set_function(dev_program,"k_local");
  k_global.set_function(dev_program, "k_global");
}

}

#endif /* LAL_SORT_H_ */
