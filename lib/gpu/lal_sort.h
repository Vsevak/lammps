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

template<typename ktyp, typename vtyp>
class RadixSort {
public:
  RadixSort(UCL_D_Vec<ktyp> *k, UCL_D_Vec<vtyp> *v);
  ~RadixSort() {};
  void Sort(int n);
  void Sort(UCL_D_Vec<ktyp> *k, UCL_D_Vec<vtyp> *v, int n);

private:
  void sum_scan(){};
  void compile_kernels();

  UCL_Kernel k_local, k_global;
  UCL_D_Vec<ktyp> *key;
  UCL_D_Vec<vtyp> *value;

  UCL_D_Vec<ktyp> out;
  UCL_D_Vec<ktyp> prefix;
  UCL_D_Vec<ktyp> block;
  UCL_D_Vec<ktyp> scan_block;
};


template<class ktyp, class vtyp>
RadixSort<ktyp, vtyp>::RadixSort(UCL_D_Vec<ktyp> *k, UCL_D_Vec<vtyp> *v)
    : key(k), value(v) {

  Device<PRECISION,ACC_PRECISION> *dev = &global_device;
  UCL_Device *d = dev->gpu;
  out.alloc(16, *d);
  prefix.alloc(16, *d);
  block.alloc(16, *d);
  scan_block.alloc(8, *d);
  compile_kernels();
}

template<class ktyp, class vtyp>
void RadixSort<ktyp, vtyp>::Sort(
    UCL_D_Vec<ktyp> *key, UCL_D_Vec<vtyp> *value, const int n) {

  Device<PRECISION,ACC_PRECISION> *d = &global_device;
  const int block_size = 256;
  int t = static_cast<int>(std::ceil(static_cast<double>(n)/block_size));
  if (key->cols() < n) {
    printf("\nWarning: RadixSort key is too short.\n");
    key->resize(n);
  }
  if (value->cols() < n) {
    printf("\nWarning: RadixSort value is too short.\n");
    value->resize(n);
  }
  out.resize_ib(n);
  prefix.resize_ib(n);
  prefix.zero(n);
  block.resize_ib(4*t);
  block.zero(4*t);
  scan_block.resize_ib(4*t);
  scan_block.zero(4*t);
  for (int b = 0; b <= 32; b+=2) {
    k_local.set_size(t, block_size);
    k_local.run(key, value, &out, &n, &prefix, &block, &b);

    sum_scan();

    k_global.set_size(t, block_size);
    k_global.run(key, value, &out, &n, &prefix, &block, &b);

    printf("\n\n========OUT%d===============\n", b);
    ucl_print(out, 256);

    printf("\n\n========OUT===============\n");
  }
}

template<class ktyp, class vtyp>
void RadixSort<ktyp, vtyp>::Sort(const int n) {
  Sort(key, value, n);
}

template<class ktyp, class vtyp>
void RadixSort<ktyp, vtyp>::compile_kernels() {
  Device<PRECISION,ACC_PRECISION> *d = &global_device;
  UCL_Program dev_program(*(d->gpu));

  int success = dev_program.load_string(sort, d->compile_string().c_str());
//  if(!success) {
//    printf("RadixSort program was not loaded.\n");
//    return;
//  }
  k_local.set_function(dev_program,"k_local");
  k_global.set_function(dev_program, "k_global");
}

}

#endif /* LAL_SORT_H_ */
