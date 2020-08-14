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

#define TSort RadixSort<ktyp,vtyp,cont>;

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

  UCL_D_Vec<ktyp> *key;
  UCL_D_Vec<vtyp> *value;
  UCL_Kernel k_local, k_global;
};


template<class ktyp, class vtyp>
RadixSort<ktyp, vtyp>::RadixSort(UCL_D_Vec<ktyp> *k, UCL_D_Vec<vtyp> *v)
    : key(k), value(v) {
  compile_kernels();
}

template<class ktyp, class vtyp>
void RadixSort<ktyp, vtyp>::Sort(
    UCL_D_Vec<ktyp> *key, UCL_D_Vec<vtyp> *value, const int n) {

  Device<PRECISION,ACC_PRECISION> *d = &global_device;
  int block = d->pair_block_size();
  int t = static_cast<int>(std::ceil(static_cast<double>(n)/block));
  k_local.set_size(t, block);
  k_local.run(key, &n);
  k_global.set_size(t, block);
  k_global.run(key, &n);
}

template<class ktyp, class vtyp>
void RadixSort<ktyp, vtyp>::Sort(const int n){
  Sort(key, value, n);
}

template<class ktyp, class vtyp>
void RadixSort<ktyp, vtyp>::compile_kernels() {
  Device<PRECISION,ACC_PRECISION> *d = &global_device;
  UCL_Program dev_program(*(d->gpu));

  int success = dev_program.load_string(sort, d->compile_string().c_str());
  k_local.set_function(dev_program,"k_local");
  k_global.set_function(dev_program, "k_global");

}

}

#endif /* LAL_SORT_H_ */
