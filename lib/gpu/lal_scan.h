/*
 * lal_scan.h
 *
 *  Created on: Aug 17, 2020
 *      Author: vsevak
 */

#ifndef LIB_GPU_LAL_SCAN_H_
#define LIB_GPU_LAL_SCAN_H_

#include "lal_device.h"
#include "lal_precision.h"
#if defined(USE_OPENCL)
#include "scan_cl.h"
#elif defined(USE_CUDART)
const char *scan=0;
#else
#include "scan_cubin.h"
#endif

namespace LAMMPS_AL {

extern Device<PRECISION,ACC_PRECISION> global_device;

class Scan {
public:
  Scan();
  ~Scan() = default;
  void scan(UCL_D_Vec<unsigned int> &input,
      UCL_D_Vec<unsigned int> &output, const int n);
private:
  void compile_kernels();
  UCL_Device *gpu;
  UCL_Kernel k_scan, k_sum;
  //UCL_D_Vec<unsigned int> block_res;

  static const int block_size;
};

// This value must be consistent with BLOCK in lal_scan.cu
// BLOCK = Scan::block_size
const int Scan::block_size = 256;

Scan::Scan() {
  Device<PRECISION,ACC_PRECISION> *dev = &global_device;
  gpu = dev->gpu;
  //block_res.alloc(8, *gpu);
  compile_kernels();
}

void Scan::compile_kernels() {
  Device<PRECISION,ACC_PRECISION> *d = &global_device;
  UCL_Program dev_program(*(d->gpu));
  dev_program.load_string(::scan, d->compile_string().c_str());
  k_scan.set_function(dev_program,"k_scan");
  k_sum.set_function(dev_program, "k_add");
}

void Scan::scan(UCL_D_Vec<unsigned int> &input,
    UCL_D_Vec<unsigned int> &output, const int n) {

  int t = static_cast<int>(std::ceil(static_cast<double>(n)/block_size));
  UCL_D_Vec<unsigned int> block_res;
  block_res.alloc( std::max(t, 8), *gpu);

  k_scan.set_size(t, block_size);
  k_scan.run(&input, &output, &n, &block_res);

  if (t > 1) {
    UCL_D_Vec<unsigned int> block_res_out;
    block_res_out.alloc( std::max(t, 8), *gpu);
    scan(block_res, block_res_out, t);
    k_sum.set_size(t, block_size);
    k_sum.run(&block_res, &output, &n);
  }
}

}




#endif /* LIB_GPU_LAL_SCAN_H_ */
