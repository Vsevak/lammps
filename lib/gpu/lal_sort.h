/*
 * lal_sort.h
 *
 *  Created on: Sep 8, 2020
 *      Author: vsevak
 */

#ifndef LIB_GPU_LAL_SORT_H_
#define LIB_GPU_LAL_SORT_H_

#if defined(USE_OPENCL)
#include "geryon/ocl_timer.h"
#include "geryon/ocl_mat.h"
using namespace ucl_opencl;
#elif defined(USE_CUDART)
#include "geryon/nvc_timer.h"
#include "geryon/nvc_mat.h"
using namespace ucl_cudart;
#elif defined(USE_HIP)
#include "geryon/hip_timer.h"
#include "geryon/hip_mat.h"
using namespace ucl_hip;
#else
#include "geryon/nvd_timer.h"
#include "geryon/nvd_mat.h"
using namespace ucl_cudadr;
#endif

namespace LAMMPS_AL {

class Sort {
public:
  virtual ~Sort() = default;
  virtual bool alloc(const int n) = 0;
  virtual void clear() = 0;
  virtual bool resize(const int n) { clear(); return alloc(n); }
  virtual void sort(UCL_D_Vec<unsigned int> &k,
      UCL_D_Vec<int> &v, const int n) = 0;
};

} /* namespace LAMMPS_AL */

#endif /* LIB_GPU_LAL_SORT_H_ */
