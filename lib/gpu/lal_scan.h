/*
 * lal_scan.h
 *
 *  Created on: Aug 17, 2020
 *      Author: Vsevolod Nikolskiy
 *       Email: thevsevak@gmail.com
 */

#ifndef LIB_GPU_LAL_SCAN_H_
#define LIB_GPU_LAL_SCAN_H_

#if defined(USE_OPENCL)
#include "geryon/ocl_timer.h"
#include "geryon/ocl_mat.h"
#include "geryon/ocl_kernel.h"
using namespace ucl_opencl;
#elif defined(USE_CUDART)
#include "geryon/nvc_timer.h"
#include "geryon/nvc_mat.h"
#include "geryon/nvc_kernel.h"
using namespace ucl_cudart;
#elif defined(USE_HIP)
#include "geryon/hip_timer.h"
#include "geryon/hip_mat.h"
#include "geryon/hip_kernel.h"
using namespace ucl_hip;
#else
#include "geryon/nvd_timer.h"
#include "geryon/nvd_mat.h"
#include "geryon/nvd_kernel.h"
using namespace ucl_cudadr;
#endif

#include <string>
#include <list>
#include <vector>
#include "lal_precision.h"

namespace LAMMPS_AL {

class Scan {
public:
  Scan(UCL_Device &d, std::string);
  ~Scan() = default;
  Scan(const Scan&) = delete;
  Scan& operator=(const Scan&) = delete;
  Scan(Scan &&) noexcept = default;
  Scan& operator=(Scan &&) noexcept = default;

  void clear();
  void scan(UCL_D_Vec<unsigned int> &input,
      UCL_D_Vec<unsigned int> &output, const int n, const int iter=0);
private:
  void compile_kernels();
  UCL_Device &gpu;
  UCL_Kernel k_scan, k_sum;
  std::string ocl_param;
  std::vector<UCL_D_Vec<unsigned int>> block_res, block_res_out;

  static const int block_size;
};

}




#endif /* LIB_GPU_LAL_SCAN_H_ */
