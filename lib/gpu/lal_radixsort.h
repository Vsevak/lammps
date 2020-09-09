/*
 * lal_sort.h
 *
 *  Created on: Aug 14, 2020
 *      Author: Vsevolod Nikolskiy
 *       Email: thevsevak@gmail.com
 */

#ifndef LAL_SORT_H_
#define LAL_SORT_H_

#include "lal_sort.h"
#include "lal_scan.h"

namespace LAMMPS_AL {

#define RADIX_PRINT 0

class RadixSort : public Sort {
public:
  RadixSort(UCL_Device &gpu, std::string);
  ~RadixSort() = default;
  RadixSort(const RadixSort&) = delete;
  RadixSort& operator=(const RadixSort&) = delete;
  RadixSort(RadixSort&&) noexcept = default;
  RadixSort& operator=(RadixSort&&) noexcept = default;

  bool alloc(const int n) override;
  bool resize(const int n) override;
  void clear() override;
  void sort(UCL_D_Vec<unsigned int> &k,
      UCL_D_Vec<int> &v, const int n) override;
private:
  void compile_kernels();

  UCL_Device &gpu;
  UCL_Kernel k_local, k_global_scatter, k_check;
  std::string ocl_param;

  Scan scanner;
  UCL_D_Vec<unsigned int> k_out{};
  UCL_D_Vec<int>          v_out{};
  UCL_D_Vec<unsigned int> prefix{};
  UCL_D_Vec<unsigned int> block{};
  UCL_D_Vec<unsigned int> scan_block{};

  bool _allocated{false};
  // This value must be consistent with BLOCK in lal_sort.cu
  static const int block_size{256};
};

}

#endif /* LAL_SORT_H_ */
