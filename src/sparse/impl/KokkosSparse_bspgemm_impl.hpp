/*
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 3.0
//       Copyright (2020) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY NTESS "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NTESS OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Siva Rajamanickam (srajama@sandia.gov)
//
// ************************************************************************
//@HEADER
*/

#ifndef _KOKKOSBSPGEMMIMPL_HPP
#define _KOKKOSBSPGEMMIMPL_HPP

#include <KokkosSparse_spgemm_impl.hpp>

namespace KokkosSparse{

namespace Impl{



template <typename HandleType,
  typename a_row_view_t_, typename a_lno_nnz_view_t_, typename a_scalar_nnz_view_t_,
  typename b_lno_row_view_t_, typename b_lno_nnz_view_t_, typename b_scalar_nnz_view_t_  >
class KokkosBSPGEMM:
  public KokkosSPGEMM<HandleType,
    a_row_view_t_, a_lno_nnz_view_t_, a_scalar_nnz_view_t_,
    b_lno_row_view_t_, b_lno_nnz_view_t_, b_scalar_nnz_view_t_ >
{
  using Base = KokkosSparse::Impl::KokkosSPGEMM<HandleType,
    a_row_view_t_, a_lno_nnz_view_t_, a_scalar_nnz_view_t_,
    b_lno_row_view_t_, b_lno_nnz_view_t_, b_scalar_nnz_view_t_ >;

  #define DECL_BASE_TYPE(type) using type = typename Base::type;

  DECL_BASE_TYPE(nnz_lno_t); DECL_BASE_TYPE(size_type); DECL_BASE_TYPE(scalar_t);
  DECL_BASE_TYPE(const_a_lno_row_view_t); DECL_BASE_TYPE(const_a_lno_nnz_view_t); DECL_BASE_TYPE(const_a_scalar_nnz_view_t);
  DECL_BASE_TYPE(const_b_lno_row_view_t); DECL_BASE_TYPE(const_b_lno_nnz_view_t); DECL_BASE_TYPE(const_b_scalar_nnz_view_t);
  DECL_BASE_TYPE(row_lno_persistent_work_view_t);
  DECL_BASE_TYPE(MultiCoreTag); DECL_BASE_TYPE(MultiCoreTag2); DECL_BASE_TYPE(MultiCoreTag4);
  DECL_BASE_TYPE(GPUTag); DECL_BASE_TYPE(GPUTag4); DECL_BASE_TYPE(GPUTag6);
  DECL_BASE_TYPE(team_member_t);
  DECL_BASE_TYPE(MyExecSpace);
  DECL_BASE_TYPE(MyTempMemorySpace);

protected:
  nnz_lno_t blockDim;

public:

private:
  /**
   * \brief Numeric phase with speed method
   */
  template <typename c_row_view_t, typename c_lno_nnz_view_t, typename c_scalar_nnz_view_t>
  void KokkosBSPGEMM_numeric_speed(
      c_row_view_t rowmapC_,
      c_lno_nnz_view_t entriesC_,
      c_scalar_nnz_view_t valuesC_,
      KokkosKernels::Impl::ExecSpaceType my_exec_space);

public:
private:
  /**
   * \brief Numeric phase with speed method
   */
public:
  //////////////////////////////////////////////////////////////////////////
  /////BELOW CODE IS TO for kkmem SPGEMM
  ////DECL IS AT _kkmem.hpp
  //////////////////////////////////////////////////////////////////////////
  template <typename a_row_view_t, typename a_nnz_view_t, typename a_scalar_view_t,
            typename b_row_view_t, typename b_nnz_view_t, typename b_scalar_view_t,
            typename c_row_view_t, typename c_nnz_view_t, typename c_scalar_view_t,
            typename pool_memory_type>
  struct PortableNumericCHASH;
private:
  //KKMEM only difference is work memory does not use output memory for 2nd level accumulator.
  template <typename c_row_view_t, typename c_lno_nnz_view_t, typename c_scalar_nnz_view_t>
  void KokkosBSPGEMM_numeric_hash2(
        c_row_view_t rowmapC_,
        c_lno_nnz_view_t entriesC_,
        c_scalar_nnz_view_t valuesC_,
        KokkosKernels::Impl::ExecSpaceType my_exec_space);

  template <typename c_row_view_t, typename c_lno_nnz_view_t, typename c_scalar_nnz_view_t>
  void KokkosBSPGEMM_numeric_hash(
        c_row_view_t rowmapC_,
        c_lno_nnz_view_t entriesC_,
        c_scalar_nnz_view_t valuesC_,
        KokkosKernels::Impl::ExecSpaceType my_exec_space);

  template <typename c_row_view_t, typename c_lno_nnz_view_t, typename c_scalar_nnz_view_t>
    void KokkosBSPGEMM_numeric_outer(
          c_row_view_t &rowmapC_,
          c_lno_nnz_view_t &entriesC_,
          c_scalar_nnz_view_t &valuesC_,
          KokkosKernels::Impl::ExecSpaceType my_exec_space);
  //////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////

public:

  //////////////////////////////////////////////////////////////////////////
  /////BELOW CODE IS for public symbolic and numeric functions
  ////DECL IS AT _def.hpp
  //////////////////////////////////////////////////////////////////////////
  template <typename c_row_view_t, typename c_lno_nnz_view_t, typename c_scalar_nnz_view_t>
  void KokkosBSPGEMM_numeric(c_row_view_t &rowmapC_, c_lno_nnz_view_t &entriesC_, c_scalar_nnz_view_t &valuesC_);

  KokkosBSPGEMM(
      HandleType *handle_,
      nnz_lno_t m_,
      nnz_lno_t n_,
      nnz_lno_t k_,
      nnz_lno_t blockDim_,
      const_a_lno_row_view_t row_mapA_,
      const_a_lno_nnz_view_t entriesA_,
      bool transposeA_,
      const_b_lno_row_view_t row_mapB_,
      const_b_lno_nnz_view_t entriesB_,
      bool transposeB_): Base(handle_, m_, n_, k_,
          row_mapA_, entriesA_, transposeA_,
          row_mapB_, entriesB_, transposeB_),
          blockDim(blockDim_)
  {}

  KokkosBSPGEMM(
      HandleType *handle_,
      nnz_lno_t m_,
      nnz_lno_t n_,
      nnz_lno_t k_,
      nnz_lno_t blockDim_,
      const_a_lno_row_view_t row_mapA_,
      const_a_lno_nnz_view_t entriesA_,
      const_a_scalar_nnz_view_t valsA_,
      bool transposeA_,
      const_b_lno_row_view_t row_mapB_,
      const_b_lno_nnz_view_t entriesB_,
      const_b_scalar_nnz_view_t valsB_,
      bool transposeB_): Base(handle_, m_, n_, k_,
          row_mapA_, entriesA_, valsA_, transposeA_,
          row_mapB_, entriesB_, valsB_, transposeB_),
          blockDim(blockDim_)
  {}
};

}
}
// #include "KokkosSparse_spgemm_imp_outer.hpp"
// #include "KokkosSparse_spgemm_impl_memaccess.hpp"

#include "KokkosSparse_bspgemm_impl_kkmem.hpp"

// #include "KokkosSparse_spgemm_impl_speed.hpp"
// #include "KokkosSparse_spgemm_impl_compression.hpp"

#include "KokkosSparse_bspgemm_impl_def.hpp"

// #include "KokkosSparse_spgemm_impl_symbolic.hpp"
// #include "KokkosSparse_spgemm_impl_triangle.hpp"
#endif
