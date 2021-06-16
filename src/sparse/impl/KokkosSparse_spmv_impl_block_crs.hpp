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

#ifndef KOKKOSKERNELS_KOKKOSSPARSE_SPMV_IMPL_BLOCK_CRS_HPP
#define KOKKOSKERNELS_KOKKOSSPARSE_SPMV_IMPL_BLOCK_CRS_HPP

#include "Kokkos_Core.hpp"

#include "KokkosBlas.hpp"
#include "KokkosKernels_default_types.hpp"
#include "KokkosSparse_BlockCrsMatrix.hpp"

#include "KokkosKernels_helpers.hpp"
#include "KokkosKernels_Controls.hpp"
#include "KokkosKernels_Utils.hpp"
#include "KokkosKernels_ExecSpaceUtils.hpp"

namespace KokkosSparse {
namespace Impl {

//////////////////////////////////////////////////////////

template <class AMatrix, class XVector, class YVector>
bool verifyArguments(const char mode[], const AMatrix &A, const XVector &x,
                     const YVector &y) {
  // Make sure that both x and y have the same rank.
  static_assert(
      static_cast<int>(XVector::rank) == static_cast<int>(YVector::rank),
      "KokkosSparse::spmv: Vector ranks do not match.");
  // Make sure that y is non-const.
  static_assert(std::is_same<typename YVector::value_type,
                             typename YVector::non_const_value_type>::value,
                "KokkosSparse::spmv: Output Vector must be non-const.");

  // Check compatibility of dimensions at run time.
  if ((mode[0] == KokkosSparse::NoTranspose[0]) ||
      (mode[0] == KokkosSparse::Conjugate[0])) {
    if ((x.extent(1) != y.extent(1)) ||
        (static_cast<size_t>(A.numCols()) > static_cast<size_t>(x.extent(0))) ||
        (static_cast<size_t>(A.numRows()) > static_cast<size_t>(y.extent(0)))) {
      std::ostringstream os;
      os << "KokkosSparse::spmv: Dimensions do not match: "
         << ", A: " << A.numRows() << " x " << A.numCols()
         << ", x: " << x.extent(0) << " x " << x.extent(1)
         << ", y: " << y.extent(0) << " x " << y.extent(1);
      Kokkos::Impl::throw_runtime_exception(os.str());
    }
  } else {
    if ((x.extent(1) != y.extent(1)) ||
        (static_cast<size_t>(A.numCols()) > static_cast<size_t>(y.extent(0))) ||
        (static_cast<size_t>(A.numRows()) > static_cast<size_t>(x.extent(0)))) {
      std::ostringstream os;
      os << "KokkosSparse::spmv: Dimensions do not match (transpose): "
         << ", A: " << A.numRows() << " x " << A.numCols()
         << ", x: " << x.extent(0) << " x " << x.extent(1)
         << ", y: " << y.extent(0) << " x " << y.extent(1);
      Kokkos::Impl::throw_runtime_exception(os.str());
    }
  }

  return true;
}

//////////////////////////////////////////////////////////

template <class AMatrix, class XVector, class YVector>
struct BSPMV_Functor {
  typedef typename AMatrix::execution_space execution_space;
  typedef typename AMatrix::non_const_ordinal_type ordinal_type;
  typedef typename AMatrix::non_const_value_type value_type;
  typedef typename Kokkos::TeamPolicy<execution_space> team_policy;
  typedef typename team_policy::member_type team_member;
  typedef Kokkos::Details::ArithTraits<value_type> ATV;

  const value_type alpha;
  AMatrix m_A;
  XVector m_x;
  YVector m_y;
  const ordinal_type block_size;

  const ordinal_type blocks_per_team;

  bool conjugate = false;

  BSPMV_Functor(const value_type alpha_, const AMatrix m_A_, const XVector m_x_,
                const YVector m_y_, const int blocks_per_team_, bool conj_)
      : alpha(alpha_),
        m_A(m_A_),
        m_x(m_x_),
        m_y(m_y_),
        block_size(m_A_.blockDim()),
        blocks_per_team(blocks_per_team_),
        conjugate(conj_)
  {
    static_assert(static_cast<int>(XVector::rank) == 1,
                  "XVector must be a rank 1 View.");
    static_assert(static_cast<int>(YVector::rank) == 1,
                  "YVector must be a rank 1 View.");
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const ordinal_type iBlock) const {
    //
    if (iBlock >= m_A.numRows()) {
      return;
    }
    //
    const auto jbeg       = m_A.graph.row_map[iBlock];
    const auto jend       = m_A.graph.row_map[iBlock + 1];
    const auto block_size_2 = block_size * block_size;
    //
    auto yvec = &m_y[iBlock * block_size];
    //
    for (auto jb = jbeg; jb < jend; ++jb) {
      const auto col_block = m_A.graph.entries[jb];
      const auto xval_ptr  = &m_x[block_size * col_block];
      const auto Aval_ptr  = &m_A.values[jb * block_size_2];
      for (ordinal_type kr = 0; kr < block_size; ++kr) {
        for (ordinal_type ic = 0; ic < block_size; ++ic) {
          const auto aval = conjugate ? ATV::conj(Aval_ptr[ic + kr * block_size])
              : Aval_ptr[ic + kr * block_size];
          yvec[kr] += alpha * aval * xval_ptr[ic];
        }
      }
    }
    //
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const team_member &dev) const {
    using y_value_type = typename YVector::non_const_value_type;
    //
    // UH -- To be completed
    //
    /*
    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(dev, 0, rows_per_team),
        [&](const ordinal_type &loop) {
          const ordinal_type iRow =
              static_cast<ordinal_type>(dev.league_rank()) * rows_per_team +
              loop;
          if (iRow >= m_A.numRows()) {
            return;
          }
          const KokkosSparse::SparseRowViewConst<AMatrix> row =
              m_A.rowConst(iRow);
          const auto row_length = static_cast<ordinal_type>(row.length);
          y_value_type sum              = 0;

          Kokkos::parallel_reduce(
              Kokkos::ThreadVectorRange(dev, row_length),
              [&](const ordinal_type &iEntry, y_value_type &lsum) {
                const value_type val = conjugate ? ATV::conj(row.value(iEntry))
                                                 : row.value(iEntry);
                lsum += val * m_x(row.colidx(iEntry));
              },
              sum);

          Kokkos::single(Kokkos::PerThread(dev), [&]() {
            sum *= alpha;

            if (dobeta == 0) {
              m_y(iRow) = sum;
            } else {
              m_y(iRow) = beta * m_y(iRow) + sum;
            }
          });
        });
        */
  }
};

/***********************************/
//
//  This needs to be generalized
//
constexpr size_t bmax = 12;

using Scalar  = default_scalar;
using Ordinal = default_lno_t;
using Offset  = default_size_type;
using Layout  = default_layout;

using device_type = typename Kokkos::Device<
    Kokkos::DefaultExecutionSpace,
    typename Kokkos::DefaultExecutionSpace::memory_space>;

using crs_matrix_t_ =
    typename KokkosSparse::CrsMatrix<Scalar, Ordinal, device_type, void,
                                     Offset>;

using values_type = typename crs_matrix_t_::values_type;
/***********************************/

template <int M>
inline void spmv_serial_gemv(const Scalar *Aval, const Ordinal lda,
                             const Scalar *x_ptr,
                             std::array<Scalar, Impl::bmax> &y) {
  for (Ordinal ic = 0; ic < M; ++ic) {
    const auto xvalue = x_ptr[ic];
    for (Ordinal kr = 0; kr < M; ++kr) {
      y[kr] += Aval[ic + kr * lda] * xvalue;
    }
  }
}

//
// Explicit blockSize=N case
//
template <int N, class StaticGraph>
struct SerialSPMVHelper {
static inline void spmv(const Scalar alpha, const Scalar *Avalues,
                      const StaticGraph &Agraph, const Scalar *x, Scalar *y, const int blockSize) {

  const Ordinal numBlockRows = Agraph.numRows();
  std::array<double, Impl::bmax> tmp{0};
  const Ordinal N2 = N * N;
  for (Ordinal iblock = 0; iblock < numBlockRows; ++iblock) {
    const auto jbeg       = Agraph.row_map[iblock];
    const auto jend       = Agraph.row_map[iblock + 1];
    const auto num_blocks = jend - jbeg;
    tmp.fill(0);
    for (Ordinal jb = 0; jb < num_blocks; ++jb) {
      const auto col_block = Agraph.entries[jb + jbeg];
      const auto xval_ptr  = x + N * col_block;
      auto Aval_ptr        = Avalues + (jb + jbeg) * N2;
      spmv_serial_gemv<N>(Aval_ptr, N, xval_ptr, tmp);
    }
    //
    auto yvec = &y[iblock * N];
    for (Ordinal ii = 0; ii < N; ++ii) {
      yvec[ii] += alpha * tmp[ii];
    }
  }

}
};

//
// Special blockSize=1 case (optimized)
//
template <class StaticGraph>
struct SerialSPMVHelper<1, StaticGraph> {
static inline void spmv(const Scalar alpha, const Scalar *Avalues,
                        const StaticGraph &Agraph, const Scalar *x,
                        Scalar *y, const int blockSize) {

  const Ordinal numBlockRows = Agraph.numRows();
  for (Ordinal i = 0; i < numBlockRows; ++i) {
    const auto jbeg = Agraph.row_map[i];
    const auto jend = Agraph.row_map[i + 1];
    double tmp      = 0.0;
    for (Ordinal j = jbeg; j < jend; ++j) {
      const auto alpha_value1 = alpha * Avalues[j];
      const auto col_idx1     = Agraph.entries[j];
      const auto x_val1       = x[col_idx1];
      tmp += alpha_value1 * x_val1;
    }
    y[i] += tmp;
  }
}
};

//
// --- Basic approach for large block sizes
//
template <class StaticGraph>
struct SerialSPMVHelper<0, StaticGraph> {
static inline void spmv(const Scalar alpha, const Scalar *Avalues,
                        const StaticGraph &A_graph, const Scalar *x,
                        Scalar *y, const int blockSize) {

    const Ordinal numBlockRows = A_graph.numRows();
    const Ordinal blockSize_squared = blockSize * blockSize;
    auto yvec = &y[0];
    for (Ordinal iblock = 0; iblock < numBlockRows; ++iblock) {
      const auto jbeg       = A_graph.row_map[iblock];
      const auto jend       = A_graph.row_map[iblock + 1];
      //
      for (Ordinal jb = jbeg; jb < jend; ++jb) {
        const auto col_block = A_graph.entries[jb];
        const auto xval_ptr  = &x[0] + blockSize * col_block;
        const auto Aval_ptr  = Avalues + jb * blockSize_squared;
        for (Ordinal kr = 0; kr < blockSize; ++kr) {
          for (Ordinal ic = 0; ic < blockSize; ++ic) {
            auto q = Aval_ptr[ic + kr * blockSize];
            yvec[kr] += alpha * q * xval_ptr[ic];
          }
        }
      }
      //
      yvec = yvec + blockSize;
    }
}
};

//
// --- ETI helper
//
template <class StaticGraph>
inline void spmv_serial(const int blockSize, const Scalar alpha, const Scalar *Avalues,
                        const StaticGraph &Agraph, const Scalar *x, Scalar *y) {

#ifdef EXPAND
  #error Macro name collision on EXPAND()
#endif
#define EXPAND(N) case N: SerialSPMVHelper<N, StaticGraph>::spmv(alpha, Avalues, Agraph, &x[0], &y[0], blockSize); // this should be template function param or lambda...

  switch (blockSize) { // TODO: if (blockSize <= std::min<size_t>(12, Impl::bmax)) ??
    EXPAND(1)
    EXPAND(2)
    EXPAND(3)
    EXPAND(4)
    EXPAND(5)
    EXPAND(6)
    EXPAND(7)
    EXPAND(8)
    EXPAND(9)
    EXPAND(10)
    EXPAND(11)
    EXPAND(12)
    default: EXPAND(0)
  }
#undef EXPAND
}
/* ******************* */


#ifdef KOKKOS_ENABLE_OPENMP
template<typename AMatrix, typename XVector, typename YVector>
void bspmv_raw_openmp_no_transpose(typename YVector::const_value_type& s_a,
                                   AMatrix A, XVector x,
                                   YVector y)
{
  typedef typename YVector::non_const_value_type value_type;
  typedef typename AMatrix::ordinal_type         ordinal_type;
  typedef typename AMatrix::non_const_size_type  size_type;

  typename XVector::const_value_type* KOKKOS_RESTRICT x_ptr = x.data();
  typename YVector::non_const_value_type* KOKKOS_RESTRICT y_ptr = y.data();

  const typename AMatrix::value_type* KOKKOS_RESTRICT matrixCoeffs = A.values.data();
  const ordinal_type* KOKKOS_RESTRICT matrixCols     = A.graph.entries.data();
  const size_type* KOKKOS_RESTRICT matrixRowOffsets  = A.graph.row_map.data();
  const size_type* KOKKOS_RESTRICT threadStarts      = A.graph.row_block_offsets.data();

#if defined(KOKKOS_ENABLE_PROFILING)
  uint64_t kpID = 0;
  if(Kokkos::Profiling::profileLibraryLoaded()) {
    Kokkos::Profiling::beginParallelFor("KokkosSparse::spmv<RawOpenMP,NoTranspose>", 0, &kpID);
  }
#endif

  typename YVector::const_value_type zero = 0;
  #pragma omp parallel
  {
#if defined(KOKKOS_COMPILER_INTEL) && !defined(__clang__)
    __assume_aligned(x_ptr, 64);
    __assume_aligned(y_ptr, 64);
#endif

    const int myID    = omp_get_thread_num();
    const size_type myStart = threadStarts[myID];
    const size_type myEnd   = threadStarts[myID + 1];
    const auto blockSize = A.blockDim();
    const auto blockSize2 = blockSize * blockSize;

    for(size_type row = myStart; row < myEnd; ++row) {
      const size_type rowStart = matrixRowOffsets[row];
      const size_type rowEnd   = matrixRowOffsets[row + 1];
      //
      auto yvec = &y[row * blockSize];
      //
      for (Ordinal jblock = rowStart; jblock < rowEnd; ++jblock) {
        const auto col_block = A.graph.entries[jblock];
        const auto xval_ptr  = &x[blockSize * col_block];
        const auto Aval_ptr  = &matrixCoeffs[jblock * blockSize2];
        for (Ordinal ic = 0; ic < blockSize; ++ic) {
          const auto xvalue = xval_ptr[ic];
          for (Ordinal kr = 0; kr < blockSize; ++kr) {
            yvec[kr] += s_a * Aval_ptr[ic + kr * blockSize] * xvalue;
          }
        }
      }
      //
    }
  }

#if defined(KOKKOS_ENABLE_PROFILING)
  if(Kokkos::Profiling::profileLibraryLoaded()) {
    Kokkos::Profiling::endParallelFor(kpID);
  }
#endif

}
#endif


/* ******************* */


//
// spMatVec_no_transpose: version for CPU execution spaces (RangePolicy or
// trivial serial impl used)
//
template < class AT, class AO, class AD, class AM, class AS,
        class AlphaType, class XVector, class BetaType, class YVector,
        typename std::enable_if<!KokkosKernels::Impl::kk_is_gpu_exec_space<
        typename YVector::execution_space>()>::type * = nullptr>
void spMatVec_no_transpose(KokkosKernels::Experimental::Controls controls,
                           const AlphaType &alpha,
                           const KokkosSparse::Experimental::BlockCrsMatrix< AT, AO, AD, AM, AS> &A,
                           const XVector &x, const BetaType &beta, YVector &y,
                           bool useFallback) {

  typedef Kokkos::View<
      typename XVector::const_value_type *,
      typename KokkosKernels::Impl::GetUnifiedLayout<XVector>::array_layout,
      typename XVector::device_type,
      Kokkos::MemoryTraits<Kokkos::Unmanaged | Kokkos::RandomAccess> >
      XVector_Internal;

  typedef Kokkos::View<
      typename YVector::non_const_value_type *,
      typename KokkosKernels::Impl::GetUnifiedLayout<YVector>::array_layout,
      typename YVector::device_type, Kokkos::MemoryTraits<Kokkos::Unmanaged> >
      YVector_Internal;

  YVector_Internal y_i = y;

  // This is required to maintain semantics of KokkosKernels native SpMV:
  // if y contains NaN but beta = 0, the result y should be filled with 0.
  // For example, this is useful for passing in uninitialized y and beta=0.
  if (beta == Kokkos::ArithTraits<BetaType>::zero())
    Kokkos::deep_copy(y_i, Kokkos::ArithTraits<BetaType>::zero());
  else
    KokkosBlas::scal(y_i, beta, y_i);

  //
  // Treat the case y <- alpha * A * x + beta * y
  //

  XVector_Internal x_i       = x;
  const Ordinal blockSize    = A.blockDim();
  const Ordinal numBlockRows = A.numRows();
  //
  const bool conjugate = false;
  //
  ////////////
  assert(useFallback);
  ////////////

  typedef KokkosSparse::Experimental::BlockCrsMatrix<AT, AO, AD,
          Kokkos::MemoryTraits<Kokkos::Unmanaged>, AS> AMatrix_Internal;

  AMatrix_Internal A_internal = A;
  const auto &A_graph = A.graph;

#if defined(KOKKOS_ENABLE_SERIAL)
  if (std::is_same< typename AMatrix_Internal::device_type::execution_space,
      Kokkos::Serial>::value) {
    spmv_serial<decltype(A_graph)>(blockSize, alpha, &A.values[0], A_graph, &x[0], &y[0]);
    return;
  }
#endif

  typedef typename AMatrix_Internal::execution_space execution_space;

#ifdef KOKKOS_ENABLE_OPENMP
  if ((std::is_same<execution_space, Kokkos::OpenMP>::value) &&
      (std::is_same<typename std::remove_cv<typename AMatrix_Internal::value_type>::type, double>::value) &&
      (std::is_same<typename XVector::non_const_value_type, double>::value) &&
      (std::is_same<typename YVector::non_const_value_type, double>::value) &&
      ((int)A.graph.row_block_offsets.extent(0) == (int)omp_get_max_threads() + 1) &&
      (((uintptr_t)(const void *)(x.data()) % 64) == 0) &&
      (((uintptr_t)(const void *)(y.data()) % 64) == 0))
  {
    bspmv_raw_openmp_no_transpose<AMatrix_Internal, XVector, YVector>(alpha, A, x, y);
    return;
  }
#endif

  bool use_dynamic_schedule = false; // Forces the use of a dynamic schedule
  bool use_static_schedule  = false; // Forces the use of a static schedule
  if (controls.isParameter("schedule")) {
    if (controls.getParameter("schedule") == "dynamic") {
      use_dynamic_schedule = true;
    } else if (controls.getParameter("schedule") == "static") {
      use_static_schedule  = true;
    }
  }
  BSPMV_Functor<AMatrix_Internal,XVector,YVector> func (alpha, A_internal, x, y, 1, conjugate);
  if(((A.nnz()>10000000) || use_dynamic_schedule) && !use_static_schedule)
    Kokkos::parallel_for("KokkosSparse::bspmv<NoTranspose,Dynamic>",Kokkos::RangePolicy<execution_space, Kokkos::Schedule<Kokkos::Dynamic>>(0, A.numRows()),func);
  else
    Kokkos::parallel_for("KokkosSparse::bspmv<NoTranspose,Static>",Kokkos::RangePolicy<execution_space, Kokkos::Schedule<Kokkos::Static>>(0, A.numRows()),func);

}


/* ******************* */


template <int M>
inline void spmv_transpose_gemv(const Scalar alpha, const Scalar *Aval,
                                const Ordinal lda, const Ordinal xrow,
                                const Scalar *x_ptr, Scalar *y) {
  for (Ordinal ic = 0; ic < xrow; ++ic) {
    for (Ordinal kr = 0; kr < M; ++kr) {
      const auto alpha_value = alpha * Aval[ic + kr * lda];
      Kokkos::atomic_add(&y[ic], static_cast<Scalar>(alpha_value * x_ptr[kr]));
    }
  }
}

template <class StaticGraph, int N>
inline void spmv_transpose_serial(const Scalar alpha, Scalar *Avalues,
                                  const StaticGraph &Agraph, const Scalar *x,
                                  Scalar *y, const Ordinal ldy) {
  const Ordinal numBlockRows = Agraph.numRows();

  if (N == 1) {
    for (Ordinal i = 0; i < numBlockRows; ++i) {
      const auto jbeg = Agraph.row_map[i];
      const auto jend = Agraph.row_map[i + 1];
      for (Ordinal j = jbeg; j < jend; ++j) {
        const auto alpha_value = alpha * Avalues[j];
        const auto col_idx1    = Agraph.entries[j];
        y[col_idx1] += alpha_value * x[i];
      }
    }
    return;
  }

  const auto blockSize2 = N * N;
  for (Ordinal iblock = 0; iblock < numBlockRows; ++iblock) {
    const auto jbeg       = Agraph.row_map[iblock];
    const auto jend       = Agraph.row_map[iblock + 1];
    const auto xval_ptr   = &x[iblock * N];
    for (Ordinal jb = jbeg; jb < jend; ++jb) {
      const auto col_block = Agraph.entries[jb];
      auto yvec            = &y[N * col_block];
      const auto Aval_ptr  = &Avalues[jb * blockSize2];
      //
      spmv_transpose_gemv<N>(alpha, Aval_ptr, N, N, xval_ptr, yvec);
    }
  }
}

//
// spMatVec_transpose: version for CPU execution spaces (RangePolicy or
// trivial serial impl used)
//
template < class AT, class AO, class AD, class AM, class AS,
          class AlphaType, class XVector, class BetaType, class YVector,
          typename std::enable_if<!KokkosKernels::Impl::kk_is_gpu_exec_space<
              typename YVector::execution_space>()>::type * = nullptr>
void spMatVec_transpose(KokkosKernels::Experimental::Controls controls,
                        const AlphaType &alpha,
                        const KokkosSparse::Experimental::BlockCrsMatrix< AT, AO, AD, AM, AS> &A,
                        const XVector &x, const BetaType &beta, YVector &y,
                        bool useFallback) {

  typedef Kokkos::View<
      typename XVector::const_value_type *,
      typename KokkosKernels::Impl::GetUnifiedLayout<XVector>::array_layout,
      typename XVector::device_type,
      Kokkos::MemoryTraits<Kokkos::Unmanaged | Kokkos::RandomAccess> >
      XVector_Internal;

  typedef Kokkos::View<
      typename YVector::non_const_value_type *,
      typename KokkosKernels::Impl::GetUnifiedLayout<YVector>::array_layout,
      typename YVector::device_type, Kokkos::MemoryTraits<Kokkos::Unmanaged> >
      YVector_Internal;

  YVector_Internal y_i = y;

  // This is required to maintain semantics of KokkosKernels native SpMV:
  // if y contains NaN but beta = 0, the result y should be filled with 0.
  // For example, this is useful for passing in uninitialized y and beta=0.
  if (beta == Kokkos::ArithTraits<BetaType>::zero())
    Kokkos::deep_copy(y_i, Kokkos::ArithTraits<BetaType>::zero());
  else
    KokkosBlas::scal(y_i, beta, y_i);

  ////////////
  assert(useFallback);
  ////////////

  typedef KokkosSparse::Experimental::BlockCrsMatrix<AT, AO, AD,
          Kokkos::MemoryTraits<Kokkos::Unmanaged>, AS> AMatrix_Internal;

  AMatrix_Internal A_internal = A;

  //
  // Treat the case y <- alpha * A^T * x + beta * y
  //

  XVector_Internal x_i       = x;
  const Ordinal blockSize    = A_internal.blockDim();
  const Ordinal numBlockRows = A_internal.numRows();
  //
  const bool conjugate = false;
  const auto &A_graph = A_internal.graph;
  //
#if defined(KOKKOS_ENABLE_SERIAL)
  if (std::is_same< typename AMatrix_Internal::device_type::execution_space,
                   Kokkos::Serial>::value) {
    //
    if (blockSize <= std::min<size_t>(8, Impl::bmax)) {
      switch (blockSize) {
        default:
        case 1:
          spmv_transpose_serial<decltype(A_graph), 1>(
              alpha, &A_internal.values[0], A_graph, &x[0], &y[0], blockSize);
          break;
        case 2:
          spmv_transpose_serial<decltype(A_graph), 2>(
              alpha, &A_internal.values[0], A_graph, &x[0], &y[0], blockSize);
          break;
        case 3:
          spmv_transpose_serial<decltype(A_graph), 3>(
              alpha, &A_internal.values[0], A_graph, &x[0], &y[0], blockSize);
          break;
        case 4:
          spmv_transpose_serial<decltype(A_graph), 4>(
              alpha, &A_internal.values[0], A_graph, &x[0], &y[0], blockSize);
          break;
        case 5:
          spmv_transpose_serial<decltype(A_graph), 5>(
              alpha, &A_internal.values[0], A_graph, &x[0], &y[0], blockSize);
          break;
        case 6:
          spmv_transpose_serial<decltype(A_graph), 6>(
              alpha, &A_internal.values[0], A_graph, &x[0], &y[0], blockSize);
          break;
        case 7:
          spmv_transpose_serial<decltype(A_graph), 7>(
              alpha, &A_internal.values[0], A_graph, &x[0], &y[0], blockSize);
          break;
        case 8:
          spmv_transpose_serial<decltype(A_graph), 8>(
              alpha, &A_internal.values[0], A_graph, &x[0], &y[0], blockSize);
          break;
        case 9:
          spmv_transpose_serial<decltype(A_graph), 9>(
              alpha, &A_internal.values[0], A_graph, &x[0], &y[0], blockSize);
          break;
        case 10:
          spmv_transpose_serial<decltype(A_graph), 10>(
              alpha, &A_internal.values[0], A_graph, &x[0], &y[0], blockSize);
          break;
        case 11:
          spmv_transpose_serial<decltype(A_graph), 11>(
              alpha, &A_internal.values[0], A_graph, &x[0], &y[0], blockSize);
          break;
        case 12:
          spmv_transpose_serial<decltype(A_graph), 12>(
              alpha, &A_internal.values[0], A_graph, &x[0], &y[0], blockSize);
          break;
      }
      return;
    }
    //
    // --- Basic approach for large block sizes
    //
    const auto blockSize2 = blockSize * blockSize;
    for (Ordinal iblock = 0; iblock < numBlockRows; ++iblock) {
      const auto jbeg       = A_graph.row_map[iblock];
      const auto jend       = A_graph.row_map[iblock + 1];
      const auto xval_ptr   = &x[iblock * blockSize];
      for (Ordinal jb = jbeg; jb < jend; ++jb) {
        auto yvec           = &y[blockSize * A_graph.entries[jb]];
        const auto Aval_ptr = &A_internal.values[jb * blockSize2];
        for (Ordinal ic = 0; ic < blockSize; ++ic) {
          for (Ordinal kr = 0; kr < blockSize; ++kr) {
            yvec[ic] += alpha * Aval_ptr[ic + kr * blockSize] * xval_ptr[kr];
          }
        }
      }
    }
    return;
  }
#endif

#ifdef KOKKOS_ENABLE_OPENMP
  //
  const Ordinal blockSize2 = blockSize * blockSize;
  //
  if (blockSize <= std::min<size_t>(8, Impl::bmax)) {
    //
    // 2021/06/09 --- Cases for blockSize > 1 need to be modified
    //
    switch (blockSize) {
      default:
      case 1: {
#pragma omp parallel for schedule(static)
        for (Ordinal iblock = 0; iblock < numBlockRows; ++iblock) {
          const auto jbeg = A.graph.row_map[iblock];
          const auto jend = A.graph.row_map[iblock + 1];
          for (Ordinal j = jbeg; j < jend; ++j) {
            const auto col_block = A.graph.entries[j];
            Kokkos::atomic_add(
                &y[col_block],
                static_cast<Scalar>(alpha * A.values[j] * x[iblock]));
          }
        }
        break;
      }
      case 2: {
#pragma omp parallel for schedule(static)
        for (Ordinal iblock = 0; iblock < numBlockRows; ++iblock) {
          const auto jbeg       = A_graph.row_map[iblock];
          const auto jend       = A_graph.row_map[iblock + 1];
          const auto xval_ptr   = &x[iblock * blockSize];
          for (Ordinal jb = jbeg; jb < jend; ++jb) {
            const auto col_block = A.graph.entries[jb];
            auto yvec            = &y[blockSize * col_block];
            const auto Aval_ptr  = &A.values[blockSize2 * jb];
            spmv_transpose_gemv<2>(alpha, Aval_ptr, 2, 2, xval_ptr,
                                   yvec);
          }
        }
        break;
      }
      case 3: {
#pragma omp parallel for schedule(static)
        for (Ordinal iblock = 0; iblock < numBlockRows; ++iblock) {
          const auto jbeg       = A_graph.row_map[iblock];
          const auto jend       = A_graph.row_map[iblock + 1];
          const auto xval_ptr   = &x[iblock * blockSize];
          for (Ordinal jb = jbeg; jb < jend; ++jb) {
            const auto col_block = A.graph.entries[jb];
            auto yvec            = &y[blockSize * col_block];
            const auto Aval_ptr  = &A.values[blockSize2 * jb];
            spmv_transpose_gemv<3>(alpha, Aval_ptr, 3, 3, xval_ptr,
                                   yvec);
          }
        }
        break;
      }
      case 4: {
#pragma omp parallel for schedule(static)
        for (Ordinal iblock = 0; iblock < numBlockRows; ++iblock) {
          const auto jbeg       = A_graph.row_map[iblock];
          const auto jend       = A_graph.row_map[iblock + 1];
          const auto xval_ptr   = &x[iblock * blockSize];
          for (Ordinal jb = jbeg; jb < jend; ++jb) {
            const auto col_block = A.graph.entries[jb];
            auto yvec            = &y[blockSize * col_block];
            const auto Aval_ptr  = &A.values[blockSize2 * jb];
            spmv_transpose_gemv<4>(alpha, Aval_ptr, 4, 4, xval_ptr,
                                   yvec);
          }
        }
        break;
      }
      case 5: {
#pragma omp parallel for schedule(static)
        for (Ordinal iblock = 0; iblock < numBlockRows; ++iblock) {
          const auto jbeg       = A_graph.row_map[iblock];
          const auto jend       = A_graph.row_map[iblock + 1];
          const auto xval_ptr   = &x[iblock * blockSize];
          for (Ordinal jb = jbeg; jb < jend; ++jb) {
            const auto col_block = A.graph.entries[jb];
            auto yvec            = &y[blockSize * col_block];
            const auto Aval_ptr  = &A.values[blockSize2 * jb];
            spmv_transpose_gemv<5>(alpha, Aval_ptr, 5, 5, xval_ptr,
                                   yvec);
          }
        }
        break;
      }
      case 6: {
#pragma omp parallel for schedule(static)
        for (Ordinal iblock = 0; iblock < numBlockRows; ++iblock) {
          const auto jbeg       = A_graph.row_map[iblock];
          const auto jend       = A_graph.row_map[iblock + 1];
          const auto xval_ptr   = &x[iblock * blockSize];
          for (Ordinal jb = jbeg; jb < jend; ++jb) {
            const auto col_block = A.graph.entries[jb];
            auto yvec            = &y[blockSize * col_block];
            const auto Aval_ptr  = &A.values[blockSize2 * jb];
            spmv_transpose_gemv<6>(alpha, Aval_ptr, 6, 6, xval_ptr,
                                   yvec);
          }
        }
        break;
      }
      case 7: {
#pragma omp parallel for schedule(static)
        for (Ordinal iblock = 0; iblock < numBlockRows; ++iblock) {
          const auto jbeg       = A_graph.row_map[iblock];
          const auto jend       = A_graph.row_map[iblock + 1];
          const auto xval_ptr   = &x[iblock * blockSize];
          for (Ordinal jb = jbeg; jb < jend; ++jb) {
            const auto col_block = A.graph.entries[jb];
            auto yvec            = &y[blockSize * col_block];
            const auto Aval_ptr  = &A.values[blockSize2 * jb];
            spmv_transpose_gemv<7>(alpha, Aval_ptr, 7, 7, xval_ptr,
                                   yvec);
          }
        }
        break;
      }
      case 8: {
#pragma omp parallel for schedule(static)
        for (Ordinal iblock = 0; iblock < numBlockRows; ++iblock) {
          const auto jbeg       = A_graph.row_map[iblock];
          const auto jend       = A_graph.row_map[iblock + 1];
          const auto xval_ptr   = &x[iblock * blockSize];
          for (Ordinal jb = jbeg; jb < jend; ++jb) {
            const auto col_block = A.graph.entries[jb];
            auto yvec            = &y[blockSize * col_block];
            const auto Aval_ptr  = &A.values[blockSize2 * jb];
            spmv_transpose_gemv<8>(alpha, Aval_ptr, 8, 8, xval_ptr,
                                   yvec);
          }
        }
        break;
      }
    }
  } else {
#pragma omp parallel for schedule(static)
    for (Ordinal iblock = 0; iblock < numBlockRows; ++iblock) {
      const auto jbeg       = A.graph.row_map[iblock];
      const auto jend       = A.graph.row_map[iblock + 1];
      const auto xvec       = &x[iblock * blockSize];
      for (Ordinal jb = jbeg; jb < jend; ++jb) {
        const auto col_block = A.graph.entries[jb];
        auto yvec            = &y[blockSize * col_block];
        const auto Aval_ptr  = &A.values[blockSize2 * jb];
        for (Ordinal ic = 0; ic < blockSize; ++ic) {
          const auto xvalue = xvec[ic];
          for (Ordinal kr = 0; kr < blockSize; ++kr) {
            Kokkos::atomic_add( &yvec[kr],
                static_cast<Scalar>(alpha * Aval_ptr[kr + ic * blockSize] * xvalue) );
          }
        }
      }
    }
  }
  return;
#endif

  bool use_dynamic_schedule = false;  // Forces the use of a dynamic schedule
  bool use_static_schedule  = false;  // Forces the use of a static schedule
  if (controls.isParameter("schedule")) {
    if (controls.getParameter("schedule") == "dynamic") {
      use_dynamic_schedule = true;
    } else if (controls.getParameter("schedule") == "static") {
      use_static_schedule = true;
    }
  }

}

}  // namespace Impl

}  // namespace KokkosSparse

#endif  // KOKKOSKERNELS_KOKKOSSPARSE_SPMV_IMPL_BLOCK_CRS_HPP