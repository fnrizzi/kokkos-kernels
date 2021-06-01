#include "Kokkos_Core.hpp"

#include "KokkosBlas.hpp"
#include "KokkosKernels_default_types.hpp"
#include "KokkosSparse_BlockCrsMatrix.hpp"
#include "KokkosSparse_CrsMatrix.hpp"
#include "KokkosSparse_spmv.hpp"

#include "KokkosKernels_helpers.hpp"
#include "KokkosKernels_Controls.hpp"
#include "KokkosKernels_Test_Structured_Matrix.hpp"
#include "KokkosKernels_Utils.hpp"
#include "KokkosKernels_ExecSpaceUtils.hpp"

#include <chrono>
#include <type_traits>

using Scalar  = default_scalar;
using Ordinal = default_lno_t;
using Offset  = default_size_type;
using Layout  = default_layout;

using device_type = typename Kokkos::Device<
    Kokkos::DefaultExecutionSpace,
    typename Kokkos::DefaultExecutionSpace::memory_space>;

using crs_matrix_t_ =
typename KokkosSparse::CrsMatrix<Scalar, Ordinal, device_type, void, Offset>;

using values_type = typename crs_matrix_t_::values_type;

using bcrs_matrix_t_ =
typename KokkosSparse::Experimental::BlockCrsMatrix<Scalar, Ordinal, device_type, void,
    Offset>;

namespace details {

const Scalar SC_ONE = Kokkos::ArithTraits<Scalar>::one();
const Scalar SC_ZERO = Kokkos::ArithTraits<Scalar>::zero();

template <typename mat_structure>
crs_matrix_t_ generate_crs_matrix(const std::string stencil,
                                const mat_structure& structure,
                                const int blockSize,
                                std::vector<Ordinal> &mat_rowmap,
                                std::vector<Ordinal> &mat_colidx,
                                std::vector<Scalar> &mat_val
) {
  crs_matrix_t_ mat_b1 =
      Test::generate_structured_matrix2D<crs_matrix_t_>(stencil, structure);

  if (blockSize == 1)
    return mat_b1;

  //
  // Fill blocks with random values
  //

  int nRow = blockSize * mat_b1.numRows();
  int nCol = blockSize * mat_b1.numCols();
  size_t nnz = blockSize * blockSize * mat_b1.nnz();

  mat_val.resize(nnz);
  Scalar *val_ptr = &mat_val[0];

  for (size_t ii = 0; ii < nnz; ++ii)
    val_ptr[ii] = static_cast<Scalar>( std::rand() / (RAND_MAX + static_cast<Scalar>(1) ));

  mat_rowmap.resize(nRow + 1);
  int *rowmap = &mat_rowmap[0];
  rowmap[0] = 0;

  mat_colidx.resize(nnz);
  int *cols = &mat_colidx[0];

  for (int ir = 0; ir < mat_b1.numRows(); ++ir) {
    auto mat_row = mat_b1.rowConst(ir);
    for (int ib = 0; ib < blockSize; ++ib) {
      int my_row = ir * blockSize + ib;
      rowmap[my_row + 1] = rowmap[my_row] + mat_row.length * blockSize;
      for (int ijk = 0; ijk < mat_row.length; ++ijk) {
        int col0 = mat_row.colidx(ijk);
        for (int jb = 0; jb < blockSize; ++jb) {
          cols[rowmap[my_row] + ijk * blockSize + jb] = col0 * blockSize + jb;
        }
      } // for (int ijk = 0; ijk < mat_row.length; ++ijk)
    }
  } // for (int ir = 0; ir < mat_b1.numRows(); ++ir)

  return crs_matrix_t_("new_crs_matr", nRow, nCol, nnz, val_ptr, rowmap, cols);

}


//
// spmv_beta_no_transpose: version for GPU execution
//
template <class AlphaType,
    class XVector,
    class BetaType,
    class YVector,
    typename std::enable_if<KokkosKernels::Impl::kk_is_gpu_exec_space<typename YVector::execution_space>()>::type* = nullptr
>
void
spmv (const AlphaType& alpha,
      const bcrs_matrix_t_& A,
      const XVector& x,
      const BetaType& beta,
      YVector &y,
      const std::vector< Ordinal > &val_entries_ptr)
{
  std::cerr << " GPU implementation is not complete !!! \n";
  assert(0 > 1);
  ///
  ///  Check spmv and Tpetra::BlockCrsMatrix
  ///

}

template<class AMatrix,
    class XVector,
    class YVector,
    int dobeta,
    bool conjugate>
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
  const value_type beta;
  YVector m_y;

  const ordinal_type rows_per_team;
  const ordinal_type block_size;

  using y_value_type = typename YVector::non_const_value_type;

  const std::vector< ordinal_type > &val_entries_ptr;

  mutable std::vector<typename YVector::non_const_value_type> tmp;

  BSPMV_Functor(const value_type alpha_, const AMatrix m_A_, const XVector m_x_,
               const value_type beta_, const YVector m_y_,
               const int rows_per_team_,
               const std::vector< ordinal_type > &val_entries_ptr_)
      : alpha(alpha_),
        m_A(m_A_),
        m_x(m_x_),
        beta(beta_),
        m_y(m_y_),
        rows_per_team(rows_per_team_),
        block_size(m_A_.blockDim()),
        val_entries_ptr(val_entries_ptr_)
  {
    tmp.resize(block_size);
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const ordinal_type iRowBlock) const {
    //
    if (iRowBlock >= m_A.numRows())
      return;
    //
    const auto& A_graph  = m_A.graph;
    const auto Avalues = &m_A.values[val_entries_ptr[iRowBlock]];
    //
    const auto jbeg = A_graph.row_map[iRowBlock];
    const auto jend = A_graph.row_map[iRowBlock + 1];
    const auto num_blocks = jend - jbeg;
    for (Ordinal jb = 0; jb < num_blocks; ++jb) {
      const auto col_block = A_graph.entries[jb + jbeg];
      const auto x_val = &m_x[block_size * col_block];
      for (Ordinal k_dof = 0; k_dof < block_size; ++k_dof) {
        const auto A_row_values = &Avalues[jb * block_size +
                                           k_dof * block_size * num_blocks];
        for (Ordinal j_dof = 0; j_dof < block_size; ++j_dof) {
          tmp[k_dof] += A_row_values[j_dof] * x_val[j_dof];
        }
      }
    }
    //
    // Need to generalize
    //
    for (Ordinal k_dof = 0; k_dof < block_size; ++k_dof) {
      m_y[block_size * iRowBlock + k_dof] = alpha * tmp[k_dof];
    }
    //
  }

}; // struct BSPMV_Functor


template< int M >
inline void
spmv_serial_gemv(
            double *Aval,
            const Ordinal lda,
            const Ordinal xrow,
            const double *x_ptr,
            std::array<double, M> &y
)
{
  auto Aval_ptr = Aval;
  for (Ordinal ic = 0; ic < xrow; ++ic) {
    const auto xvalue = x_ptr[ic];
    for (Ordinal kr = 0; kr < M; ++kr) {
      y[kr] += Aval_ptr[kr * lda] * xvalue;
    }
    Aval_ptr = Aval_ptr + 1;
  }
}


template< int N >
inline double
spmv_serial_dot(
    const double *Aval,
    const double *x_ptr
)
{
  double res = 0.0;
  for (int ii = 0; ii < N; ++ii)
    res += Aval[ii] * x_ptr[ii];
  return res;
}


template< class StaticGraph, int N >
inline void
spmv_serial(const double alpha,
            double *Avalues,
            const StaticGraph &Agraph,
            const int* val_entries_ptr,
            const double *x,
            double *y
            )
{
  const Ordinal numBlockRows = Agraph.numRows();

  if (N == 1) {
    //typename YVector::non_const_value_type tmp1 = 0;
    double tmp1 = 0.0;
    for (Ordinal i = 0; i < numBlockRows; ++i) {
      const auto jbeg  = Agraph.row_map[i];
      const auto jend  = Agraph.row_map[i + 1];
      Ordinal j            = jbeg;
      const auto jdist = jend - jbeg;
      tmp1 = 0;
      for (Ordinal jj = 0; jj < jdist; ++jj, ++j) {
        const auto value1 = Avalues[j];
        const auto col_idx1 = Agraph.entries[j];
        const auto x_val1 = x[col_idx1];
        tmp1 += value1 * x_val1;
      }
      //
      // Need to generalize
      //
      y[i] += alpha * tmp1;
    }
    return;
  }

  std::array<double, N> tmp;
  for (Ordinal iblock = 0; iblock < numBlockRows; ++iblock) {
    const auto jbeg       = Agraph.row_map[iblock];
    int j                 = jbeg;
    const auto jend = Agraph.row_map[iblock + 1];
    const auto num_blocks = jend - jbeg;
    const auto Aval = Avalues + val_entries_ptr[iblock];
    tmp.fill(0);

    const auto lda = num_blocks * N;
    for (Ordinal jb = 0; jb < num_blocks; ++jb, ++j) {
      const auto col_block = Agraph.entries[j];
      const auto xval_ptr  = x + N * col_block;
      const auto shift = jb * N;
      auto Aval_ptr = Aval + shift;
      //
      spmv_serial_gemv<N>(Aval_ptr, lda, N, xval_ptr, tmp);
    }
    //
    auto yvec = &y[iblock * N];
    for (Ordinal ii = 0; ii < N; ++ii) {
      yvec[ii] += alpha * tmp[ii];
    }
  }

}


template< int M >
inline void
spmv_row_gemv(
    const int *col_idx,
    const int num_blocks,
    double *Aval,
    const Ordinal lda,
    const Ordinal x_block_size,
    const double *x_ptr,
    double *yvec
)
{
  std::array<double, M> tmp;
  tmp.fill(0);
  for (Ordinal jb = 0; jb < num_blocks; ++jb) {
    const auto col_block = col_idx[jb];
    const auto x_val = &x_ptr[x_block_size * col_block];
    auto Aval_ptr = Aval + x_block_size * jb;
    for (Ordinal ic = 0; ic < x_block_size; ++ic) {
      const auto xvalue = x_val[ic];
      for (Ordinal kr = 0; kr < M; ++kr) {
        tmp[kr] += Aval_ptr[kr * lda] * xvalue;
      }
      Aval_ptr = Aval_ptr + 1;
    }
//    spmv_serial_gemv<M>(Aval_ptr, lda, x_block_size, x_val, tmp);
  }
  //
  for (Ordinal kk = 0; kk < M; ++kk)
    yvec[kk] += tmp[kk];
}


//
// spmv_beta_no_transpose: version for CPU execution spaces (RangePolicy or trivial serial impl used)
//
template <class AlphaType,
          class XVector,
          class BetaType,
          class YVector,
          typename std::enable_if<!KokkosKernels::Impl::kk_is_gpu_exec_space<typename YVector::execution_space>()>::type* = nullptr
          >
void
spmv (const AlphaType& alpha,
      const bcrs_matrix_t_& A,
      const XVector& x,
      const BetaType& beta,
      YVector &y,
      const std::vector< Ordinal > &val_entries_ptr)
{
  typedef Kokkos::View<
      typename XVector::const_value_type*,
      typename KokkosKernels::Impl::GetUnifiedLayout<XVector>::array_layout,
      typename XVector::device_type,
      Kokkos::MemoryTraits<Kokkos::Unmanaged|Kokkos::RandomAccess> > XVector_Internal;

  typedef Kokkos::View<
      typename YVector::non_const_value_type*,
      typename KokkosKernels::Impl::GetUnifiedLayout<YVector>::array_layout,
      typename YVector::device_type,
      Kokkos::MemoryTraits<Kokkos::Unmanaged> > YVector_Internal;

  typedef typename bcrs_matrix_t_::device_type::execution_space execution_space;

  YVector_Internal y_i = y;

  //This is required to maintain semantics of KokkosKernels native SpMV:
  //if y contains NaN but beta = 0, the result y should be filled with 0.
  //For example, this is useful for passing in uninitialized y and beta=0.
  if (beta == Kokkos::ArithTraits<BetaType>::zero())
    Kokkos::deep_copy(y_i, Kokkos::ArithTraits<BetaType>::zero());
  else
    KokkosBlas::scal(y_i, beta, y_i);

  if(alpha == Kokkos::ArithTraits<AlphaType>::zero() ||
     A.numRows() == 0 || A.numCols() == 0 || A.nnz() == 0 || A.blockDim() == 0)
  {
    return;
  }

  //
  // Treat the case y <- alpha * A * x + beta * y
  //

  XVector_Internal x_i = x;
  const Ordinal blockSize    = A.blockDim();
  const Ordinal numBlockRows = A.numRows();
  //
  const bool conjugate = false;
  //

#if defined(KOKKOS_ENABLE_SERIAL)
  if (std::is_same<execution_space, Kokkos::Serial>::value) {
    //
    const auto& A_graph = A.graph;
    //
    if (blockSize < 9) {
      switch (blockSize) {
        default:
        case 1:
          spmv_serial<decltype(A_graph), 1>(alpha, &A.values[0], A_graph,
                                            &val_entries_ptr[0], &x[0], &y[0]);
          break;
        case 2:
          spmv_serial<decltype(A_graph), 2>(alpha, &A.values[0], A_graph,
                                            &val_entries_ptr[0], &x[0], &y[0]);
          break;
        case 3:
          spmv_serial<decltype(A_graph), 3>(alpha, &A.values[0], A_graph,
                                            &val_entries_ptr[0], &x[0], &y[0]);
          break;
        case 4:
          spmv_serial<decltype(A_graph), 4>(alpha, &A.values[0], A_graph,
                                            &val_entries_ptr[0], &x[0], &y[0]);
          break;
        case 5:
          spmv_serial<decltype(A_graph), 5>(alpha, &A.values[0], A_graph,
                                            &val_entries_ptr[0], &x[0], &y[0]);
          break;
        case 6:
          spmv_serial<decltype(A_graph), 6>(alpha, &A.values[0], A_graph,
                                            &val_entries_ptr[0], &x[0], &y[0]);
          break;
        case 7:
          spmv_serial<decltype(A_graph), 7>(alpha, &A.values[0], A_graph,
                                            &val_entries_ptr[0], &x[0], &y[0]);
          break;
        case 8:
          spmv_serial<decltype(A_graph), 8>(alpha, &A.values[0], A_graph,
                                            &val_entries_ptr[0], &x[0], &y[0]);
          break;
      }
    } else {
      constexpr Ordinal unroll   = 8;
      const Ordinal bs_unroll    = (blockSize / unroll);
      const Ordinal jj_dof_start = bs_unroll * unroll;
      //std::array<double, unroll> tmp;
      for (Ordinal iblock = 0; iblock < numBlockRows; ++iblock) {
        const auto jbeg       = A_graph.row_map[iblock];
        const auto jend       = A_graph.row_map[iblock + 1];
        const auto num_blocks = jend - jbeg;
        auto Avalues    = &A.values[val_entries_ptr[iblock]];
        const auto len_blocks = blockSize * num_blocks;
        for (Ordinal kdist = 0, k_dof = 0; kdist < bs_unroll; kdist += 1, k_dof += unroll) {
          spmv_row_gemv<unroll>(&A_graph.entries[jbeg], num_blocks, &Avalues[k_dof * len_blocks], len_blocks,
                                blockSize, &x_i[0], &y_i[blockSize * iblock + k_dof]);
        }
        //
        const auto kdiff = blockSize - jj_dof_start;
        const Ordinal k_dof = jj_dof_start;
        switch (kdiff) {
          default:
          case 0:
            break;
          case 1:
              spmv_row_gemv<1>(&A_graph.entries[jbeg], num_blocks, &Avalues[k_dof * len_blocks], len_blocks,
                               blockSize, &x_i[0], &y_i[blockSize * iblock + k_dof]);
                break;
            case 2:
spmv_row_gemv<2>(&A_graph.entries[jbeg], num_blocks, &Avalues[k_dof * len_blocks], len_blocks,
                      blockSize, &x_i[0],  &y_i[blockSize * iblock + k_dof]);
                break;
            case 3:
spmv_row_gemv<3>(&A_graph.entries[jbeg], num_blocks, &Avalues[k_dof * len_blocks], len_blocks,
                      blockSize, &x_i[0],  &y_i[blockSize * iblock + k_dof]);
                break;
            case 4:
spmv_row_gemv<4>(&A_graph.entries[jbeg], num_blocks, &Avalues[k_dof * len_blocks], len_blocks,
                      blockSize, &x_i[0],  &y_i[blockSize * iblock + k_dof]);
                break;
            case 5:
spmv_row_gemv<5>(&A_graph.entries[jbeg], num_blocks, &Avalues[k_dof * len_blocks], len_blocks,
                      blockSize, &x_i[0],  &y_i[blockSize * iblock + k_dof]);
                break;
            case 6:
spmv_row_gemv<6>(&A_graph.entries[jbeg], num_blocks, &Avalues[k_dof * len_blocks], len_blocks,
                      blockSize, &x_i[0],  &y_i[blockSize * iblock + k_dof]);
                break;
            case 7:
spmv_row_gemv<7>(&A_graph.entries[jbeg], num_blocks, &Avalues[k_dof * len_blocks], len_blocks,
                      blockSize, &x_i[0],  &y_i[blockSize * iblock + k_dof]);
                break;
            case 8:
spmv_row_gemv<8>(&A_graph.entries[jbeg], num_blocks, &Avalues[k_dof * len_blocks], len_blocks,
                      blockSize, &x_i[0],  &y_i[blockSize * iblock + k_dof]);
                break;
          } // switch (kdiff)
        }
      }
    return;
  }
#endif

#ifdef KOKKOS_ENABLE_OPENMP
  //
  // This section terminates for the moment.
  // terminate called recursively
  //
  const auto& A_graph      = A.graph;
  const Ordinal blockSize2 = blockSize * blockSize;
  //
  KokkosKernels::Experimental::Controls controls;
  bool use_dynamic_schedule = false; // Forces the use of a dynamic schedule
  bool use_static_schedule  = false; // Forces the use of a static schedule
  if(controls.isParameter("schedule")) {
    if(controls.getParameter("schedule") == "dynamic") {
      use_dynamic_schedule = true;
    } else if(controls.getParameter("schedule") == "static") {
      use_static_schedule  = true;
    }
  }
  //
#pragma omp parallel for schedule(static)
  for (Ordinal iblock = 0; iblock < numBlockRows; ++iblock) {
    const auto jbeg       = A_graph.row_map[iblock];
    const auto jend       = A_graph.row_map[iblock + 1];
    const auto num_blocks = jend - jbeg;
    const auto Avalues    = &A.values[val_entries_ptr[iblock]];
    std::vector<typename YVector::non_const_value_type> tmp(blockSize, 0);
    const auto len_blocks = blockSize * num_blocks;
    for (Ordinal jb = 0; jb < num_blocks; ++jb) {
      const auto col_block = A_graph.entries[jb + jbeg];
      const auto x_val = &x_i[blockSize * col_block];
      auto A_row_k = Avalues + blockSize * jb;
      for (Ordinal k_dof = 0; k_dof < blockSize; ++k_dof) {
        double tmp_k = 0;
        for (Ordinal j_dof = 0; j_dof < blockSize; ++j_dof) {
          tmp_k += A_row_k[j_dof] * x_val[j_dof];
        }
        tmp[k_dof] += tmp_k;
        A_row_k = A_row_k + len_blocks;
      }
      //
    }
    //
    // Need to generalize
    //
    auto yvec = &y_i[blockSize * iblock];
    for (Ordinal k_dof = 0; k_dof < blockSize; ++k_dof) {
      yvec[k_dof] = alpha * tmp[k_dof];
    }
  }
#endif

}

typename values_type::non_const_type make_lhs(const int numRows) {
  typename values_type::non_const_type x("lhs", numRows);
  for (Ordinal ir = 0; ir < numRows; ++ir)
    x(ir) = std::rand() / static_cast<Scalar>(RAND_MAX);
  return x;
}


template<typename mtx_t>
std::chrono::duration<double> measure(const mtx_t &myMatrix, const Scalar alpha, const Scalar beta, const int repeat) {
  const Ordinal numRows = myMatrix.numRows();

  auto const x = make_lhs(numRows);
  typename values_type::non_const_type y("rhs", numRows);

  auto tBegin = std::chrono::high_resolution_clock::now();
  for (int ir = 0; ir < repeat; ++ir) {
    KokkosSparse::spmv("N", alpha, myMatrix, x, beta, y);
  }
  auto tEnd = std::chrono::high_resolution_clock::now();

  return tEnd - tBegin;
}

template<typename bmtx_t>
std::chrono::duration<double> measureBlock(const bmtx_t &myBlockMatrix, const std::vector<Ordinal> &val_entries_ptr, const Scalar alpha, const Scalar beta, const int repeat) {
  auto const numRows = myBlockMatrix.numRows() * myBlockMatrix.blockDim();
  auto const x = make_lhs(numRows);
  typename values_type::non_const_type y("rhs", numRows);

  auto tBegin = std::chrono::high_resolution_clock::now();
  for (int ir = 0; ir < repeat; ++ir) {
    details::spmv(alpha, myBlockMatrix, x, beta, y, val_entries_ptr);
  }
  auto tEnd = std::chrono::high_resolution_clock::now();

  return tEnd - tBegin;
}

template<typename mtx_t>
std::vector<Ordinal> buildEntryValuePointers(const mtx_t &myBlockMatrix) {
  // Build pointer to entry values
  const Ordinal blockSize = myBlockMatrix.blockDim();
  const Ordinal numBlocks = myBlockMatrix.numRows();
  std::vector<Ordinal> val_entries_ptr(numBlocks + 1, 0);
  for (Ordinal ir = 0; ir < numBlocks; ++ir) {
    const auto jbeg = myBlockMatrix.graph.row_map[ir];
    const auto jend = myBlockMatrix.graph.row_map[ir + 1];
    val_entries_ptr[ir + 1] = val_entries_ptr[ir] + blockSize * blockSize * (jend - jbeg);
  }
  return val_entries_ptr;
}

template<typename mtx_t, typename bmtx_t>
void compare(const mtx_t &myMatrix, const bmtx_t &myBlockMatrix, const std::vector<Ordinal> &val_entries_ptr, const Scalar alpha, const Scalar beta, double &error, double &maxNorm) {
  error = 0.0;
  maxNorm = 0.0;

  const int numRows = myMatrix.numRows();
  auto const x = make_lhs(numRows);
  typename values_type::non_const_type y("rhs", numRows);
  typename values_type::non_const_type yref("ref", numRows);

  KokkosSparse::spmv("N", alpha, myMatrix, x, beta, yref);
  details::spmv(alpha, myBlockMatrix, x, beta, y, val_entries_ptr);

  for (Ordinal ir = 0, numRows = y.size(); ir < numRows; ++ir) {
    //std::cout << '\t' << yref(ir) << '\t' << y(ir) << std::endl;
    error   = std::max<double>(error, std::abs(yref(ir) - y(ir)));
    maxNorm = std::max<double>(maxNorm, std::abs(yref(ir)));
  }
}

template<typename mtx_t>
void testMatrix(const mtx_t &myMatrix, const int blockSize, const int repeat) {

  const Scalar alpha = details::SC_ONE;
  const Scalar beta  = details::SC_ZERO;

  auto const numRows = myMatrix.numRows();

  std::chrono::duration<double> dt_crs = measure(myMatrix, alpha, beta, repeat);

  std::cout << " Total time for Crs Mat-Vec " << dt_crs.count() << " Avg. "
            << dt_crs.count() / static_cast<double>(repeat);
  std::cout << " Flops (mult only) " << myMatrix.nnz() * static_cast<double>(repeat / dt_crs.count() ) << "\n";
  std::cout << " ------------------------ \n";

  //
  // Use BlockCrsMatrix format
  //
  bcrs_matrix_t_ myBlockMatrix(myMatrix, blockSize);

  auto const val_entries_ptr = buildEntryValuePointers(myBlockMatrix);

  double error = 0.0, maxNorm = 0.0;
  compare(myMatrix, myBlockMatrix, val_entries_ptr, alpha, beta, error, maxNorm);

  std::cout << " Error BlockCrsMatrix " << error << " maxNorm " << maxNorm << "\n";
  std::cout << " ------------------------ \n";

  //
  // Test speed of Mat-Vec product
  //
  std::chrono::duration<double> dt_bcrs = measureBlock(myBlockMatrix, val_entries_ptr, alpha, beta, repeat);

  std::cout << " Total time for BlockCrs Mat-Vec " << dt_bcrs.count()
            << " Avg. " << dt_bcrs.count() / static_cast<double>(repeat);
  std::cout << " Flops (mult only) " << myMatrix.nnz() * static_cast<double>(repeat / dt_bcrs.count() );
  std::cout << "\n";
  //
  std::cout << " Ratio = " << dt_bcrs.count() / dt_crs.count();

  if (dt_bcrs.count() < dt_crs.count()) {
    std::cout << " --- GOOD --- ";
  } else {
    std::cout << " ((( Not Faster ))) ";
  }
  std::cout << "\n";

  std::cout << " ======================== \n";
}

int testRandom(const int repeat = 7500, const int minBlockSize = 1, const int maxBlockSize = 12) {
    int return_value = 0;

    // The mat_structure view is used to generate a matrix using
    // finite difference (FD) or finite element (FE) discretization
    // on a cartesian grid.
    // Each row corresponds to an axis (x, y and z)
    // In each row the first entry is the number of grid point in
    // that direction, the second and third entries are used to apply
    // BCs in that direction, BC=0 means Neumann BC is applied,
    // BC=1 means Dirichlet BC is applied by zeroing out the row and putting
    // one on the diagonal.
    Kokkos::View<Ordinal* [3], Kokkos::HostSpace> mat_structure("Matrix Structure", 2);
    mat_structure(0, 0) = 64;  // Request 10 grid point in 'x' direction
    mat_structure(0, 1) = 0;   // Add BC to the left
    mat_structure(0, 2) = 0;   // Add BC to the right
    mat_structure(1, 0) = 63;  // Request 10 grid point in 'y' direction
    mat_structure(1, 1) = 0;   // Add BC to the bottom
    mat_structure(1, 2) = 0;   // Add BC to the top

    for (int blockSize = minBlockSize; blockSize <= maxBlockSize; ++blockSize) {

      std::vector<int> mat_rowmap, mat_colidx;
      std::vector<double> mat_val;

      crs_matrix_t_ myMatrix = details::generate_crs_matrix(
          "FD", mat_structure, blockSize, mat_rowmap, mat_colidx, mat_val);

      std::cout << " ======================== \n";
      std::cout << " Block Size " << blockSize;
      std::cout << " Matrix Size " << myMatrix.numRows() << " nnz " << myMatrix.nnz()
                << "\n";

      testMatrix(myMatrix, blockSize, repeat);

    }
    return return_value;
}

} // namespace details

//#define TEST_RANDOM_BSPMV

int main() {

  Kokkos::initialize();

  int return_value = details::testRandom();

  Kokkos::finalize();

  return return_value;
}
