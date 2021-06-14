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
#include <set>
#include <type_traits>

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

using bcrs_matrix_t_ = typename KokkosSparse::Experimental::BlockCrsMatrix<
    Scalar, Ordinal, device_type, void, Offset>;

using MultiVector_Internal = typename Kokkos::View< Scalar**, Layout, device_type >;

namespace details {

const Scalar SC_ONE  = Kokkos::ArithTraits<Scalar>::one();
const Scalar SC_ZERO = Kokkos::ArithTraits<Scalar>::zero();

/// \brief Generate a CrsMatrix object for a matrix
/// "with multiple DOFs per node"
///
/// \tparam mat_structure
/// \param stencil
/// \param structure
/// \param blockSize
/// \param mat_rowmap
/// \param mat_colidx
/// \param mat_val
/// \return
template <typename mat_structure>
crs_matrix_t_ generate_crs_matrix(const std::string stencil,
                                  const mat_structure &structure,
                                  const int blockSize,
                                  std::vector<Ordinal> &mat_rowmap,
                                  std::vector<Ordinal> &mat_colidx,
                                  std::vector<Scalar> &mat_val) {
  crs_matrix_t_ mat_b1 =
      Test::generate_structured_matrix2D<crs_matrix_t_>(stencil, structure);

  if (blockSize == 1) return mat_b1;

  //
  // Fill blocks with random values
  //

  int nRow   = blockSize * mat_b1.numRows();
  int nCol   = blockSize * mat_b1.numCols();
  size_t nnz = blockSize * blockSize * mat_b1.nnz();

  mat_val.resize(nnz);
  Scalar *val_ptr = &mat_val[0];

  for (size_t ii = 0; ii < nnz; ++ii)
    val_ptr[ii] =
        static_cast<Scalar>(std::rand() / (RAND_MAX + static_cast<Scalar>(1)));

  mat_rowmap.resize(nRow + 1);
  int *rowmap = &mat_rowmap[0];
  rowmap[0]   = 0;

  mat_colidx.resize(nnz);
  int *cols = &mat_colidx[0];

  for (int ir = 0; ir < mat_b1.numRows(); ++ir) {
    auto mat_b1_row = mat_b1.rowConst(ir);
    for (int ib = 0; ib < blockSize; ++ib) {
      int my_row         = ir * blockSize + ib;
      rowmap[my_row + 1] = rowmap[my_row] + mat_b1_row.length * blockSize;
      for (int ijk = 0; ijk < mat_b1_row.length; ++ijk) {
        int col0 = mat_b1_row.colidx(ijk);
        for (int jb = 0; jb < blockSize; ++jb) {
          cols[rowmap[my_row] + ijk * blockSize + jb] = col0 * blockSize + jb;
        }
      }  // for (int ijk = 0; ijk < mat_row.length; ++ijk)
    }
  }  // for (int ir = 0; ir < mat_b1.numRows(); ++ir)

  return crs_matrix_t_("new_crs_matr", nRow, nCol, nnz, val_ptr, rowmap, cols);
}

/// \brief Convert a CrsMatrix object to a BlockCrsMatrix object
///
/// \param mat_crs
/// \param blockSize
/// \return
///
/// \note We assume that each block has a constant block size
/// (in both directions, i.e. row and column)
/// \note The numerical values for each individual block are stored
/// contiguously in a (blockSize * blockSize) space
/// in a row-major fashion.
/// (2021/06/08) This storage is different from the BlockCrsMatrix constructor
/// ```  BlockCrsMatrix (const KokkosSparse::CrsMatrix<SType, OType, DType,
/// MTType, IType> &crs_mtx,
///                      const OrdinalType blockDimIn) ```
bcrs_matrix_t_ to_block_crs_matrix(const crs_matrix_t_ &mat_crs,
                                   const int blockSize) {
  if (blockSize == 1) {
    bcrs_matrix_t_ bmat(mat_crs, blockSize);
    return bmat;
  }

  if ((mat_crs.numRows() % blockSize > 0) ||
      (mat_crs.numCols() % blockSize > 0)) {
    std::cerr
        << "\n !!! Matrix Dimensions Do Not Match Block Structure !!! \n\n";
    exit(-123);
  }

  // block_rows will accumulate the number of blocks per row - this is NOT the
  // row_map with cum sum!!
  Ordinal nbrows = mat_crs.numRows() / blockSize;
  std::vector<Ordinal> block_rows(nbrows, 0);

  Ordinal nbcols = mat_crs.numCols() / blockSize;

  Ordinal numBlocks = 0;
  for (Ordinal i = 0; i < mat_crs.numRows(); i += blockSize) {
    Ordinal current_blocks = 0;
    for (Ordinal j = 0; j < blockSize; ++j) {
      auto n_entries = mat_crs.graph.row_map(i + 1 + j) -
                       mat_crs.graph.row_map(i + j) + blockSize - 1;
      current_blocks = std::max(current_blocks, n_entries / blockSize);
    }
    numBlocks += current_blocks;                 // cum sum
    block_rows[i / blockSize] = current_blocks;  // frequency counts
  }

  Kokkos::View<Ordinal *, Kokkos::LayoutLeft, device_type> rows("new_row",
                                                                nbrows + 1);
  rows(0) = 0;
  for (Ordinal i = 0; i < nbrows; ++i) rows(i + 1) = rows(i) + block_rows[i];

  Kokkos::View<Ordinal *, Kokkos::LayoutLeft, device_type> cols("new_col",
                                                                rows[nbrows]);
  cols(0) = 0;

  for (Ordinal ib = 0; ib < nbrows; ++ib) {
    auto ir_start = ib * blockSize;
    auto ir_stop  = (ib + 1) * blockSize;
    std::set<Ordinal> col_set;
    for (Ordinal ir = ir_start; ir < ir_stop; ++ir) {
      for (Ordinal jk = mat_crs.graph.row_map(ir);
           jk < mat_crs.graph.row_map(ir + 1); ++jk) {
        col_set.insert(mat_crs.graph.entries(jk) / blockSize);
      }
    }
    assert(col_set.size() == block_rows[ib]);
    Ordinal icount = 0;
    auto *col_list = &cols(rows(ib));
    for (auto col_block : col_set) col_list[icount++] = col_block;
  }

  Ordinal annz = numBlocks * blockSize * blockSize;
  bcrs_matrix_t_::values_type vals("values", annz);
  for (Ordinal i = 0; i < annz; ++i) vals(i) = 0.0;

  for ( Ordinal ir = 0; ir < mat_crs.numRows(); ++ir ) {
    const auto iblock = ir / blockSize;
    const auto ilocal = ir % blockSize;
    Ordinal lda = blockSize * (rows[iblock + 1] - rows[iblock]);
    for (Ordinal jk = mat_crs.graph.row_map(ir); jk < mat_crs.graph.row_map(ir+1); ++jk) {
      const auto jc = mat_crs.graph.entries(jk);
      const auto jblock = jc / blockSize;
      const auto jlocal = jc % blockSize;
      for (Ordinal jkb = rows[iblock]; jkb < rows[iblock + 1]; ++jkb) {
        if (cols(jkb) == jblock) {
          Ordinal shift = rows[iblock] * blockSize * blockSize
              + blockSize * (jkb - rows[iblock]);
          vals(shift + jlocal + ilocal * lda) = mat_crs.values(jk);
          break;
        }
      }
    }
  }

  bcrs_matrix_t_ bmat("newblock", nbrows, nbcols, annz, vals, rows, cols,
                      blockSize);
  return bmat;
}

//
// spmv: version for GPU execution
//
template <class AlphaType, class XVector, class BetaType, class YVector,
          typename std::enable_if<KokkosKernels::Impl::kk_is_gpu_exec_space<
              typename YVector::execution_space>()>::type * = nullptr>
void spmv(const AlphaType &alpha, const bcrs_matrix_t_ &A, const XVector &x,
          const BetaType &beta, YVector &y)
{
  std::cerr << " GPU implementation is not complete !!! \n";
  assert(0 > 1);
  ///
  ///  Check spmv and Tpetra::BlockCrsMatrix
  ///
}


/////////////////////////////////////////////////////


constexpr size_t bmax = 12;

template <int M>
inline void spmv_serial_gemv(Scalar *Aval, const Ordinal lda,
                             const Scalar *x_ptr,
                             std::array<Scalar, details::bmax> &y) {
  for (Ordinal ic = 0; ic < M; ++ic) {
    const auto xvalue = x_ptr[ic];
    for (Ordinal kr = 0; kr < M; ++kr) {
      y[kr] += Aval[ic + kr * lda] * xvalue;
    }
  }
}

template <class StaticGraph, int N>
inline void spmv_serial(const double alpha, double *Avalues,
                        const StaticGraph &Agraph,
                        const double *x, double *y) {
  const Ordinal numBlockRows = Agraph.numRows();

  if (N == 1) {
    for (Ordinal i = 0; i < numBlockRows; ++i) {
      const auto jbeg  = Agraph.row_map[i];
      const auto jend  = Agraph.row_map[i + 1];
      double tmp = 0.0;
      for (Ordinal j = jbeg; j < jend; ++j) {
        const auto alpha_value1   = alpha * Avalues[j];
        const auto col_idx1 = Agraph.entries[j];
        const auto x_val1   = x[col_idx1];
        tmp += alpha_value1 * x_val1;
      }
      y[i] += tmp;
    }
    return;
  }

  std::array<double, details::bmax> tmp;
  auto Aval = Avalues;
  const Ordinal N2 = N * N;
  for (Ordinal iblock = 0; iblock < numBlockRows; ++iblock) {
    const auto jbeg       = Agraph.row_map[iblock];
    const auto jend       = Agraph.row_map[iblock + 1];
    const auto num_blocks = jend - jbeg;
    const auto lda = num_blocks * N;
    tmp.fill(0);
    for (Ordinal jb = 0; jb < num_blocks; ++jb) {
      const auto col_block = Agraph.entries[jb + jbeg];
      const auto xval_ptr  = x + N * col_block;
      auto Aval_ptr        = Aval + jb * N;
      //
      spmv_serial_gemv<N>(Aval_ptr, lda, xval_ptr, tmp);
    }
    //
    Aval = Aval + num_blocks * N2;
    //
    auto yvec = &y[iblock * N];
    for (Ordinal ii = 0; ii < N; ++ii) {
      yvec[ii] += alpha * tmp[ii];
    }
  }
}

//
// spMatVec_no_transpose: version for CPU execution spaces (RangePolicy or
// trivial serial impl used)
//
template <class AlphaType, class XVector, class BetaType, class YVector,
          typename std::enable_if<!KokkosKernels::Impl::kk_is_gpu_exec_space<
              typename YVector::execution_space>()>::type * = nullptr>
void spMatVec_no_transpose(const AlphaType &alpha, const bcrs_matrix_t_ &A, const XVector &x,
          const BetaType &beta, YVector &y)
{

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

    typedef
        typename bcrs_matrix_t_::device_type::execution_space execution_space;

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

#if defined(KOKKOS_ENABLE_SERIAL)
    if (std::is_same<execution_space, Kokkos::Serial>::value) {
      //
      const auto &A_graph = A.graph;
      //
      if (blockSize <= std::min<size_t>(8, details::bmax)) {
        switch (blockSize) {
          default:
          case 1:
            spmv_serial<decltype(A_graph), 1>(alpha, &A.values[0], A_graph,
                                              &x[0], &y[0]);
            break;
          case 2:
            spmv_serial<decltype(A_graph), 2>(alpha, &A.values[0], A_graph,
                                              &x[0], &y[0]);
            break;
          case 3:
            spmv_serial<decltype(A_graph), 3>(alpha, &A.values[0], A_graph,
                                              &x[0], &y[0]);
            break;
          case 4:
            spmv_serial<decltype(A_graph), 4>(alpha, &A.values[0], A_graph,
                                              &x[0], &y[0]);
            break;
          case 5:
            spmv_serial<decltype(A_graph), 5>(alpha, &A.values[0], A_graph,
                                              &x[0], &y[0]);
            break;
          case 6:
            spmv_serial<decltype(A_graph), 6>(alpha, &A.values[0], A_graph,
                                              &x[0], &y[0]);
            break;
          case 7:
            spmv_serial<decltype(A_graph), 7>(alpha, &A.values[0], A_graph,
                                              &x[0], &y[0]);
            break;
          case 8:
            spmv_serial<decltype(A_graph), 8>(alpha, &A.values[0], A_graph,
                                              &x[0], &y[0]);
            break;
          case 9:
            spmv_serial<decltype(A_graph), 9>(alpha, &A.values[0], A_graph,
                                              &x[0], &y[0]);
            break;
          case 10:
            spmv_serial<decltype(A_graph), 10>(alpha, &A.values[0], A_graph,
                                               &x[0], &y[0]);
            break;
          case 11:
            spmv_serial<decltype(A_graph), 11>(alpha, &A.values[0], A_graph,
                                               &x[0], &y[0]);
            break;
          case 12:
            spmv_serial<decltype(A_graph), 12>(alpha, &A.values[0], A_graph,
                                               &x[0], &y[0]);
            break;
        }
        return;
      }
      //
      // --- Basic approach for large block sizes
      //
      const Ordinal blockSize_squared = blockSize * blockSize;
      auto Aval = &A.values[0];
      for (Ordinal iblock = 0; iblock < numBlockRows; ++iblock) {
        const auto jbeg       = A_graph.row_map[iblock];
        const auto jend       = A_graph.row_map[iblock + 1];
        const auto num_blocks = jend - jbeg;
        auto yvec             = &y[iblock * blockSize];
        const auto lda = num_blocks * blockSize;
        for (Ordinal jb = 0; jb < num_blocks; ++jb) {
          const auto col_block = A_graph.entries[jb + jbeg];
          const auto xval_ptr  = &x[0] + blockSize * col_block;
          const auto Aval_ptr = Aval + jb * blockSize;
          for (Ordinal ic = 0; ic < blockSize; ++ic) {
            const auto xvalue = xval_ptr[ic];
            for (Ordinal kr = 0; kr < blockSize; ++kr) {
              yvec[kr] += Aval_ptr[ic + kr * lda] * xvalue;
            }
          }
        }
        Aval = Aval + num_blocks * blockSize_squared;
      }
      return;
    }
#endif

#ifdef KOKKOS_ENABLE_OPENMP
    //
    // This section terminates for the moment.
    // terminate called recursively
    //
    const auto &A_graph      = A.graph;
    const Ordinal blockSize2 = blockSize * blockSize;
    //
    KokkosKernels::Experimental::Controls controls;
    bool use_dynamic_schedule = false;  // Forces the use of a dynamic schedule
    bool use_static_schedule  = false;  // Forces the use of a static schedule
    if (controls.isParameter("schedule")) {
      if (controls.getParameter("schedule") == "dynamic") {
        use_dynamic_schedule = true;
      } else if (controls.getParameter("schedule") == "static") {
        use_static_schedule = true;
      }
    }
    //
    if (blockSize <= std::min<size_t>(8, details::bmax)) {
      switch (blockSize) {
        default:
        case 1: {
#pragma omp parallel for schedule(static)
          for (Ordinal iblock = 0; iblock < numBlockRows; ++iblock) {
            const auto jbeg = A.graph.row_map[iblock];
            const auto jend = A.graph.row_map[iblock + 1];
            for (Ordinal j = jbeg; j < jend; ++j) {
              const auto col_block = A.graph.entries[j];
              y[iblock] += alpha * A.values[j] * x[col_block];
            }
          }
          break;
        }
        case 2: {
#pragma omp parallel for schedule(static)
          for (Ordinal iblock = 0; iblock < numBlockRows; ++iblock) {
            const auto jbeg       = A.graph.row_map[iblock];
            int j                 = jbeg;
            const auto jend       = A.graph.row_map[iblock + 1];
            const auto num_blocks = jend - jbeg;
            const auto lda = blockSize * num_blocks;
            const auto Aval       = &A.values[blockSize2 * jbeg];
            std::array<Scalar, details::bmax> tmp;
            tmp.fill(0);
            for (Ordinal jb = 0; jb < num_blocks; ++jb, ++j) {
              const auto col_block = A.graph.entries[j];
              const auto xval_ptr  = &x[0] + blockSize * col_block;
              auto Aval_ptr        = Aval + jb * blockSize;
              //
              spmv_serial_gemv<2>(Aval_ptr, lda, xval_ptr, tmp);
            }
            //
            auto yvec = &y[iblock * blockSize];
            for (Ordinal ii = 0; ii < 2; ++ii) {
              yvec[ii] += alpha * tmp[ii];
            }
          }
          break;
        }
        case 3: {
#pragma omp parallel for schedule(static)
          for (Ordinal iblock = 0; iblock < numBlockRows; ++iblock) {
            const auto jbeg       = A.graph.row_map[iblock];
            int j                 = jbeg;
            const auto jend       = A.graph.row_map[iblock + 1];
            const auto num_blocks = jend - jbeg;
            const auto lda = blockSize * num_blocks;
            const auto Aval       = &A.values[blockSize2 * jbeg];
            std::array<Scalar, details::bmax> tmp;
            tmp.fill(0);
            for (Ordinal jb = 0; jb < num_blocks; ++jb, ++j) {
              const auto col_block = A.graph.entries[j];
              const auto xval_ptr  = &x[0] + blockSize * col_block;
              auto Aval_ptr        = Aval + jb * blockSize;
              //
              spmv_serial_gemv<3>(Aval_ptr, lda, xval_ptr, tmp);
            }
            //
            auto yvec = &y[iblock * blockSize];
            for (Ordinal ii = 0; ii < 3; ++ii) {
              yvec[ii] += alpha * tmp[ii];
            }
          }
          break;
        }
        case 4: {
#pragma omp parallel for schedule(static)
          for (Ordinal iblock = 0; iblock < numBlockRows; ++iblock) {
            const auto jbeg       = A.graph.row_map[iblock];
            int j                 = jbeg;
            const auto jend       = A.graph.row_map[iblock + 1];
            const auto num_blocks = jend - jbeg;
            const auto lda = blockSize * num_blocks;
            const auto Aval       = &A.values[blockSize2 * jbeg];
            std::array<Scalar, details::bmax> tmp;
            tmp.fill(0);
            for (Ordinal jb = 0; jb < num_blocks; ++jb, ++j) {
              const auto col_block = A.graph.entries[j];
              const auto xval_ptr  = &x[0] + blockSize * col_block;
              auto Aval_ptr        = Aval + jb * blockSize;
              //
              spmv_serial_gemv<4>(Aval_ptr, lda, xval_ptr, tmp);
            }
            //
            auto yvec = &y[iblock * blockSize];
            for (Ordinal ii = 0; ii < 4; ++ii) {
              yvec[ii] += alpha * tmp[ii];
            }
          }
          break;
        }
        case 5: {
#pragma omp parallel for schedule(static)
          for (Ordinal iblock = 0; iblock < numBlockRows; ++iblock) {
            const auto jbeg       = A.graph.row_map[iblock];
            int j                 = jbeg;
            const auto jend       = A.graph.row_map[iblock + 1];
            const auto num_blocks = jend - jbeg;
            const auto lda = blockSize * num_blocks;
            const auto Aval       = &A.values[blockSize2 * jbeg];
            std::array<Scalar, details::bmax> tmp;
            tmp.fill(0);
            for (Ordinal jb = 0; jb < num_blocks; ++jb, ++j) {
              const auto col_block = A.graph.entries[j];
              const auto xval_ptr  = &x[0] + blockSize * col_block;
              auto Aval_ptr        = Aval + jb * blockSize;
              //
              spmv_serial_gemv<5>(Aval_ptr, lda, xval_ptr, tmp);
            }
            //
            auto yvec = &y[iblock * blockSize];
            for (Ordinal ii = 0; ii < 5; ++ii) {
              yvec[ii] += alpha * tmp[ii];
            }
          }
          break;
        }
        case 6: {
#pragma omp parallel for schedule(static)
          for (Ordinal iblock = 0; iblock < numBlockRows; ++iblock) {
            const auto jbeg       = A.graph.row_map[iblock];
            int j                 = jbeg;
            const auto jend       = A.graph.row_map[iblock + 1];
            const auto num_blocks = jend - jbeg;
            const auto lda = blockSize * num_blocks;
            const auto Aval       = &A.values[blockSize2 * jbeg];
            std::array<Scalar, details::bmax> tmp;
            tmp.fill(0);
            for (Ordinal jb = 0; jb < num_blocks; ++jb, ++j) {
              const auto col_block = A.graph.entries[j];
              const auto xval_ptr  = &x[0] + blockSize * col_block;
              auto Aval_ptr        = Aval + jb * blockSize;
              //
              spmv_serial_gemv<6>(Aval_ptr, lda, xval_ptr, tmp);
            }
            //
            auto yvec = &y[iblock * blockSize];
            for (Ordinal ii = 0; ii < 6; ++ii) {
              yvec[ii] += alpha * tmp[ii];
            }
          }
          break;
        }
        case 7: {
#pragma omp parallel for schedule(static)
          for (Ordinal iblock = 0; iblock < numBlockRows; ++iblock) {
            const auto jbeg       = A.graph.row_map[iblock];
            int j                 = jbeg;
            const auto jend       = A.graph.row_map[iblock + 1];
            const auto num_blocks = jend - jbeg;
            const auto lda = blockSize * num_blocks;
            const auto Aval       = &A.values[blockSize2 * jbeg];
            std::array<Scalar, details::bmax> tmp;
            tmp.fill(0);
            for (Ordinal jb = 0; jb < num_blocks; ++jb, ++j) {
              const auto col_block = A.graph.entries[j];
              const auto xval_ptr  = &x[0] + blockSize * col_block;
              auto Aval_ptr        = Aval + jb * blockSize;
              //
              spmv_serial_gemv<7>(Aval_ptr, lda, xval_ptr, tmp);
            }
            //
            auto yvec = &y[iblock * blockSize];
            for (Ordinal ii = 0; ii < 7; ++ii) {
              yvec[ii] += alpha * tmp[ii];
            }
          }
          break;
        }
        case 8: {
#pragma omp parallel for schedule(static)
          for (Ordinal iblock = 0; iblock < numBlockRows; ++iblock) {
            const auto jbeg       = A.graph.row_map[iblock];
            int j                 = jbeg;
            const auto jend       = A.graph.row_map[iblock + 1];
            const auto num_blocks = jend - jbeg;
            const auto lda = blockSize * num_blocks;
            const auto Aval       = &A.values[blockSize2 * jbeg];
            std::array<Scalar, details::bmax> tmp;
            tmp.fill(0);
            for (Ordinal jb = 0; jb < num_blocks; ++jb, ++j) {
              const auto col_block = A.graph.entries[j];
              const auto xval_ptr  = &x[0] + blockSize * col_block;
              auto Aval_ptr        = Aval + jb * blockSize;
              //
              spmv_serial_gemv<8>(Aval_ptr, lda, xval_ptr, tmp);
            }
            //
            auto yvec = &y[iblock * blockSize];
            for (Ordinal ii = 0; ii < 8; ++ii) {
              yvec[ii] += alpha * tmp[ii];
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
        const auto num_blocks = jend - jbeg;
        const auto lda = blockSize * num_blocks;
        const auto Aval       = &A.values[blockSize2 * jbeg];
        auto yvec             = &y[iblock * blockSize];
        for (Ordinal jb = 0; jb < num_blocks; ++jb) {
          const auto col_block = A.graph.entries[jb + jbeg];
          const auto xval_ptr  = &x[0] + blockSize * col_block;
          const auto Aval_ptr = Aval + jb * blockSize;
          for (Ordinal ic = 0; ic < blockSize; ++ic) {
            const auto xvalue = xval_ptr[ic];
            for (Ordinal kr = 0; kr < blockSize; ++kr) {
              yvec[kr] += Aval_ptr[ic + kr * lda] * xvalue;
            }
          }
        }
      }
    }
#endif
    return;

}


template <int M>
inline void spmv_transpose_gemv(const Scalar alpha, Scalar *Aval,
                                       const Ordinal lda,
                             const Ordinal xrow, const Scalar *x_ptr,
                             Scalar *y) {
  for (Ordinal ic = 0; ic < xrow; ++ic) {
    for (Ordinal kr = 0; kr < M; ++kr) {
      const auto alpha_value = alpha * Aval[ic + kr * lda];
      Kokkos::atomic_add (&y[ic],
                          static_cast<Scalar>(alpha_value * x_ptr[kr]));
    }
  }
}


template <class StaticGraph, int N>
inline void spmv_transpose_serial(const Scalar alpha, Scalar *Avalues,
                        const StaticGraph &Agraph,
                        const Scalar *x, Scalar *y, const Ordinal ldy) {

  const Ordinal numBlockRows = Agraph.numRows();

  if (N == 1) {
    for (Ordinal i = 0; i < numBlockRows; ++i) {
      const auto jbeg  = Agraph.row_map[i];
      const auto jend  = Agraph.row_map[i + 1];
      for (Ordinal j = jbeg; j < jend; ++j) {
        const auto alpha_value = alpha * Avalues[j];
        const auto col_idx1 = Agraph.entries[j];
        y[col_idx1] += alpha_value * x[i];
      }
    }
    return;
  }

  const auto blockSize2 = N * N;
  for (Ordinal iblock = 0; iblock < numBlockRows; ++iblock) {
    const auto jbeg       = Agraph.row_map[iblock];
    const auto jend       = Agraph.row_map[iblock + 1];
    const auto num_blocks = jend - jbeg;
    const auto lda = num_blocks * N;
    const auto Aval       = &Avalues[blockSize2 * jbeg];
    const auto xval_ptr = &x[iblock * N];
    for (Ordinal jb = 0; jb < num_blocks; ++jb) {
      const auto col_block = Agraph.entries[jb + jbeg];
      auto yvec  = &y[N * col_block];
      auto Aval_ptr        = Aval + jb * N;
      //
      spmv_transpose_gemv<N>(alpha, Aval_ptr, lda, N, xval_ptr, yvec);
    }
  }
}


//
// spMatVec_transpose: version for CPU execution spaces (RangePolicy or
// trivial serial impl used)
//
template <class AlphaType, class XVector, class BetaType, class YVector,
    typename std::enable_if<!KokkosKernels::Impl::kk_is_gpu_exec_space<
        typename YVector::execution_space>()>::type * = nullptr>
void spMatVec_transpose(const AlphaType &alpha, const bcrs_matrix_t_ &A, const XVector &x,
                           const BetaType &beta, YVector &y)
{

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

  typedef
  typename bcrs_matrix_t_::device_type::execution_space execution_space;

  YVector_Internal y_i = y;

  // This is required to maintain semantics of KokkosKernels native SpMV:
  // if y contains NaN but beta = 0, the result y should be filled with 0.
  // For example, this is useful for passing in uninitialized y and beta=0.
  if (beta == Kokkos::ArithTraits<BetaType>::zero())
    Kokkos::deep_copy(y_i, Kokkos::ArithTraits<BetaType>::zero());
  else
    KokkosBlas::scal(y_i, beta, y_i);

  //
  // Treat the case y <- alpha * A^T * x + beta * y
  //

  XVector_Internal x_i       = x;
  const Ordinal blockSize    = A.blockDim();
  const Ordinal numBlockRows = A.numRows();
  //
  const bool conjugate = false;
  //

#if defined(KOKKOS_ENABLE_SERIAL)
  if (std::is_same<execution_space, Kokkos::Serial>::value) {
      //
      const auto &A_graph = A.graph;
      //
      if (blockSize <= std::min<size_t>(8, details::bmax)) {
        switch (blockSize) {
          default:
          case 1:
            spmv_transpose_serial<decltype(A_graph), 1>(alpha, &A.values[0], A_graph,
                                                        &x[0], &y[0], blockSize);
            break;
          case 2:
            spmv_transpose_serial<decltype(A_graph), 2>(alpha, &A.values[0], A_graph,
                                                        &x[0], &y[0], blockSize);
            break;
          case 3:
            spmv_transpose_serial<decltype(A_graph), 3>(alpha, &A.values[0], A_graph,
                                                        &x[0], &y[0], blockSize);
            break;
          case 4:
            spmv_transpose_serial<decltype(A_graph), 4>(alpha, &A.values[0], A_graph,
                                                        &x[0], &y[0], blockSize);
            break;
          case 5:
            spmv_transpose_serial<decltype(A_graph), 5>(alpha, &A.values[0], A_graph,
                                                        &x[0], &y[0], blockSize);
            break;
          case 6:
            spmv_transpose_serial<decltype(A_graph), 6>(alpha, &A.values[0], A_graph,
                                                        &x[0], &y[0], blockSize);
            break;
          case 7:
            spmv_transpose_serial<decltype(A_graph), 7>(alpha, &A.values[0], A_graph,
                                                        &x[0], &y[0], blockSize);
            break;
          case 8:
            spmv_transpose_serial<decltype(A_graph), 8>(alpha, &A.values[0], A_graph,
                                                        &x[0], &y[0], blockSize);
            break;
          case 9:
            spmv_transpose_serial<decltype(A_graph), 9>(alpha, &A.values[0], A_graph,
                                                        &x[0], &y[0], blockSize);
            break;
          case 10:
            spmv_transpose_serial<decltype(A_graph), 10>(alpha, &A.values[0], A_graph,
                                                         &x[0], &y[0], blockSize);
            break;
          case 11:
            spmv_transpose_serial<decltype(A_graph), 11>(alpha, &A.values[0], A_graph,
                                                         &x[0], &y[0], blockSize);
            break;
          case 12:
            spmv_transpose_serial<decltype(A_graph), 12>(alpha, &A.values[0], A_graph,
                                                         &x[0], &y[0], blockSize);
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
        const auto num_blocks = jend - jbeg;
        const auto lda = num_blocks * blockSize;
        const auto xval_ptr  = &x[iblock * blockSize];
        const auto Aval = &A.values[jbeg * blockSize2];
        for (Ordinal jb = 0; jb < num_blocks; ++jb) {
          auto yvec = &y[blockSize * A_graph.entries[jb + jbeg]];
          const auto Aval_ptr = Aval + jb * blockSize;
          for (Ordinal ic = 0; ic < blockSize; ++ic) {
            for (Ordinal kr = 0; kr < blockSize; ++kr) {
              yvec[ic] += Aval_ptr[ic + kr * lda] * xval_ptr[kr];
            }
          }
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
    const auto &A_graph      = A.graph;
    const Ordinal blockSize2 = blockSize * blockSize;
    //
    KokkosKernels::Experimental::Controls controls;
    bool use_dynamic_schedule = false;  // Forces the use of a dynamic schedule
    bool use_static_schedule  = false;  // Forces the use of a static schedule
    if (controls.isParameter("schedule")) {
      if (controls.getParameter("schedule") == "dynamic") {
        use_dynamic_schedule = true;
      } else if (controls.getParameter("schedule") == "static") {
        use_static_schedule = true;
      }
    }
    //
    if (blockSize <= std::min<size_t>(8, details::bmax)) {
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
              Kokkos::atomic_add (&y[col_block],
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
            const auto num_blocks = jend - jbeg;
            const auto lda = num_blocks * blockSize;
            const auto xval_ptr  = &x[iblock * blockSize];
            const auto Aval = &A.values[blockSize2 * jbeg];
            for (Ordinal jb = 0; jb < num_blocks; ++jb) {
              const auto col_block = A.graph.entries[jb + jbeg];
              auto yvec = &y[blockSize * col_block];
              const auto Aval_ptr = Aval + jb * blockSize;
              spmv_transpose_gemv<2>(alpha, Aval_ptr, lda, blockSize, xval_ptr, yvec);
            }
          }
          break;
        }
        case 3: {
#pragma omp parallel for schedule(static)
          for (Ordinal iblock = 0; iblock < numBlockRows; ++iblock) {
            const auto jbeg       = A_graph.row_map[iblock];
            const auto jend       = A_graph.row_map[iblock + 1];
            const auto num_blocks = jend - jbeg;
            const auto lda = num_blocks * blockSize;
            const auto xval_ptr  = &x[iblock * blockSize];
            const auto Aval = &A.values[blockSize2 * jbeg];
            for (Ordinal jb = 0; jb < num_blocks; ++jb) {
              const auto col_block = A.graph.entries[jb + jbeg];
              auto yvec = &y[blockSize * col_block];
              const auto Aval_ptr = Aval + jb * blockSize;
              spmv_transpose_gemv<3>(alpha, Aval_ptr, lda, blockSize, xval_ptr, yvec);
            }
          }
          break;
        }
        case 4: {
#pragma omp parallel for schedule(static)
          for (Ordinal iblock = 0; iblock < numBlockRows; ++iblock) {
            const auto jbeg       = A_graph.row_map[iblock];
            const auto jend       = A_graph.row_map[iblock + 1];
            const auto num_blocks = jend - jbeg;
            const auto lda = num_blocks * blockSize;
            const auto xval_ptr  = &x[iblock * blockSize];
            const auto Aval = &A.values[blockSize2 * jbeg];
            for (Ordinal jb = 0; jb < num_blocks; ++jb) {
              const auto col_block = A.graph.entries[jb + jbeg];
              auto yvec = &y[blockSize * col_block];
              const auto Aval_ptr = Aval + jb * blockSize;
              spmv_transpose_gemv<4>(alpha, Aval_ptr, lda, blockSize, xval_ptr, yvec);
            }
          }
          break;
        }
        case 5: {
#pragma omp parallel for schedule(static)
          for (Ordinal iblock = 0; iblock < numBlockRows; ++iblock) {
            const auto jbeg       = A_graph.row_map[iblock];
            const auto jend       = A_graph.row_map[iblock + 1];
            const auto num_blocks = jend - jbeg;
            const auto lda = num_blocks * blockSize;
            const auto xval_ptr  = &x[iblock * blockSize];
            const auto Aval = &A.values[blockSize2 * jbeg];
            for (Ordinal jb = 0; jb < num_blocks; ++jb) {
              const auto col_block = A.graph.entries[jb + jbeg];
              auto yvec = &y[blockSize * col_block];
              const auto Aval_ptr = Aval + jb * blockSize;
              spmv_transpose_gemv<5>(alpha, Aval_ptr, lda, blockSize, xval_ptr, yvec);
            }
          }
          break;
        }
        case 6: {
#pragma omp parallel for schedule(static)
          for (Ordinal iblock = 0; iblock < numBlockRows; ++iblock) {
            const auto jbeg       = A_graph.row_map[iblock];
            const auto jend       = A_graph.row_map[iblock + 1];
            const auto num_blocks = jend - jbeg;
            const auto lda = num_blocks * blockSize;
            const auto xval_ptr  = &x[iblock * blockSize];
            const auto Aval = &A.values[blockSize2 * jbeg];
            for (Ordinal jb = 0; jb < num_blocks; ++jb) {
              const auto col_block = A.graph.entries[jb + jbeg];
              auto yvec = &y[blockSize * col_block];
              const auto Aval_ptr = Aval + jb * blockSize;
              spmv_transpose_gemv<6>(alpha, Aval_ptr, lda, blockSize, xval_ptr, yvec);
            }
          }
          break;
        }
        case 7: {
#pragma omp parallel for schedule(static)
          for (Ordinal iblock = 0; iblock < numBlockRows; ++iblock) {
            const auto jbeg       = A_graph.row_map[iblock];
            const auto jend       = A_graph.row_map[iblock + 1];
            const auto num_blocks = jend - jbeg;
            const auto lda = num_blocks * blockSize;
            const auto xval_ptr  = &x[iblock * blockSize];
            const auto Aval = &A.values[blockSize2 * jbeg];
            for (Ordinal jb = 0; jb < num_blocks; ++jb) {
              const auto col_block = A.graph.entries[jb + jbeg];
              auto yvec = &y[blockSize * col_block];
              const auto Aval_ptr = Aval + jb * blockSize;
              spmv_transpose_gemv<7>(alpha, Aval_ptr, lda, blockSize, xval_ptr, yvec);
            }
          }
          break;
        }
        case 8: {
#pragma omp parallel for schedule(static)
          for (Ordinal iblock = 0; iblock < numBlockRows; ++iblock) {
            const auto jbeg       = A_graph.row_map[iblock];
            const auto jend       = A_graph.row_map[iblock + 1];
            const auto num_blocks = jend - jbeg;
            const auto lda = num_blocks * blockSize;
            const auto xval_ptr  = &x[iblock * blockSize];
            const auto Aval = &A.values[blockSize2 * jbeg];
            for (Ordinal jb = 0; jb < num_blocks; ++jb) {
              const auto col_block = A.graph.entries[jb + jbeg];
              auto yvec = &y[blockSize * col_block];
              const auto Aval_ptr = Aval + jb * blockSize;
              spmv_transpose_gemv<8>(alpha, Aval_ptr, lda, blockSize, xval_ptr, yvec);
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
        const auto num_blocks = jend - jbeg;
        const auto xvec       = &x[iblock * blockSize];
        const auto lda = blockSize * num_blocks;
        const auto Aval       = &A.values[blockSize2 * jbeg];
        for (Ordinal jb = 0; jb < num_blocks; ++jb) {
          const auto col_block = A.graph.entries[jb + jbeg];
          auto yvec  = &y[blockSize * col_block];
          const auto Aval_ptr = Aval + jb * blockSize;
          for (Ordinal ic = 0; ic < blockSize; ++ic) {
            const auto xvalue = xvec[ic];
            for (Ordinal kr = 0; kr < blockSize; ++kr) {
              Kokkos::atomic_add (&yvec[kr],
                                 static_cast<Scalar>(alpha * Aval_ptr[kr + ic * lda] * xvalue));
            }
          }
        }
      }
    }
#endif
  return;

}


//
// spmv: version for CPU execution spaces (RangePolicy or
// trivial serial impl used)
//
/*
template <class AlphaType, class XVector, class BetaType, class YVector,
    typename std::enable_if<!KokkosKernels::Impl::kk_is_gpu_exec_space<
        typename YVector::execution_space>()>::type * = nullptr>
void spmv(const char fOp,
          const AlphaType &alpha, const bcrs_matrix_t_ &A, const XVector &x,
          const BetaType &beta, YVector &y) {
  //
  // Call single-vector version if appropriate
  //
  if (x.extent(1) == 1) {
    if (fOp == 'N') {
      return spMatVec_no_transpose(alpha, A, x, beta, y);
    }
    else if (fOp == 'T') {
      return spMatVec_transpose(alpha, A, x, beta, y);
    }
  }
  else {
    if (fOp == 'N') {
//      return spMatMultiVec_no_transpose(alpha, A, x, beta, y);
    }
    else if (fOp == 'T') {
//      return spMatMultiVec_transpose(alpha, A, x, beta, y);
    }
  }
}
 */


/////////////////////////////////////////////////////

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
  if ((mode[0] == KokkosSparse::NoTranspose[0]) || (mode[0] == KokkosSparse::Conjugate[0])) {
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

/////////////////////////////////////////////////////

template< typename ScalarType , typename OrdinalType, class Device, class MemoryTraits, typename SizeType,
    class AlphaType, class XVector, class BetaType, class YVector >
void spmv(KokkosKernels::Experimental::Controls controls,
          const char mode[],
          const AlphaType& alpha,
          const KokkosSparse::Experimental::BlockCrsMatrix< ScalarType, OrdinalType, Device, MemoryTraits, SizeType>& A,
          const XVector& X,
          const BetaType& beta,
          const YVector& Y) {
  //
  details::verifyArguments(mode, A, X, Y);
  //
  if(alpha == Kokkos::ArithTraits<AlphaType>::zero() ||
      A.numRows() == 0 || A.numCols() == 0 || A.nnz() == 0)
  {
    //This is required to maintain semantics of KokkosKernels native SpMV:
    //if y contains NaN but beta = 0, the result y should be filled with 0.
    //For example, this is useful for passing in uninitialized y and beta=0.
    if(beta == Kokkos::ArithTraits<BetaType>::zero())
      Kokkos::deep_copy(Y, Kokkos::ArithTraits<BetaType>::zero());
    else
      KokkosBlas::scal(Y, beta, Y);
    return;
  }
  //
  //Whether to call KokkosKernel's native implementation, even if a TPL impl is available
  bool useFallback = controls.isParameter("algorithm") && controls.getParameter("algorithm") == "native";
  //
  if (X.extent(1) == 1) {
    if (mode[0] == KokkosSparse::NoTranspose[0]) {
      return spMatVec_no_transpose(alpha, A, X, beta, Y);
    }
    else if (mode[0] == KokkosSparse::Transpose[0]) {
      return spMatVec_transpose(alpha, A, X, beta, Y);
    }
  }
  else {
    if (mode[0] == KokkosSparse::NoTranspose[0]) {
      //      return spMatMultiVec_no_transpose(alpha, A, X, beta, Y);
    }
    else if (mode[0] == KokkosSparse::Transpose[0]) {
      //      return spMatMultiVec_transpose(alpha, A, X, beta, Y);
    }
  }
}



/////////////////////////////////////////////////////


/// \brief Generate a random multi-vector
///
/// \param numRows Number of rows
/// \param numCols Number of columns
/// \return Vector
MultiVector_Internal make_lhs(const int numRows, const int numCols) {
  MultiVector_Internal X("lhs", numRows, numCols);
  for (Ordinal ir = 0; ir < numRows; ++ir) {
    for (Ordinal jc = 0; jc < numCols; ++jc) {
      X(ir, jc) = std::rand() / static_cast<Scalar>(RAND_MAX);
    }
  }
  return X;
}

/// \brief Generate a random vector
///
/// \param numRows Number of rows
/// \return Vector
typename values_type::non_const_type make_lhs(const int numRows) {
  typename values_type::non_const_type x("lhs", numRows);
  for (Ordinal ir = 0; ir < numRows; ++ir)
    x(ir) = std::rand() / static_cast<Scalar>(RAND_MAX);
  return x;
}


template <typename mtx_t>
std::chrono::duration<double> measure(const char fOp[],
                                      const mtx_t &myMatrix, const Scalar alpha,
                                      const Scalar beta, const int repeat) {
  const Ordinal numRows = myMatrix.numRows();

  auto const x = make_lhs(numRows);
  typename values_type::non_const_type y("rhs", numRows);

  std::chrono::duration<double> dt;
  if (fOp[0] == KokkosSparse::NoTranspose[0]) {
    auto tBegin = std::chrono::high_resolution_clock::now();
    for (int ir = 0; ir < repeat; ++ir) {
      KokkosSparse::spmv("N", alpha, myMatrix, x, beta, y);
    }
    auto tEnd = std::chrono::high_resolution_clock::now();
    dt = tEnd - tBegin;
  }
  else if (fOp[0] == KokkosSparse::Transpose[0]){
    auto tBegin = std::chrono::high_resolution_clock::now();
    for (int ir = 0; ir < repeat; ++ir) {
      KokkosSparse::spmv("T", alpha, myMatrix, x, beta, y);
    }
    auto tEnd = std::chrono::high_resolution_clock::now();
    dt = tEnd - tBegin;
  }

  return dt;
}

template <typename bmtx_t>
std::chrono::duration<double> measure_block(
    const char fOp[],
    const bmtx_t &myBlockMatrix,
    const Scalar alpha, const Scalar beta, const int repeat) {
  auto const numRows = myBlockMatrix.numRows() * myBlockMatrix.blockDim();
  auto const x       = make_lhs(numRows);
  typename values_type::non_const_type y("rhs", numRows);
  KokkosKernels::Experimental::Controls controls;

  std::chrono::duration<double> dt;
  auto tBegin = std::chrono::high_resolution_clock::now();
  for (int ir = 0; ir < repeat; ++ir) {
    details::spmv(controls, fOp, alpha, myBlockMatrix, x, beta, y);
  }
  auto tEnd = std::chrono::high_resolution_clock::now();
  dt = tEnd - tBegin;

  return dt;
}

template <typename mtx_t>
std::vector<Ordinal> build_entry_ptr(const mtx_t &myBlockMatrix) {
  // Build pointer to entry values
  const Ordinal blockSize = myBlockMatrix.blockDim();
  const Ordinal numBlocks = myBlockMatrix.numRows();
  std::vector<Ordinal> val_entries_ptr(numBlocks + 1, 0);
  for (Ordinal ir = 0; ir < numBlocks; ++ir) {
    const auto jbeg = myBlockMatrix.graph.row_map[ir];
    const auto jend = myBlockMatrix.graph.row_map[ir + 1];
    val_entries_ptr[ir + 1] =
        val_entries_ptr[ir] + blockSize * blockSize * (jend - jbeg);
  }
  return val_entries_ptr;
}

template <typename mtx_t, typename bmtx_t>
void compare(const char fOp[],
             const mtx_t &myMatrix, const bmtx_t &myBlockMatrix,
             const Scalar alpha,
             const Scalar beta, double &error, double &maxNorm) {
  error   = 0.0;
  maxNorm = 0.0;

  const int numRows = myMatrix.numRows();
  auto const x      = make_lhs(numRows);
  typename values_type::non_const_type y("rhs", numRows);
  typename values_type::non_const_type yref("ref", numRows);
  KokkosKernels::Experimental::Controls controls;

  if (fOp[0] == KokkosSparse::NoTranspose[0]) {
    KokkosSparse::spmv("N", alpha, myMatrix, x, beta, yref);
  }
  else if (fOp[0] == KokkosSparse::Transpose[0]) {
    KokkosSparse::spmv("T", alpha, myMatrix, x, beta, yref);
  }
  details::spmv(controls, fOp, alpha, myBlockMatrix, x, beta, y);

  for (Ordinal ir = 0, numRows = y.size(); ir < numRows; ++ir) {
    /*
    if (ir < 16) {
      std::cout << '\t' << ir << '\t' << x(ir) << '\t' << yref(ir)
      << '\t' << y(ir) << std::endl;
    }
    */
    error   = std::max<double>(error, std::abs(yref(ir) - y(ir)));
    maxNorm = std::max<double>(maxNorm, std::abs(yref(ir)));
  }
}

template <typename mtx_t>
void test_matrix(const mtx_t &myMatrix, const int blockSize, const int repeat) {
  const Scalar alpha = details::SC_ONE;
  const Scalar beta  = details::SC_ZERO;

  auto const numRows = myMatrix.numRows();

  //
  // Use BlockCrsMatrix format
  //
  bcrs_matrix_t_ myBlockMatrix = to_block_crs_matrix(myMatrix, blockSize);

  double error = 0.0, maxNorm = 0.0;

  //
  // Assess y <- A * x
  //
  compare("N", myMatrix, myBlockMatrix, alpha, beta, error, maxNorm);

  std::cout << " Error BlockCrsMatrix " << error << " maxNorm " << maxNorm
            << "\n";
  std::cout << " ------------------------ \n";

  //---

  std::chrono::duration<double> dt_crs = measure("N", myMatrix, alpha, beta, repeat);

  std::cout << " Total time for Crs Mat-Vec " << dt_crs.count() << " Avg. "
            << dt_crs.count() / static_cast<double>(repeat);
  std::cout << " Flops (mult only) "
            << myMatrix.nnz() * static_cast<double>(repeat / dt_crs.count())
            << "\n";
  std::cout << " ------------------------ \n";

  //---
  std::chrono::duration<double> dt_bcrs =
      measure_block("N", myBlockMatrix, alpha, beta, repeat);

  std::cout << " Total time for BlockCrs Mat-Vec " << dt_bcrs.count()
            << " Avg. " << dt_bcrs.count() / static_cast<double>(repeat);
  std::cout << " Flops (mult only) "
            << myMatrix.nnz() * static_cast<double>(repeat / dt_bcrs.count());
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

  //
  // Assess y <- A * x
  //
  compare("T", myMatrix, myBlockMatrix, alpha, beta, error, maxNorm);

  std::cout << " Error Transpose BlockCrsMatrix " << error << " maxNorm " << maxNorm
            << "\n";
  std::cout << " ------------------------ \n";

  //---

  dt_crs = measure("T", myMatrix, alpha, beta, repeat);

  std::cout << " Total time for Crs Mat^T-Vec " << dt_crs.count() << " Avg. "
            << dt_crs.count() / static_cast<double>(repeat);
  std::cout << " Flops (mult only) "
            << myMatrix.nnz() * static_cast<double>(repeat / dt_crs.count())
            << "\n";
  std::cout << " ------------------------ \n";

  //---
  dt_bcrs = measure_block("T", myBlockMatrix, alpha, beta, repeat);

  std::cout << " Total time for BlockCrs Mat^T-Vec " << dt_bcrs.count()
            << " Avg. " << dt_bcrs.count() / static_cast<double>(repeat);
  std::cout << " Flops (mult only) "
            << myMatrix.nnz() * static_cast<double>(repeat / dt_bcrs.count());
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

int test_random(const int repeat = 1024, const int minBlockSize = 1,
                const int maxBlockSize = 12) {
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
  Kokkos::View<Ordinal *[3], Kokkos::HostSpace> mat_structure(
      "Matrix Structure", 2);
  mat_structure(0, 0) = 196;  // Request 150 grid point in 'x' direction
  mat_structure(0, 1) = 0;    // Add BC to the left
  mat_structure(0, 2) = 0;    // Add BC to the right
  mat_structure(1, 0) = 212;  // Request 140 grid point in 'y' direction
  mat_structure(1, 1) = 0;    // Add BC to the bottom
  mat_structure(1, 2) = 0;    // Add BC to the top

  for (int blockSize = minBlockSize; blockSize <= maxBlockSize; ++blockSize) {
    std::vector<int> mat_rowmap, mat_colidx;
    std::vector<double> mat_val;

    crs_matrix_t_ myMatrix = details::generate_crs_matrix(
        "FD", mat_structure, blockSize, mat_rowmap, mat_colidx, mat_val);

    std::cout << " ======================== \n";
    std::cout << " Block Size " << blockSize;
    std::cout << " Matrix Size " << myMatrix.numRows() << " nnz "
              << myMatrix.nnz() << "\n";

    test_matrix(myMatrix, blockSize, repeat);
  }
  return return_value;
}

int test_samples(const int repeat = 3000) {
  int return_value = 0;

  srand(17312837);

  const std::vector<std::tuple<const char *, int> > SAMPLES{
      // std::tuple(char* fileName, int blockSize)
      std::make_tuple(
          "GT01R.mtx", 5)  // ID:2335	Fluorem	GT01R	7980	7980	430909	1	0
                           // 1	0	0.8811455350661695	9.457852263618717e-06
                           // computational fluid dynamics problem	430909
      ,
      //std::make_tuple("raefsky4.mtx", 3)  // ID:817	Simon	raefsky4	19779
                                          // 19779	1316789	1	0	1	1	1
                                          // 1	structural problem	1328611
      //    , std::make_tuple("bmw7st_1.mtx", 6) // ID:1253	GHS_psdef
      //    bmw7st_1	141347	141347	7318399	1	0	1	1	1
      //    1	structural problem	7339667
      //    , std::make_tuple("pwtk.mtx", 6) // ID:369	Boeing	pwtk 217918
      //    217918	11524432	1	0	1	1	1	1	structural
      //    problem	11634424
      //,
      //std::make_tuple("RM07R.mtx",
      //                7)  // ID:2337	Fluorem	RM07R	381689	381689 37464962
                          // 1	0	1	0
                          // 0.9261667922354103	4.260681089287885e-06
                          // computational fluid dynamics problem	37464962
      //,
      std::make_tuple("audikw_1.mtx",
                      3)  // ID:1252 GHS_psdef	audikw_1	943695	943695
                          // 77651847	1	0	1	1	1	1	structural
                          // problem	77651847
  };

  // Loop over sample matrix files
  std::for_each(SAMPLES.begin(), SAMPLES.end(), [=](auto const &sample) {
    const char *fileName = std::get<0>(sample);
    const int blockSize  = std::get<1>(sample);
    auto myMatrix =
        KokkosKernels::Impl::read_kokkos_crst_matrix<crs_matrix_t_>(fileName);

    std::cout << " ======================== \n";
    std::cout << " Sample: '" << fileName << "', Block Size " << blockSize;
    std::cout << " Matrix Size " << myMatrix.numCols() << " x "
              << myMatrix.numRows() << ", nnz " << myMatrix.nnz() << "\n";

    test_matrix(myMatrix, blockSize, repeat);
  });
  return return_value;
}

}  // namespace details

#define TEST_RANDOM_BSPMV

int main() {
  Kokkos::initialize();

#ifdef TEST_RANDOM_BSPMV
  int return_value = details::test_random();
#else
  int return_value = details::test_samples();
#endif

  Kokkos::finalize();

  return return_value;
}
