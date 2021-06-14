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

using MultiVector_Internal =
    typename Kokkos::View<Scalar **, Layout, device_type>;

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

  for (Ordinal ir = 0; ir < mat_crs.numRows(); ++ir) {
    const auto iblock = ir / blockSize;
    const auto ilocal = ir % blockSize;
    Ordinal lda       = blockSize * (rows[iblock + 1] - rows[iblock]);
    for (Ordinal jk = mat_crs.graph.row_map(ir);
         jk < mat_crs.graph.row_map(ir + 1); ++jk) {
      const auto jc     = mat_crs.graph.entries(jk);
      const auto jblock = jc / blockSize;
      const auto jlocal = jc % blockSize;
      for (Ordinal jkb = rows[iblock]; jkb < rows[iblock + 1]; ++jkb) {
        if (cols(jkb) == jblock) {
          Ordinal shift = rows[iblock] * blockSize * blockSize +
                          blockSize * (jkb - rows[iblock]);
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
          const BetaType &beta, YVector &y) {
  std::cerr << " GPU implementation is not complete !!! \n";
  assert(0 > 1);
  ///
  ///  Check spmv and Tpetra::BlockCrsMatrix
  ///
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
std::chrono::duration<double> measure(const char fOp[], const mtx_t &myMatrix,
                                      const Scalar alpha, const Scalar beta,
                                      const int repeat) {
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
    dt        = tEnd - tBegin;
  } else if (fOp[0] == KokkosSparse::Transpose[0]) {
    auto tBegin = std::chrono::high_resolution_clock::now();
    for (int ir = 0; ir < repeat; ++ir) {
      KokkosSparse::spmv("T", alpha, myMatrix, x, beta, y);
    }
    auto tEnd = std::chrono::high_resolution_clock::now();
    dt        = tEnd - tBegin;
  }

  return dt;
}

template <typename bmtx_t>
std::chrono::duration<double> measure_block(const char fOp[],
                                            const bmtx_t &myBlockMatrix,
                                            const Scalar alpha,
                                            const Scalar beta,
                                            const int repeat) {
  auto const numRows = myBlockMatrix.numRows() * myBlockMatrix.blockDim();
  auto const x       = make_lhs(numRows);
  typename values_type::non_const_type y("rhs", numRows);
  KokkosKernels::Experimental::Controls controls;

  std::chrono::duration<double> dt;
  auto tBegin = std::chrono::high_resolution_clock::now();
  for (int ir = 0; ir < repeat; ++ir) {
    KokkosSparse::spmv(controls, fOp, alpha, myBlockMatrix, x, beta, y);
  }
  auto tEnd = std::chrono::high_resolution_clock::now();
  dt        = tEnd - tBegin;

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
void compare(const char fOp[], const mtx_t &myMatrix,
             const bmtx_t &myBlockMatrix, const Scalar alpha, const Scalar beta,
             double &error, double &maxNorm) {
  error   = 0.0;
  maxNorm = 0.0;

  const int numRows = myMatrix.numRows();
  auto const x      = make_lhs(numRows);
  typename values_type::non_const_type y("rhs", numRows);
  typename values_type::non_const_type yref("ref", numRows);
  KokkosKernels::Experimental::Controls controls;

  if (fOp[0] == KokkosSparse::NoTranspose[0]) {
    KokkosSparse::spmv("N", alpha, myMatrix, x, beta, yref);
  } else if (fOp[0] == KokkosSparse::Transpose[0]) {
    KokkosSparse::spmv("T", alpha, myMatrix, x, beta, yref);
  }
  KokkosSparse::spmv(controls, fOp, alpha, myBlockMatrix, x, beta, y);

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

  std::chrono::duration<double> dt_crs =
      measure("N", myMatrix, alpha, beta, repeat);

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

  std::cout << " Error Transpose BlockCrsMatrix " << error << " maxNorm "
            << maxNorm << "\n";
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
      std::make_tuple("GT01R.mtx", 5)  // ID:2335	Fluorem	GT01R
                                       // 7980	7980	430909	1	0
      // 1	0	0.8811455350661695	9.457852263618717e-06
      // computational fluid dynamics problem	430909
      ,
      std::make_tuple("raefsky4.mtx",
                      3)  // ID:817	Simon	raefsky4	19779
      // 19779	1316789	1	0	1	1	1
      // 1	structural problem	1328611
      //    , std::make_tuple("bmw7st_1.mtx", 6) // ID:1253	GHS_psdef
      //    bmw7st_1	141347	141347	7318399	1	0	1	1
      //    1
      //    1	structural problem	7339667
      //    , std::make_tuple("pwtk.mtx", 6) // ID:369	Boeing	pwtk 217918
      //    217918	11524432	1	0	1	1	1	1
      //    structural
      //    problem	11634424
      ,
      std::make_tuple("RM07R.mtx",
                      7)  // ID:2337	Fluorem	RM07R	381689	381689 37464962
      // 1	0	1	0
      // 0.9261667922354103	4.260681089287885e-06
      // computational fluid dynamics problem	37464962
      ,
      std::make_tuple("audikw_1.mtx",
                      3)  // ID:1252 GHS_psdef	audikw_1	943695	943695
                          // 77651847	1	0	1	1	1	1
                          // structural
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