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

using Ordinal = default_lno_t;
using Offset  = default_size_type;
using Layout  = default_layout;

using clock_type = std::chrono::high_resolution_clock;
using duration_t = std::chrono::duration<double>;

using device_type = typename Kokkos::Device<
    Kokkos::DefaultExecutionSpace,
    typename Kokkos::DefaultExecutionSpace::memory_space>;

template<typename Scalar>
using crs_matrix_t_ =
    typename KokkosSparse::CrsMatrix<Scalar, Ordinal, device_type, void,
                                     Offset>;

template<typename Scalar>
using values_type = typename crs_matrix_t_<Scalar>::values_type;

template<typename Scalar>
using bcrs_matrix_t_ = typename KokkosSparse::Experimental::BlockCrsMatrix<
    Scalar, Ordinal, device_type, void, Offset>;

template<typename Scalar>
using MultiVector_Internal =
    typename Kokkos::View<Scalar **, Layout, device_type>;

namespace details {

} // namespace details

namespace test {

template<typename Scalar=default_scalar>
const Scalar SC_ONE  = Kokkos::ArithTraits<Scalar>::one();

template<typename Scalar=default_scalar>
const Scalar SC_ZERO = Kokkos::ArithTraits<Scalar>::zero();

template<typename Scalar>
inline Scalar random() {
  auto const max = static_cast<Scalar>(RAND_MAX) + static_cast<Scalar>(1);
  return static_cast<Scalar>(std::rand()) / max;
}

template<typename Scalar>
inline void set_random_value(Kokkos::complex<Scalar> &v) {
  v.real(random<Scalar>());
  v.imag(random<Scalar>());
}

template<typename Scalar>
inline void set_random_value(std::complex<Scalar> &v) {
  v.real(random<Scalar>());
  v.imag(random<Scalar>());
}

template<typename Scalar>
inline void set_random_value(Scalar &v) {
  v = random<Scalar>();
}

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
template <typename mat_structure, typename Scalar=default_scalar>
crs_matrix_t_<Scalar> generate_crs_matrix(const std::string stencil,
                                  const mat_structure &structure,
                                  const int blockSize,
                                  std::vector<Ordinal> &mat_rowmap,
                                  std::vector<Ordinal> &mat_colidx,
                                  std::vector<Scalar> &mat_val) {
  using crs_mtx_t = crs_matrix_t_<Scalar>;
  crs_mtx_t mat_b1 =
      Test::generate_structured_matrix2D<crs_mtx_t>(stencil, structure);

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
    set_random_value<Scalar>(val_ptr[ii]);

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

  return crs_mtx_t("new_crs_matr", nRow, nCol, nnz, val_ptr, rowmap, cols);
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
template<typename Scalar>
bcrs_matrix_t_<Scalar> to_block_crs_matrix(const crs_matrix_t_<Scalar> &mat_crs,
                                   const int blockSize) {
  using bcrs_mtx_t = bcrs_matrix_t_<Scalar>;
  if (blockSize == 1) {
    bcrs_mtx_t bmat(mat_crs, blockSize);
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
  values_type<Scalar> vals("values", annz);
  for (Ordinal i = 0; i < annz; ++i) vals(i) = 0.0;

  for (Ordinal ir = 0; ir < mat_crs.numRows(); ++ir) {
    const auto iblock = ir / blockSize;
    const auto ilocal = ir % blockSize;
    for (Ordinal jk = mat_crs.graph.row_map(ir); jk < mat_crs.graph.row_map(ir + 1); ++jk) {
      const auto jc     = mat_crs.graph.entries(jk);
      const auto jblock = jc / blockSize;
      const auto jlocal = jc % blockSize;
      for (Ordinal jkb = rows[iblock]; jkb < rows[iblock + 1]; ++jkb) {
        if (cols(jkb) == jblock) {
          Ordinal shift = jkb * blockSize * blockSize;
          vals(shift + jlocal + ilocal * blockSize) = mat_crs.values(jk);
          break;
        }
      }
    }
  }

  bcrs_mtx_t bmat("newblock", nbrows, nbcols, annz, vals, rows, cols,
                      blockSize);
  return bmat;
}


/////////////////////////////////////////////////////


/// \brief Generate a random multi-vector
///
/// \param numRows Number of rows
/// \param numCols Number of columns
/// \return Vector
template<typename Scalar>
MultiVector_Internal<Scalar> make_lhs(const int numRows, const int numCols) {
  MultiVector_Internal<Scalar> X("lhs", numRows, numCols);
  for (Ordinal ir = 0; ir < numRows; ++ir) {
    for (Ordinal jc = 0; jc < numCols; ++jc) {
      set_random_value(X(ir, jc));
    }
  }
  return X;
}

/// \brief Generate a random vector
///
/// \param numRows Number of rows
/// \return Vector
template<typename Scalar>
typename values_type<Scalar>::non_const_type make_lhs(const int numRows) {
  typename values_type<Scalar>::non_const_type x("lhs", numRows);
  for (Ordinal ir = 0; ir < numRows; ++ir)
    set_random_value(x(ir));
  return x;
}

template <typename mtx_t, typename Scalar=typename mtx_t::value_type>
duration_t measure(const char fOp[], const mtx_t &myMatrix,
                                      const Scalar alpha, const Scalar beta,
                                      const int repeat) {
  static_assert(std::is_same<Scalar, typename mtx_t::value_type>::value);

  const Ordinal numRows = myMatrix.numRows();

  auto const x = make_lhs<Scalar>(numRows);
  typename values_type<Scalar>::non_const_type y("rhs", numRows);

  auto tBegin = clock_type::now();
  for (int ir = 0; ir < repeat; ++ir) {
    KokkosSparse::spmv(fOp, alpha, myMatrix, x, beta, y);
  }
  auto tEnd = clock_type::now();
  duration_t dt = tEnd - tBegin;

  return dt;
}

template <typename bmtx_t, typename Scalar=typename bmtx_t::value_type>
duration_t measure_block(const char fOp[],
                                            const bmtx_t &myBlockMatrix,
                                            const Scalar alpha,
                                            const Scalar beta,
                                            const int repeat) {
  auto const numRows = myBlockMatrix.numRows() * myBlockMatrix.blockDim();
  auto const x       = make_lhs<Scalar>(numRows);
  typename values_type<Scalar>::non_const_type y("rhs", numRows);
  KokkosKernels::Experimental::Controls controls;

  auto tBegin = clock_type::now();
  for (int ir = 0; ir < repeat; ++ir) {
    KokkosSparse::spmv(controls, fOp, alpha, myBlockMatrix, x, beta, y);
  }
  auto tEnd = clock_type::now();
  duration_t dt = tEnd - tBegin;

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

/// \brief Compare the matrix-vector product between BlockCrsMatrix and CrsMatrix
///
/// \tparam mtx_t
/// \tparam bmtx_t
/// \param fOp
/// \param myMatrix
/// \param myBlockMatrix
/// \param alpha
/// \param beta
/// \param error
/// \param maxNorm
/// \return True when the results are numerically identical (false, otherwise)
template <typename mtx_t, typename bmtx_t, typename Scalar=typename mtx_t::value_type>
bool compare(const char fOp[], const mtx_t &myMatrix,
             const bmtx_t &myBlockMatrix, const Scalar alpha, const Scalar beta,
             double &error, double &maxNorm) {
  static_assert(std::is_same<Scalar, typename mtx_t::value_type>::value);
  static_assert(std::is_same<Scalar, typename bmtx_t::value_type>::value);
  error   = 0.0;
  maxNorm = 0.0;
  //
  KokkosKernels::Experimental::Controls controls;
  const auto numRows = myMatrix.numRows();
  const auto numCols = myMatrix.numCols();
  Ordinal xrow = 0, yrow = 0;
  if (fOp[0] == KokkosSparse::NoTranspose[0]
   || fOp[0] == KokkosSparse::Conjugate[0]) {
    xrow = static_cast<Ordinal>(numCols);
    yrow = static_cast<Ordinal>(numRows);
  } else if (fOp[0] == KokkosSparse::Transpose[0]
   || fOp[0] == KokkosSparse::ConjugateTranspose[0]) {
    yrow = static_cast<Ordinal>(numCols);
    xrow = static_cast<Ordinal>(numRows);
  }
  auto const x = make_lhs<Scalar>(xrow);
  typename values_type<Scalar>::non_const_type y("rhs", yrow);
  typename values_type<Scalar>::non_const_type yref("ref", yrow);
  //
  KokkosSparse::spmv(controls, fOp, alpha, myMatrix, x, beta, yref);
  KokkosSparse::spmv(controls, fOp, alpha, myBlockMatrix, x, beta, y);
  //
  for (Ordinal ir = 0; ir < yrow; ++ir) {
    /*
    if (ir < 16) {
      std::cout << '\t' << ir << '\t' << x(ir) << '\t' << yref(ir)
      << '\t' << y(ir) << std::endl;
    }
    */
    error   = std::max<double>(error, Kokkos::ArithTraits<Scalar>::abs(yref(ir) - y(ir)));
    maxNorm = std::max<double>(maxNorm, Kokkos::ArithTraits<Scalar>::abs(yref(ir)));
  }
  double tol = 2.2e-16 * y.size();
  if (error <= tol * maxNorm)
    return true;
  else
    return false;
}

struct RunInfo {
  using Scalar = default_scalar; // TODO: test with complex alpha/beta ?

  // options
  const char *mode = "N"; // N/T/C/H
  const Scalar alpha = SC_ONE<Scalar>;
  const Scalar beta = SC_ZERO<Scalar>;
  // results
  double error = 0.0;
  double maxNorm = 0.0;
  duration_t dt_crs;
  duration_t dt_bcrs;
};

using Variants = std::vector<RunInfo>;

template<typename Output>
class TestRunner {
public:
  TestRunner(Output &out_):
    out(out_)
  {
  }

  //
  // Assess y <- A * x
  //
  template<
    typename mtx_t = crs_matrix_t_<default_scalar>,
    typename bmtx_t = bcrs_matrix_t_<default_scalar>,
    typename Scalar = typename mtx_t::value_type
    >
  void operator()(const std::string &name, mtx_t myMatrix, Variants &variants,
                  const int blockSize, const int repeat = 1024)
  {
    ++sample_id;
    int variant_id = 0;
    auto myBlockMatrix = to_block_crs_matrix(myMatrix, blockSize); // Use BlockCrsMatrix format

    std::for_each(variants.begin(), variants.end(), [&](test::RunInfo &run) {

      if (!Kokkos::ArithTraits<Scalar>::is_complex &&
        (run.mode[0] == KokkosSparse::Conjugate[0] || run.mode[0] == KokkosSparse::ConjugateTranspose[0]))
        return; // test Conjugate/Hermitian only on complex samples

      Scalar alpha = run.alpha; // Note: convert to Scalar
      Scalar beta = run.beta;
      ++variant_id;
      auto const label = name + ":" + std::to_string(variant_id);
      out.showRunInfo(label, myMatrix, blockSize, run, label, 1 != variant_id);

      bool correct = compare(run.mode, myMatrix, myBlockMatrix, alpha, beta, run.error, run.maxNorm);
      if (correct) {
        run.dt_crs = measure(run.mode, myMatrix, alpha, beta, repeat);
        run.dt_bcrs = measure_block(run.mode, myBlockMatrix, alpha, beta, repeat);
      }

      out.showRunResults(myMatrix, repeat, run, correct);
      pass = pass && correct;
    });
  }

  bool all_passed() {
    return pass;
  }

private:
  Output &out;
  int sample_id = 0;
  bool pass = true;
};

template<typename Scalar = default_scalar, typename Executor> // =default_scalar
void test_random_samples(Executor &test_matrix, Variants &variants,
                 const int repeat = 1024,
                 const int minBlockSize = 1, const int maxBlockSize = 10)
{

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
    std::vector<Scalar> mat_val;
    auto const label = std::string("rand-") + (Kokkos::ArithTraits<Scalar>::is_complex ? "complex-" : "real-") + std::to_string(blockSize);
    auto myMatrix = generate_crs_matrix(
        "FD", mat_structure, blockSize, mat_rowmap, mat_colidx, mat_val);
    test_matrix(
      label,
      myMatrix,
      variants,
      blockSize,
      repeat
    );
  }
}

template<typename Executor>
void test_market_samples(Executor test_matrix, Variants &variants, const int repeat = 3000)
{
  using mtx_t = crs_matrix_t_<default_scalar>;

  const std::vector<std::tuple<const char *, int> > SAMPLES{
      // std::tuple(char* fileName, int blockSize)
      std::make_tuple("GT01R.mtx", 5)  // ID:2335	Fluorem	GT01R
      // 7980	7980	430909	1	0
      // 1	0	0.8811455350661695	9.457852263618717e-06
      // computational fluid dynamics problem	430909
      //,
      //std::make_tuple("raefsky4.mtx",
      //                3)  // ID:817	Simon	raefsky4	19779
      // 19779	1316789	1	0	1	1	1
      // 1	structural problem	1328611
      //    , std::make_tuple("bmw7st_1.mtx", 6) // ID:1253	GHS_psdef
      //    bmw7st_1	141347	141347	7318399	1	0	1	1
      //    1
      //    1	structural problem	7339667
      //    , std::make_tuple("pwtk.mtx", 6) // ID:369	Boeing	pwtk 217918
      //    217918	11524432	1	0	1	1	1
      //    1
      //    structural
      //    problem	11634424
      //,
      //std::make_tuple("RM07R.mtx",
      //                7)  // ID:2337	Fluorem	RM07R	381689	381689 37464962
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
  std::for_each(SAMPLES.begin(), SAMPLES.end(), [&](auto const &sample) {
    const char *fileName = std::get<0>(sample);
    auto const myMatrix = KokkosKernels::Impl::read_kokkos_crst_matrix<mtx_t>(fileName);
    test_matrix(
      std::string(fileName),
      myMatrix,
      variants,
      std::get<1>(sample),
      repeat
    );
  });
}

} // namespace test

class CSVOutput
{
  public:
  static constexpr const char* const sep = "\t";

  void showHeader() {
    std::cout << "no." << sep << "name" << sep << "size" << sep << "block" << sep << "nnz" // sample info
              << sep << "mode" << sep << "alpha" << sep << "beta" // run info
              << sep << "error" << sep << "maxNorm"
              << sep << "crsTime" << sep << "crsAvg" << sep << "crsGFlops"
              << sep << "bcrsTime" << sep << "bcrsAvg" << sep << "bcrsGFlops"
              << sep << "ratio" << sep << "remarks"
              << std::endl;
  }


  template<typename mtx_t>
  void showRunInfo(const std::string &name, const mtx_t &myMatrix, int blockSize,
                   const test::RunInfo &run, const std::string &sample_id, bool skipSample = false)
  {
    std::cout << sample_id << sep;
    if (skipSample)
      std::cout << sep << "^" << sep << "^" << sep << "^" << sep << "^";
    else
      std::cout << sep << name << sep << myMatrix.numRows()
                << sep << blockSize << sep << myMatrix.nnz();
    std::cout << sep << run.mode << sep << run.alpha << sep << run.beta;
  }

  template<typename mtx_t>
  void showRunResults(const mtx_t &myMatrix, int repeat, const test::RunInfo &run, bool pass)
  {
    std::cout << sep << run.error << sep << run.maxNorm;
    if (!pass) {
      for (int i = 0; i < 7; ++i) {
        std::cout << sep << "--";
      }
      std::cout << sep << "FAILED!" << std::endl;
      return;
    }
    auto const nnz = myMatrix.nnz();
    Ordinal flops = 0;
    //
    // This flop count would not work when the matrix is not square
    //
    if ((run.alpha == 0) && (run.beta != 0)) {
      flops = myMatrix.numRows();
    }
    else if ((run.alpha != 0) && (run.beta == 0)) {
      flops = nnz;
    }
    else if ((run.alpha != 0) && (run.beta != 0)) {
      flops = nnz + myMatrix.numRows();
    }
    //
    showTime(run.dt_crs, flops, repeat);
    showTime(run.dt_bcrs, flops, repeat);
    auto const remarks = (run.dt_bcrs.count() < run.dt_crs.count()) ? "good" : "NOT_faster";
    std::cout << sep << (run.dt_bcrs.count() / run.dt_crs.count()) << sep << remarks;
    std::cout << std::endl;
  }

private:
  template <typename time_t, typename ord_t>
  void showTime(const time_t &t, ord_t flops, int repeat) {
    auto const avg = (t.count() / static_cast<double>(repeat));
    auto const total_flops = flops * static_cast<double>(repeat / t.count());
    std::cout << sep <<t.count() << sep << avg << sep << (total_flops * 1e-9);
  }
};

void set_variants(test::Variants &variants) {
  variants.push_back({KokkosSparse::NoTranspose});
  variants.push_back({KokkosSparse::Transpose});
  variants.push_back({KokkosSparse::Conjugate});
  variants.push_back({KokkosSparse::ConjugateTranspose});
  variants.push_back({KokkosSparse::NoTranspose, -1, -1.0});
  variants.push_back({KokkosSparse::NoTranspose, 3.14159, 0.25});
  variants.push_back({KokkosSparse::NoTranspose, 0.0, 0.0});
  variants.push_back({KokkosSparse::NoTranspose, 0.0, 1.0});
}

using Output     = CSVOutput;
using TestRunner = test::TestRunner<Output>;

int main() {
  Kokkos::initialize();
  bool failed = false;
  srand(17312837);
  {
    const int repeat = 1024;
    test::Variants variants;
    set_variants(variants);
    Output out;
    TestRunner runner(out);
    out.showHeader();
    auto tBegin = clock_type::now();
    // Run samples

    // cover small blocks - including optimized implementations for blockSize=1
    test::test_random_samples(runner, variants, repeat, 1, 3);

    // test complex on small blocks
    test::test_random_samples<Kokkos::complex<default_scalar> >(runner, variants, repeat, 1, 1);
    test::test_random_samples<Kokkos::complex<default_scalar> >(runner, variants, repeat,
                              KokkosSparse::Impl::bmax - 1, KokkosSparse::Impl::bmax - 1);
    test::test_random_samples<Kokkos::complex<default_scalar> >(runner, variants, repeat,
                              KokkosSparse::Impl::bmax + 1, KokkosSparse::Impl::bmax + 1);
    // FIXME: does not work with Kokkos::atomic_add(...) in our implementation
    // test::test_random_samples<std::complex<default_scalar> >(runner, variants, repeat, 3, 3);

    // cover ETI-expanded (small blocks) and dynamic (large blocks) implementations
    test::test_random_samples(runner, variants, repeat,
                              KokkosSparse::Impl::bmax - 2, KokkosSparse::Impl::bmax + 1);

    // Test MM samples
    test::test_market_samples(runner, variants, repeat);

    auto tEnd = clock_type::now();
    if (runner.all_passed())
      std::cout << "Finished in " << (tEnd - tBegin).count() << " s: All tests PASSED" << std::endl;
    else
      std::cout << "Finished in " << (tEnd - tBegin).count() << " s: FAILED" << std::endl;
  }

  Kokkos::finalize();
  return failed ? 1 : 0;
}