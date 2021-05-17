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
typename KokkosSparse::CrsMatrix<Scalar, Ordinal, device_type, void,
    Offset>;

using values_type = typename crs_matrix_t_::values_type;

using bcrs_matrix_t_ =
typename KokkosSparse::Experimental::BlockCrsMatrix<Scalar, Ordinal, device_type, void,
    Offset>;


namespace details {

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
// spmv_beta_no_transpose: version for CPU execution spaces (RangePolicy or trivial serial impl used)
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
      YVector y)
{
  std::cerr << " GPU implementation is not complete !!! \n";
  assert(0 > 1);
  int vector_length = 1;
  int max_vector_length = 1;
#ifdef KOKKOS_ENABLE_CUDA
  if(std::is_same<execution_space, Kokkos::Cuda>::value)
    max_vector_length = 32;
#endif
#ifdef KOKKOS_ENABLE_HIP
  if(std::is_same<execution_space, Kokkos::Experimental::HIP>::value)
    max_vector_length = 64;
#endif

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

  BSPMV_Functor(const value_type alpha_, const AMatrix m_A_, const XVector m_x_,
               const value_type beta_, const YVector m_y_,
               const int rows_per_team_)
      : alpha(alpha_),
        m_A(m_A_),
        m_x(m_x_),
        beta(beta_),
        m_y(m_y_),
        rows_per_team(rows_per_team_),
        block_size(m_A_.blockDim())
  {
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const ordinal_type iRowBlock) const {
    using y_value_type = typename YVector::non_const_value_type;
    if (iRowBlock >= m_A.numRows()) {
      return;
    }
    const auto& A_graph  = m_A.graph;
    const auto Ar = m_A.block_row_Const(iRowBlock);
    const ordinal_type row_length = static_cast<ordinal_type>(Ar.length);
    Kokkos::View< y_value_type*> sum("sum", block_size * block_size);
    Kokkos::deep_copy(sum, 0);

    const auto blkBeg = A_graph.row_map[iRowBlock];
    const auto blkEnd = A_graph.row_map[iRowBlock + 1];
    for (Ordinal iblk = blkBeg; iblk < blkEnd; ++iblk) {
      const auto A_cur   = Ar.block(iblk - blkBeg);
      const Ordinal xBlk = A_graph.entries[iblk];
      const Ordinal xBeg = xBlk * block_size;
      const Ordinal xEnd = xBeg + block_size;
      const auto X_cur   = subview(m_x, Kokkos::make_pair(xBeg, xEnd));
      KokkosBlas::gemv("N", alpha, A_cur, X_cur, alpha, sum);
    }

    auto Y_cur        = Kokkos::subview(
        m_y, Kokkos::make_pair(iRowBlock * block_size, iRowBlock * block_size + block_size));

    Kokkos::deep_copy(Y_cur, sum);

/*
    sum *= alpha;
    if (dobeta == 0) {
      m_y(iRow) = sum;
    } else {
      m_y(iRow) = beta * m_y(iRow) + sum;
    }
*/

  }

}; // struct BSPMV_Functor

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
      YVector y)
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

  XVector_Internal x_i = x;
  YVector_Internal y_i = y;

  if(alpha == Kokkos::ArithTraits<AlphaType>::zero() ||
     A.numRows() == 0 || A.numCols() == 0 || A.nnz() == 0 || A.blockDim() == 0)
  {
    //This is required to maintain semantics of KokkosKernels native SpMV:
    //if y contains NaN but beta = 0, the result y should be filled with 0.
    //For example, this is useful for passing in uninitialized y and beta=0.
    if(beta == Kokkos::ArithTraits<BetaType>::zero())
      Kokkos::deep_copy(y_i, Kokkos::ArithTraits<BetaType>::zero());
    else
      KokkosBlas::scal(y_i, beta, y_i);
    return;
  }

  //
  // Treat the case y <- alpha * A * x + beta * y
  //

  Ordinal blockSize    = A.blockDim();
  Ordinal numBlockRows = A.numRows();
  for (Ordinal ijk = 0; ijk < numBlockRows * blockSize; ++ijk)
    y_i(ijk) = 0.0;

#if defined(KOKKOS_ENABLE_SERIAL)
  if (std::is_same<execution_space, Kokkos::Serial>::value) {
    std::cout << " RUN SERIAL \n";
    //
    ////// This is slower than sparse Mat-Vec
    //
    const auto& A_graph  = A.graph;
    for (Ordinal row = 0; row < numBlockRows; ++row) {
      auto Ar           = A.block_row_Const(row);
      const auto blkBeg = A_graph.row_map[row];
      const auto blkEnd = A_graph.row_map[row + 1];
      auto Y_cur        = Kokkos::subview(
          y_i, Kokkos::make_pair(row * blockSize, row * blockSize + blockSize));
      for (Ordinal iblk = blkBeg; iblk < blkEnd; ++iblk) {
        const auto A_cur   = Ar.block(iblk - blkBeg);
        const Ordinal xBlk = A_graph.entries[iblk];
        const Ordinal xBeg = xBlk * blockSize;
        const Ordinal xEnd = xBeg + blockSize;
        const auto X_cur   = subview(x_i, Kokkos::make_pair(xBeg, xEnd));
        KokkosBlas::gemv("N", alpha, A_cur, X_cur, alpha, Y_cur);
      }
    }
    return;
  }
#endif

#ifdef KOKKOS_ENABLE_OPENMP
  std::cout << " Run OPENMP case \n";
  //
  // This section terminates for the moment.
  // terminate called recursively
  //
  const auto& A_graph      = A.graph;
  const Ordinal blockSize2 = blockSize * blockSize;

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
  const int dobeta = 1;
  const bool conjugate = false;
  //
  BSPMV_Functor<bcrs_matrix_t_,XVector,YVector,dobeta,conjugate>
      func(alpha, A, x, beta, y, 1);
  if (((A.nnz()>10000000) || use_dynamic_schedule) && !use_static_schedule) {
    Kokkos::parallel_for("KokkosSparse::spmv<NoTranspose,Dynamic>",
                         Kokkos::RangePolicy<execution_space, Kokkos::Schedule<Kokkos::Dynamic>>(0, A.numRows()),func);
  }
  else {
    Kokkos::parallel_for("KokkosSparse::spmv<NoTranspose,Static>",
                         Kokkos::RangePolicy<execution_space, Kokkos::Schedule<Kokkos::Static>>(0, A.numRows()),func);
  }
#endif

}

} // namespace details

int main() {

  Kokkos::initialize();

  int return_value = 0;

  {
    const Scalar SC_ONE = Kokkos::ArithTraits<Scalar>::one();
    const Scalar SC_ZERO = Kokkos::ArithTraits<Scalar>::zero();

    // The mat_structure view is used to generate a matrix using
    // finite difference (FD) or finite element (FE) discretization
    // on a cartesian grid.
    // Each row corresponds to an axis (x, y and z)
    // In each row the first entry is the number of grid point in
    // that direction, the second and third entries are used to apply
    // BCs in that direction, BC=0 means Neumann BC is applied,
    // BC=1 means Dirichlet BC is applied by zeroing out the row and putting
    // one on the diagonal.
    Kokkos::View<Ordinal* [3], Kokkos::HostSpace> mat_structure(
        "Matrix Structure", 2);
    mat_structure(0, 0) = 32;  // Request 10 grid point in 'x' direction
    mat_structure(0, 1) = 0;   // Add BC to the left
    mat_structure(0, 2) = 0;   // Add BC to the right
    mat_structure(1, 0) = 32;  // Request 10 grid point in 'y' direction
    mat_structure(1, 1) = 0;   // Add BC to the bottom
    mat_structure(1, 2) = 0;   // Add BC to the top

    const int repeat = 16, blockSize = 1;
    std::vector<int> mat_rowmap, mat_colidx;
    std::vector<double> mat_val;

    crs_matrix_t_ myMatrix = details::generate_crs_matrix("FD", mat_structure,
                                                          blockSize,
                                                          mat_rowmap, mat_colidx,
                                                          mat_val);

    const Ordinal numRows = myMatrix.numRows();

    const Scalar alpha = SC_ONE;
    const Scalar beta  = SC_ZERO;

    typename values_type::non_const_type x("lhs", numRows);
    typename values_type::non_const_type y("rhs", numRows);

    auto tBegin = std::chrono::high_resolution_clock::now();
    for (int ir = 0; ir < repeat; ++ir) {
      Kokkos::deep_copy(x, SC_ONE);
      KokkosSparse::spmv("N", alpha, myMatrix, x, beta, y);
    }
    auto tEnd = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> dt_crs = tEnd - tBegin;

    std::cout << " Block Size " << blockSize << "\n";
    std::cout << " Matrix Size " << numRows
              << " nnz " << myMatrix.nnz() << "\n";
    std::cout << " Total time for Crs Mat-Vec " << dt_crs.count()
              << " Avg. " << dt_crs.count() / static_cast<double>(repeat)
              << "\n";

    //
    // Use BlockCrsMatrix format
    //

    bcrs_matrix_t_ myBlockMatrix(myMatrix, blockSize);

    typename values_type::non_const_type yref("ref", numRows);
    for (Ordinal ir = 0; ir < numRows; ++ir)
      x(ir) = std::rand() / static_cast<Scalar>(RAND_MAX);

    KokkosSparse::spmv("N", alpha, myMatrix, x, beta, yref);
    details::spmv(alpha, myBlockMatrix, x, beta, y);

    double error = 0.0, maxNorm = 0.0;
    for (Ordinal ir = 0; ir < numRows; ++ir) {
      error = std::max<double>(error, std::abs(yref(ir) - y(ir)));
      maxNorm = std::max<double>(maxNorm, std::abs(yref(ir)));
    }
    std::cout << " Error " << error << " maxNorm " << maxNorm << "\n";

    //
    // Test speed of Mat-Vec product
    //

    tBegin = std::chrono::high_resolution_clock::now();
    for (int ir = 0; ir < repeat; ++ir) {
      Kokkos::deep_copy(x, SC_ONE);
      details::spmv(alpha, myBlockMatrix, x, beta, y);
    }
    tEnd = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> dt_bcrs = tEnd - tBegin;
    std::cout << " Total time for BlockCrs Mat-Vec " << dt_bcrs.count()
              << " Avg. " << dt_bcrs.count() / static_cast<double>(repeat)
              << "\n";

  }

  Kokkos::finalize();

  return return_value;
}
