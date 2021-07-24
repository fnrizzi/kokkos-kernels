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


// #include <gtest/gtest.h>
// #include <Kokkos_Core.hpp>

// #include "KokkosKernels_SparseUtils.hpp"
// #include "KokkosKernels_Sorting.hpp"
// #include <Kokkos_Concepts.hpp>
// #include <string>
// #include <stdexcept>

#include "KokkosSparse_bspgemm_numeric.hpp"
#include "KokkosSparse_BsrMatrix.hpp"

//#include<KokkosKernels_IOUtils.hpp>

//This file contains the matrix for test_issue402
//#include "matrixIssue402.hpp"

//const char *input_filename = "sherman1.mtx";
//const char *input_filename = "Si2.mtx";
//const char *input_filename = "wathen_30_30.mtx";
//const size_t expected_num_cols = 9906;

// using namespace KokkosSparse;
// using namespace KokkosSparse::Experimental;
// using namespace KokkosKernels;
// using namespace KokkosKernels::Experimental;

// #ifndef kokkos_complex_double
// #define kokkos_complex_double Kokkos::complex<double>
// #define kokkos_complex_float Kokkos::complex<float>
// #endif

// typedef Kokkos::complex<double> kokkos_complex_double;
// typedef Kokkos::complex<float> kokkos_complex_float;

namespace KokkosSparse { // TODO: move to impl file

template <class KernelHandle, class AMatrix, class BMatrix, class CMatrix>
void bspgemm_symbolic(KernelHandle& kh, const AMatrix& A, const bool Amode,
                     const BMatrix& B, const bool Bmode, CMatrix& C) {

  using row_map_type = typename CMatrix::row_map_type::non_const_type;
  using entries_type = typename CMatrix::index_type::non_const_type;
  using values_type  = typename CMatrix::values_type::non_const_type;

  // TODO: Support different block sizes ?
  //       (calculate proper output block size, specialize for same block size ?)
  auto blockDim = A.blockDim();
  if(blockDim != B.blockDim()) {
    throw std::invalid_argument("Block SpGEMM must be called for matrices with the same block size");
  }

  row_map_type row_mapC(
    Kokkos::view_alloc(Kokkos::WithoutInitializing, "non_const_lnow_row"),
    A.numRows() + 1);

  KokkosSparse::Experimental::spgemm_symbolic(
      &kh, A.numRows(), B.numRows(), B.numCols(), A.graph.row_map,
      A.graph.entries, Amode, B.graph.row_map, B.graph.entries, Bmode,
      row_mapC);

  entries_type entriesC;
  values_type valuesC;
  const size_t c_nnz_size = kh.get_spgemm_handle()->get_c_nnz();
  if (c_nnz_size) {
    entriesC = entries_type(
      Kokkos::view_alloc(Kokkos::WithoutInitializing, "entriesC"),
      c_nnz_size);
    valuesC  = values_type(
      Kokkos::view_alloc(Kokkos::WithoutInitializing, "valuesC"),
      c_nnz_size * blockDim * blockDim);
  }

  C = CMatrix("C=AB", A.numRows(), B.numCols(), c_nnz_size, valuesC, row_mapC, entriesC, blockDim);
}

}

namespace Test {

// TODO: can we deduce device from matrix type ?
template <typename bsrMat_t>
int run_bspgemm(bsrMat_t A, bsrMat_t B, KokkosSparse::SPGEMMAlgorithm spgemm_algorithm, bsrMat_t &C) {

  using execution_space = typename bsrMat_t::execution_space;
  using memory_space = typename bsrMat_t::memory_space;
  using size_type = typename bsrMat_t::size_type;
  using lno_t = typename bsrMat_t::ordinal_type;
  using scalar_t = typename bsrMat_t::value_type;

  typedef KokkosKernels::Experimental::KokkosKernelsHandle<
    size_type, lno_t, scalar_t,
    execution_space, memory_space, memory_space> KernelHandle;

  KernelHandle kh;
  kh.set_team_work_size(16);       // TODO: parametrize ? (see SpMV)
  kh.set_dynamic_scheduling(true); // TODO: parametrize ? (see SpMV)

  kh.create_spgemm_handle(spgemm_algorithm);

  KokkosSparse::bspgemm_symbolic(kh, A, false, B, false, C);

  KokkosSparse::Experimental::bspgemm_numeric(&kh,
      A.numRows(), B.numRows(), B.numCols(), A.blockDim(),
      A.graph.row_map, A.graph.entries, A.values, false,
      B.graph.row_map, B.graph.entries, B.values, false,
      C.graph.row_map, C.graph.entries, C.values);
  kh.destroy_spgemm_handle();

  return 0;
}

template <typename bsrMat_t>
bool is_same_block_matrix(bsrMat_t actual, bsrMat_t reference){

  typedef typename bsrMat_t::StaticCrsGraphType graph_t;
  typedef typename graph_t::row_map_type::non_const_type lno_view_t;
  typedef typename graph_t::entries_type::non_const_type   lno_nnz_view_t;
  typedef typename bsrMat_t::values_type::non_const_type scalar_view_t;
  typedef typename bsrMat_t::execution_space execution_space;

  size_t nrows_actual = actual.numRows();
  size_t nentries_actual = actual.graph.entries.extent(0) ;
  size_t nvals_actual = actual.values.extent(0);

  size_t nrows_reference = reference.numRows();
  size_t nentries_reference = reference.graph.entries.extent(0) ;
  size_t nvals_reference = reference.values.extent(0);

  if (nrows_actual != nrows_reference) {
     std::cout << "nrows_actual:" << nrows_actual << " nrows_reference:" << nrows_reference << std::endl;
     return false;
  }
  if (nentries_actual != nentries_reference) {
    std::cout << "nentries_actual:" << nentries_actual << " nentries_reference:" << nentries_reference << std::endl;
    return false;
  }
  if (nvals_actual != nvals_reference) {
    std::cout << "nvals_actual:" << nvals_actual << " nvals_reference:" << nvals_reference << std::endl;
    return false;
  }

  KokkosKernels::sort_crs_matrix(actual);
  KokkosKernels::sort_crs_matrix(reference);

  bool is_identical = true;
  is_identical = KokkosKernels::Impl::kk_is_identical_view<
      typename graph_t::row_map_type, typename graph_t::row_map_type,
      typename lno_view_t::value_type, execution_space
    >(actual.graph.row_map, reference.graph.row_map, 0);

  if (!is_identical) {
    std::cout << "rowmaps are different." << std::endl;
    std::cout << "Actual rowmap:\n";
    KokkosKernels::Impl::kk_print_1Dview(actual.graph.row_map);
    std::cout << "Correct rowmap (SPGEMM_DEBUG):\n";
    KokkosKernels::Impl::kk_print_1Dview(reference.graph.row_map);
    return false;
  }

  is_identical = KokkosKernels::Impl::kk_is_identical_view
      <lno_nnz_view_t, lno_nnz_view_t, typename lno_nnz_view_t::value_type,
      execution_space>(actual.graph.entries, reference.graph.entries, 0 );

  if (!is_identical) {
    std::cout << "entries are different." << std::endl;
    KokkosKernels::Impl::kk_print_1Dview(actual.graph.entries);
    KokkosKernels::Impl::kk_print_1Dview(reference.graph.entries);
    return false;
  }


  typedef typename Kokkos::Details::ArithTraits<typename scalar_view_t::non_const_value_type>::mag_type eps_type;
  eps_type eps = std::is_same<eps_type,float>::value?2*1e-3:1e-7;


  is_identical = KokkosKernels::Impl::kk_is_relatively_identical_view
      <scalar_view_t, scalar_view_t, eps_type,
      execution_space>(actual.values, reference.values, eps);

  if (!is_identical) {
    std::cout << "values are different." << std::endl;
    KokkosKernels::Impl::kk_print_1Dview(actual.values);
    KokkosKernels::Impl::kk_print_1Dview(reference.values);

    return false;
  }
  return true;
}
}

template<typename bsrMat_t>
void generate_sample(bsrMat_t &A, bsrMat_t &B, bsrMat_t &refC) {
  typedef typename bsrMat_t::StaticCrsGraphType          graph_t;
  typedef typename graph_t::row_map_type::non_const_type row_map_view_t;
  typedef typename graph_t::entries_type::non_const_type cols_view_t;
  typedef typename bsrMat_t::values_type::non_const_type values_view_t;

  typedef typename row_map_view_t::non_const_value_type size_type;
  typedef typename cols_view_t::non_const_value_type    lno_t;
  typedef typename values_view_t::non_const_value_type  scalar_t;

  const lno_t blockDim = 2;
  { // generate matrix A
    const lno_t numRows = 3;
    const lno_t numCols = 2;
    const lno_t nnz = 2; // numBlocks
    row_map_view_t rowmap_view("rowmap_view", numRows + 1);
    cols_view_t    columns_view("colsmap_view", nnz);
    values_view_t  values_view("values_view", nnz * blockDim * blockDim);
    typename row_map_view_t::HostMirror hr = Kokkos::create_mirror_view(rowmap_view);
    typename cols_view_t::HostMirror    hc = Kokkos::create_mirror_view(columns_view);
    typename values_view_t::HostMirror  hv = Kokkos::create_mirror_view(values_view);
    hr(0) = 0;                       // [ 0 0 ]    A = [2 0]
    hr(1) = 0; hc(0) = 1;            // [ 0 A ]        [0 2]
    hr(2) = 1; hc(1) = 1;            // [ 0 I ]    I = 2x2 ID matrix
    hr(3) = 2;
    hv(0) = 2; hv(1) = 0; hv(2) = 0; hv(3) = 2; // A
    hv(4) = 1; hv(5) = 0; hv(6) = 0; hv(7) = 1; // I
    Kokkos::deep_copy(rowmap_view, hr);
    Kokkos::deep_copy(columns_view, hc);
    Kokkos::deep_copy(values_view, hv);
    A = bsrMat_t("A", numRows, numCols, nnz, values_view, rowmap_view, columns_view, blockDim);
  }
  { // generate matrix B
    const lno_t numRows = 2;
    const lno_t numCols = 4;
    const lno_t nnz = 2; // numBlocks
    row_map_view_t rowmap_view("rowmap_view", numRows + 1);
    cols_view_t    columns_view("colsmap_view", nnz);
    values_view_t  values_view("values_view", nnz * blockDim * blockDim);
    typename row_map_view_t::HostMirror hr = Kokkos::create_mirror_view(rowmap_view);
    typename cols_view_t::HostMirror    hc = Kokkos::create_mirror_view(columns_view);
    typename values_view_t::HostMirror  hv = Kokkos::create_mirror_view(values_view);
    hr(0) = 0;                       // [ 0 0 0 0 ] B = [4 0]
    hr(1) = 0; hc(0) = 0; hc(1) = 3; // [ I 0 0 B ]     [0 4]
    hr(2) = 2;
    hv(0) = 1; hv(1) = 0; hv(2) = 0; hv(3) = 1; // I
    hv(4) = 4; hv(5) = 0; hv(6) = 0; hv(7) = 4; // B
    Kokkos::deep_copy(rowmap_view, hr);
    Kokkos::deep_copy(columns_view, hc);
    Kokkos::deep_copy(values_view, hv);
    B = bsrMat_t("B", numRows, numCols, nnz, values_view, rowmap_view, columns_view, blockDim);
  }
  { // generate matrix C (refernce output)
    const lno_t numRows = 3;
    const lno_t numCols = 4;
    const lno_t nnz = 4; // numBlocks
    row_map_view_t rowmap_view("rowmap_view", numRows + 1);
    cols_view_t    columns_view("colsmap_view", nnz);
    values_view_t  values_view("values_view", nnz * blockDim * blockDim);
    typename row_map_view_t::HostMirror hr = Kokkos::create_mirror_view(rowmap_view);
    typename cols_view_t::HostMirror    hc = Kokkos::create_mirror_view(columns_view);
    typename values_view_t::HostMirror  hv = Kokkos::create_mirror_view(values_view);
    hr(0) = 0;                       // [ 0 0 0  0 ]
    hr(1) = 0; hc(0) = 0; hc(1) = 3; // [ A 0 0 AB ]
    hr(2) = 2; hc(2) = 0; hc(3) = 3; // [ I 0 0  B ]
    hr(3) = 4;
    hv(0) = 2; hv(1) = 0; hv(2) = 0; hv(3) = 2; // A
    hv(4) = 8; hv(5) = 0; hv(6) = 0; hv(7) = 8; // AxB
    hv(8) = 1; hv(9) = 0; hv(10) = 0; hv(11) = 1; // I
    hv(12) = 4; hv(13) = 0; hv(14) = 0; hv(15) = 4; // B
    Kokkos::deep_copy(rowmap_view, hr);
    Kokkos::deep_copy(columns_view, hc);
    Kokkos::deep_copy(values_view, hv);
    refC = bsrMat_t("refC", numRows, numCols, nnz, values_view, rowmap_view, columns_view, blockDim);
  }
}

template <typename ostream_t, typename scalar_t, typename lno_t, typename size_type, typename device>
ostream_t& operator<<(ostream_t& out, const CrsMatrix<scalar_t, lno_t, device, void, size_type> &A) {
  for (int m = 0; m < A.numRows(); ++m) {
    out << ((m == 0) ? "[ " : "  ");
    auto row = A.row(m);
    int n = 0;
    for (int i = 0; i < row.length; ++i) {
      auto val = row.value(i);
      auto col = row.colidx(i);
      while (n++ < col)
        out << 0 << "\t";
      out << val << "\t";
    }
    while (n++ < A.numCols())
      out << 0 << "\t";
    out << (m + 1 == A.numRows() ? " ]\n" : "\n");
  }
  return out;
}

// Simple utility to display BSR matrix
template <typename ostream_t, typename scalar_t, typename lno_t, typename device, typename memtraits, typename size_t>
ostream_t& operator<<(ostream_t& out, const KokkosSparse::Experimental::BsrMatrix<scalar_t, lno_t, device, memtraits, size_t> &A) {
  lno_t numRows  = A.numRows();
  lno_t numCols  = A.numCols();
  lno_t blockDim = A.blockDim();
  for (lno_t y = 0, y1 = numRows * blockDim; y < y1; ++y) {
    out << "[ ";
    lno_t i = y / blockDim;
    lno_t bi = y - (i * blockDim); // block row offset
    auto row = A.block_row_Const(i);
    for (lno_t x = 0, x1 = numCols * blockDim; x < x1; ++x) {
      lno_t j = x / blockDim;
      lno_t bj = x - (j * blockDim);
      lno_t k = row.findRelBlockOffset(j); // block entry index
      if(k < row.length) {
        auto val = row.block(k);
        out << val(bi, bj) << "\t";
      } else
        out << ".\t"; // fill in [sparse] zeros
      if (bj + 1 == blockDim && x + 1 != x1)
        out << "|"; // draw block border
    }
    out << " ]\n";
    if (bi + 1 == blockDim && y + 1 != y1) {
      for (lno_t x = 0, x1 = numCols * blockDim; x < x1; ++x)
        out << "--------"; // draw block border
      out << "\n";
    }
  }
  return out;
}

template <typename scalar_t, typename lno_t, typename size_type, typename device>
void test_bspgemm(lno_t numRows, size_type nnz, lno_t bandwidth, lno_t row_size_variance) {

  using namespace Test;
  // Kokkos::print_configuration(std::cout, false);
  // device::execution_space::print_configuration(std::cout);

#if 1
  using blkcrsMat_t = KokkosSparse::Experimental::BsrMatrix<scalar_t, lno_t, device, void, size_type>;

  blkcrsMat_t A, B, refC;
  generate_sample(A, B, refC);
  //std::cout << "A = \n" << A << "\n";
  //std::cout << "B = \n" << B << "\n";

  blkcrsMat_t C;
  run_bspgemm(A, B, SPGEMM_DEBUG, C);
  // std::cout << "C = \n" << C << "\n";

  EXPECT_TRUE(is_same_block_matrix(C, refC)) << "Block SpMM: Serial/Debug";

  return;
#endif
/* // TODO:
  typedef CrsMatrix<scalar_t, lno_t, device, void, size_type> crsMat_t;

  lno_t numCols = numRows;
  // Generate random compressed sparse row matrix. Randomly generated (non-zero) values are
  // stored in a 1-D (1 rank) array.
  crsMat_t input_mat = KokkosKernels::Impl::kk_generate_sparse_matrix<crsMat_t>(numRows,numCols,nnz,row_size_variance, bandwidth);

  crsMat_t output_mat2;
  run_spgemm<crsMat_t, device>(input_mat, input_mat, SPGEMM_DEBUG, output_mat2);

  std::vector<SPGEMMAlgorithm> algorithms = {SPGEMM_KK_MEMORY, SPGEMM_KK_SPEED, SPGEMM_KK_MEMSPEED};

#ifdef HAVE_KOKKOSKERNELS_MKL
  algorithms.push_back(SPGEMM_MKL);
#endif
#ifdef KERNELS_HAVE_CUSPARSE
  algorithms.push_back(SPGEMM_CUSPARSE);
#endif
  for (auto spgemm_algorithm : algorithms)
  {
    const uint64_t max_integer = 2147483647;
    std::string algo = "UNKNOWN";
    bool is_expected_to_fail = false;

    switch (spgemm_algorithm){
    case SPGEMM_CUSPARSE:
      //TODO: add these test failure cases for cusparse too.
      algo = "SPGEMM_CUSPARSE";
#if !defined(KERNELS_HAVE_CUSPARSE) && !defined(KOKKOSKERNELS_ENABLE_TPL_CUSPARSE)
      is_expected_to_fail = true;
#endif
      break;

    case SPGEMM_MKL:
      algo = "SPGEMM_MKL";
      //MKL requires scalar to be either float or double
      if (!(std::is_same<float,scalar_t>::value || std::is_same<double,scalar_t>::value)){
        is_expected_to_fail = true;
      }
      //mkl requires local ordinals to be int.
      if (!(std::is_same<int,lno_t>::value)){
        is_expected_to_fail = true;
      }
      //if size_type is larger than int, mkl casts it to int.
      //it will fail if casting cause overflow.
      if (input_mat.values.extent(0) > max_integer){
        is_expected_to_fail = true;
      }

      if (!(Kokkos::Impl::SpaceAccessibility<typename Kokkos::HostSpace::execution_space, typename device::memory_space>::accessible)){
        is_expected_to_fail = true;
      }
      break;

    case SPGEMM_KK_MEMSPEED:
      algo = "SPGEMM_KK_MEMSPEED";
      break;
    case SPGEMM_KK_SPEED:
      algo = "SPGEMM_KK_SPEED";
      break;
    case SPGEMM_KK_MEMORY:
      algo = "SPGEMM_KK_MEMORY";
      break;
    default:
      algo = "!!! UNKNOWN ALGO !!!";
    }

    Kokkos::Impl::Timer timer1;
    crsMat_t output_mat;

    bool failed = false;
    int res = 0;
    try{
    	res = run_spgemm<crsMat_t, device>(input_mat, input_mat, spgemm_algorithm, output_mat);
    }
    catch (const char *message){
      EXPECT_TRUE(is_expected_to_fail) << algo;
      failed = true;
    }
    catch (std::string message){
      EXPECT_TRUE(is_expected_to_fail)<< algo;
      failed = true;
    }
    catch (std::exception& e){
      EXPECT_TRUE(is_expected_to_fail)<< algo;
      failed = true;
    }
    EXPECT_TRUE((failed == is_expected_to_fail));

    double spgemm_time = timer1.seconds();

    timer1.reset();
    if (!is_expected_to_fail){

      EXPECT_TRUE( (res == 0)) << algo;
      bool is_identical = is_same_matrix<crsMat_t, device>(output_mat, output_mat2);
      EXPECT_TRUE(is_identical) << algo;
    }
    std::cout << "algo:" << algo << " spgemm_time:" << spgemm_time << " output_check_time:" << timer1.seconds() << std::endl;
  }
*/
}

// template <typename scalar_t, typename lno_t, typename size_type, typename device>
// void test_issue402()
// {
//   using namespace Test;
//   typedef CrsMatrix<scalar_t, lno_t, device, void, size_type> crsMat_t;

//   //this specific matrix (from a circuit simulation) reliably replicated issue #402 (incorrect/crashing SPGEMM KKMEM)
//   typedef typename crsMat_t::StaticCrsGraphType graph_t;
//   typedef typename graph_t::row_map_type::non_const_type lno_view_t;
//   typedef typename graph_t::entries_type::non_const_type lno_nnz_view_t;
//   typedef typename crsMat_t::values_type::non_const_type scalar_view_t;
//   const lno_t numRows = 1813;
//   const size_type nnz = 11156;
//   lno_view_t Arowmap("A rowmap", numRows + 1);
//   lno_nnz_view_t Aentries("A entries", nnz);
//   scalar_view_t Avalues("A values", nnz);
//   //Read out the matrix from the header file "matrixIssue402.hpp"
//   {
//     auto rowmapHost = Kokkos::create_mirror_view(Arowmap);
//     auto entriesHost = Kokkos::create_mirror_view(Aentries);
//     auto valuesHost = Kokkos::create_mirror_view(Avalues);
//     for(lno_t i = 0; i < numRows + 1; i++)
//       rowmapHost(i) = MatrixIssue402::rowmap[i];
//     for(size_type i = 0; i < nnz; i++)
//     {
//       entriesHost(i) = MatrixIssue402::entries[i];
//       valuesHost(i) = MatrixIssue402::values[i];
//     }
//     Kokkos::deep_copy(Arowmap, rowmapHost);
//     Kokkos::deep_copy(Aentries, entriesHost);
//     Kokkos::deep_copy(Avalues, valuesHost);
//   }
//   crsMat_t A("A", numRows, numRows, nnz, Avalues, Arowmap, Aentries);
//   //compute explicit transpose: the bug was replicated by computing AA'
//   lno_view_t Browmap("B = A^T rowmap", numRows + 1);
//   lno_nnz_view_t Bentries("B = A^T entries", nnz);
//   scalar_view_t Bvalues("B = A^T values", nnz);
//   KokkosKernels::Impl::transpose_matrix<
//     lno_view_t, lno_nnz_view_t, scalar_view_t,
//     lno_view_t, lno_nnz_view_t, scalar_view_t,
//     lno_view_t, typename device::execution_space>
//       (numRows, numRows, Arowmap, Aentries, Avalues, Browmap, Bentries, Bvalues);
//   crsMat_t B("B=A^T", numRows, numRows, nnz, Bvalues, Browmap, Bentries);
//   crsMat_t Cgold;
//   run_spgemm<crsMat_t, device>(A, B, SPGEMM_DEBUG, Cgold);
//   crsMat_t C;
//   bool success = true;
//   std::string errMsg;
//   try
//   {
//     int res = run_spgemm<crsMat_t, device>(A, B, SPGEMM_KK_MEMORY, C);
//     if(res)
//       throw "run_spgemm returned error code";
//   }
//   catch(const char *message) {
//     errMsg = message;
//     success = false;
//   }
//   catch (std::string message) {
//     errMsg = message;
//     success = false;
//   }
//   catch (std::exception& e) {
//     errMsg = e.what();
//     success = false;
//   }
//   EXPECT_TRUE(success) << "KKMEM still has issue 402 bug! Error message:\n" << errMsg << '\n';
//   bool correctResult = is_same_matrix<crsMat_t, device>(C, Cgold);
//   EXPECT_TRUE(correctResult) << "KKMEM still has issue 402 bug; C=AA' is incorrect!\n";
// }

#define EXECUTE_TEST(SCALAR, ORDINAL, OFFSET, DEVICE) \
TEST_F( TestCategory, sparse_block_spgemm_ ## SCALAR ## _ ## ORDINAL ## _ ## OFFSET ## _ ## DEVICE ) { \
  test_bspgemm<SCALAR,ORDINAL,OFFSET,DEVICE>(2, 2, 500, 10); \
}

  //test_bspgemm<SCALAR,ORDINAL,OFFSET,DEVICE>(10000, 10000 * 20, 500, 10); \
  //test_bspgemm<SCALAR,ORDINAL,OFFSET,DEVICE>(0, 0, 10, 10); \
  // test_issue402<SCALAR,ORDINAL,OFFSET,DEVICE>(); \
  //test_bspgemm<SCALAR,ORDINAL,OFFSET,DEVICE>(10000, 10000 * 20, 500, 10, true);

//test_spgemm<SCALAR,ORDINAL,OFFSET,DEVICE>(50000, 50000 * 30, 100, 10);
//test_spgemm<SCALAR,ORDINAL,OFFSET,DEVICE>(50000, 50000 * 30, 200, 10);

#if (defined (KOKKOSKERNELS_INST_DOUBLE) \
 && defined (KOKKOSKERNELS_INST_ORDINAL_INT) \
 && defined (KOKKOSKERNELS_INST_OFFSET_INT) ) || (!defined(KOKKOSKERNELS_ETI_ONLY) && !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
 EXECUTE_TEST(double, int, int, TestExecSpace)
#endif
/*
#if (defined (KOKKOSKERNELS_INST_DOUBLE) \
 && defined (KOKKOSKERNELS_INST_ORDINAL_INT64_T) \
 && defined (KOKKOSKERNELS_INST_OFFSET_INT) ) || (!defined(KOKKOSKERNELS_ETI_ONLY) && !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
 EXECUTE_TEST(double, int64_t, int, TestExecSpace)
#endif

#if (defined (KOKKOSKERNELS_INST_DOUBLE) \
 && defined (KOKKOSKERNELS_INST_ORDINAL_INT) \
 && defined (KOKKOSKERNELS_INST_OFFSET_SIZE_T) ) || (!defined(KOKKOSKERNELS_ETI_ONLY) && !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
 EXECUTE_TEST(double, int, size_t, TestExecSpace)
#endif

#if (defined (KOKKOSKERNELS_INST_DOUBLE) \
 && defined (KOKKOSKERNELS_INST_ORDINAL_INT64_T) \
 && defined (KOKKOSKERNELS_INST_OFFSET_SIZE_T) ) || (!defined(KOKKOSKERNELS_ETI_ONLY) && !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
 EXECUTE_TEST(double, int64_t, size_t, TestExecSpace)
#endif

#if (defined (KOKKOSKERNELS_INST_FLOAT) \
 && defined (KOKKOSKERNELS_INST_ORDINAL_INT) \
 && defined (KOKKOSKERNELS_INST_OFFSET_INT) ) || (!defined(KOKKOSKERNELS_ETI_ONLY) && !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
 EXECUTE_TEST(float, int, int, TestExecSpace)
#endif

#if (defined (KOKKOSKERNELS_INST_FLOAT) \
 && defined (KOKKOSKERNELS_INST_ORDINAL_INT64_T) \
 && defined (KOKKOSKERNELS_INST_OFFSET_INT) ) || (!defined(KOKKOSKERNELS_ETI_ONLY) && !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
 EXECUTE_TEST(float, int64_t, int, TestExecSpace)
#endif

#if (defined (KOKKOSKERNELS_INST_FLOAT) \
 && defined (KOKKOSKERNELS_INST_ORDINAL_INT) \
 && defined (KOKKOSKERNELS_INST_OFFSET_SIZE_T) ) || (!defined(KOKKOSKERNELS_ETI_ONLY) && !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
 EXECUTE_TEST(float, int, size_t, TestExecSpace)
#endif

#if (defined (KOKKOSKERNELS_INST_FLOAT) \
 && defined (KOKKOSKERNELS_INST_ORDINAL_INT64_T) \
 && defined (KOKKOSKERNELS_INST_OFFSET_SIZE_T) ) || (!defined(KOKKOSKERNELS_ETI_ONLY) && !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
 EXECUTE_TEST(float, int64_t, size_t, TestExecSpace)
#endif


#if (defined (KOKKOSKERNELS_INST_KOKKOS_COMPLEX_DOUBLE_) \
 && defined (KOKKOSKERNELS_INST_ORDINAL_INT) \
 && defined (KOKKOSKERNELS_INST_OFFSET_INT) ) || (!defined(KOKKOSKERNELS_ETI_ONLY) && !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
 EXECUTE_TEST(kokkos_complex_double, int, int, TestExecSpace)
#endif

#if (defined (KOKKOSKERNELS_INST_KOKKOS_COMPLEX_DOUBLE_) \
 && defined (KOKKOSKERNELS_INST_ORDINAL_INT64_T) \
 && defined (KOKKOSKERNELS_INST_OFFSET_INT) ) || (!defined(KOKKOSKERNELS_ETI_ONLY) && !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
 EXECUTE_TEST(kokkos_complex_double, int64_t, int, TestExecSpace)
#endif

#if (defined (KOKKOSKERNELS_INST_KOKKOS_COMPLEX_DOUBLE_) \
 && defined (KOKKOSKERNELS_INST_ORDINAL_INT) \
 && defined (KOKKOSKERNELS_INST_OFFSET_SIZE_T) ) || (!defined(KOKKOSKERNELS_ETI_ONLY) && !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
 EXECUTE_TEST(kokkos_complex_double, int, size_t, TestExecSpace)
#endif

#if (defined (KOKKOSKERNELS_INST_KOKKOS_COMPLEX_DOUBLE_) \
 && defined (KOKKOSKERNELS_INST_ORDINAL_INT64_T) \
 && defined (KOKKOSKERNELS_INST_OFFSET_SIZE_T) ) || (!defined(KOKKOSKERNELS_ETI_ONLY) && !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
 EXECUTE_TEST(kokkos_complex_double, int64_t, size_t, TestExecSpace)
#endif

#if (defined (KOKKOSKERNELS_INST_KOKKOS_COMPLEX_FLOAT_) \
 && defined (KOKKOSKERNELS_INST_ORDINAL_INT) \
 && defined (KOKKOSKERNELS_INST_OFFSET_INT) ) || (!defined(KOKKOSKERNELS_ETI_ONLY) && !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
 EXECUTE_TEST(kokkos_complex_float, int, int, TestExecSpace)
#endif

#if (defined (KOKKOSKERNELS_INST_KOKKOS_COMPLEX_FLOAT_) \
 && defined (KOKKOSKERNELS_INST_ORDINAL_INT64_T) \
 && defined (KOKKOSKERNELS_INST_OFFSET_INT) ) || (!defined(KOKKOSKERNELS_ETI_ONLY) && !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
 EXECUTE_TEST(kokkos_complex_float, int64_t, int, TestExecSpace)
#endif

#if (defined (KOKKOSKERNELS_INST_KOKKOS_COMPLEX_FLOAT_) \
 && defined (KOKKOSKERNELS_INST_ORDINAL_INT) \
 && defined (KOKKOSKERNELS_INST_OFFSET_SIZE_T) ) || (!defined(KOKKOSKERNELS_ETI_ONLY) && !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
 EXECUTE_TEST(kokkos_complex_float, int, size_t, TestExecSpace)
#endif

#if (defined (KOKKOSKERNELS_INST_KOKKOS_COMPLEX_FLOAT_) \
 && defined (KOKKOSKERNELS_INST_ORDINAL_INT64_T) \
 && defined (KOKKOSKERNELS_INST_OFFSET_SIZE_T) ) || (!defined(KOKKOSKERNELS_ETI_ONLY) && !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
 EXECUTE_TEST(kokkos_complex_float, int64_t, size_t, TestExecSpace)
#endif
*/
#undef EXECUTE_TEST
