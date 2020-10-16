#if defined(HAVE_KOKKOS_HALFMATH)
TEST_F( TestCategory, batched_scalar_serial_gemm_nt_nt_half_half ) {
  typedef ::Test::ParamTag<Trans::NoTranspose,Trans::NoTranspose> param_tag_type;

  test_batched_gemm_half<TestExecSpace,::Test::halfScalarType,::Test::halfScalarType,param_tag_type,Algo::Gemm::Blocked>();
  test_batched_gemm_half<TestExecSpace,::Test::halfScalarType,::Test::halfScalarType,param_tag_type,Algo::Gemm::Unblocked>();
}
TEST_F( TestCategory, batched_scalar_serial_gemm_t_nt_half_half ) {
  typedef ::Test::ParamTag<Trans::Transpose,Trans::NoTranspose> param_tag_type;

  test_batched_gemm_half<TestExecSpace,::Test::halfScalarType,::Test::halfScalarType,param_tag_type,Algo::Gemm::Blocked>();
  test_batched_gemm_half<TestExecSpace,::Test::halfScalarType,::Test::halfScalarType,param_tag_type,Algo::Gemm::Unblocked>();
}
TEST_F( TestCategory, batched_scalar_serial_gemm_nt_t_half_half ) {
  typedef ::Test::ParamTag<Trans::NoTranspose,Trans::Transpose> param_tag_type;

  test_batched_gemm_half<TestExecSpace,::Test::halfScalarType,::Test::halfScalarType,param_tag_type,Algo::Gemm::Blocked>();
  test_batched_gemm_half<TestExecSpace,::Test::halfScalarType,::Test::halfScalarType,param_tag_type,Algo::Gemm::Unblocked>();
}
TEST_F( TestCategory, batched_scalar_serial_gemm_t_t_half_half ) {
  typedef ::Test::ParamTag<Trans::Transpose,Trans::Transpose> param_tag_type;

  test_batched_gemm_half<TestExecSpace,::Test::halfScalarType,::Test::halfScalarType,param_tag_type,Algo::Gemm::Blocked>();
  test_batched_gemm_half<TestExecSpace,::Test::halfScalarType,::Test::halfScalarType,param_tag_type,Algo::Gemm::Unblocked>();
}
#endif // HAVE_KOKKOS_HALFMATH

#if defined(KOKKOSKERNELS_INST_FLOAT)
TEST_F( TestCategory, batched_scalar_serial_gemm_nt_nt_float_float ) {
  typedef ::Test::ParamTag<Trans::NoTranspose,Trans::NoTranspose> param_tag_type;
  typedef Algo::Gemm::Blocked algo_tag_type;
  test_batched_gemm<TestExecSpace,float,float,param_tag_type,algo_tag_type>();
}
TEST_F( TestCategory, batched_scalar_serial_gemm_t_nt_float_float ) {
  typedef ::Test::ParamTag<Trans::Transpose,Trans::NoTranspose> param_tag_type;
  typedef Algo::Gemm::Blocked algo_tag_type;
  test_batched_gemm<TestExecSpace,float,float,param_tag_type,algo_tag_type>();
}
TEST_F( TestCategory, batched_scalar_serial_gemm_nt_t_float_float ) {
  typedef ::Test::ParamTag<Trans::NoTranspose,Trans::Transpose> param_tag_type;
  typedef Algo::Gemm::Blocked algo_tag_type;
  test_batched_gemm<TestExecSpace,float,float,param_tag_type,algo_tag_type>();
}
TEST_F( TestCategory, batched_scalar_serial_gemm_t_t_float_float ) {
  typedef ::Test::ParamTag<Trans::Transpose,Trans::Transpose> param_tag_type;
  typedef Algo::Gemm::Blocked algo_tag_type;
  test_batched_gemm<TestExecSpace,float,float,param_tag_type,algo_tag_type>();
}
#endif

#if defined(KOKKOSKERNELS_INST_DOUBLE)
TEST_F( TestCategory, batched_scalar_serial_gemm_nt_nt_double_double ) {
  typedef ::Test::ParamTag<Trans::NoTranspose,Trans::NoTranspose> param_tag_type;
  typedef Algo::Gemm::Blocked algo_tag_type;
  test_batched_gemm<TestExecSpace,double,double,param_tag_type,algo_tag_type>();
}
TEST_F( TestCategory, batched_scalar_serial_gemm_t_nt_double_double ) {
  typedef ::Test::ParamTag<Trans::Transpose,Trans::NoTranspose> param_tag_type;
  typedef Algo::Gemm::Blocked algo_tag_type;
  test_batched_gemm<TestExecSpace,double,double,param_tag_type,algo_tag_type>();
}
TEST_F( TestCategory, batched_scalar_serial_gemm_nt_t_double_double ) {
  typedef ::Test::ParamTag<Trans::NoTranspose,Trans::Transpose> param_tag_type;
  typedef Algo::Gemm::Blocked algo_tag_type;
  test_batched_gemm<TestExecSpace,double,double,param_tag_type,algo_tag_type>();
}
TEST_F( TestCategory, batched_scalar_serial_gemm_t_t_double_double ) {
  typedef ::Test::ParamTag<Trans::Transpose,Trans::Transpose> param_tag_type;
  typedef Algo::Gemm::Blocked algo_tag_type;
  test_batched_gemm<TestExecSpace,double,double,param_tag_type,algo_tag_type>();
}
#endif


