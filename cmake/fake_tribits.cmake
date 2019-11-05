INCLUDE(CMakeParseArguments)
INCLUDE(CTest)

cmake_policy(SET CMP0054 NEW)

MESSAGE(STATUS "The project name is: ${PROJECT_NAME}")

FUNCTION(ASSERT_DEFINED VARS)
FOREACH(VAR ${VARS})
  IF(NOT DEFINED ${VAR})
    MESSAGE(SEND_ERROR "Error, the variable ${VAR} is not defined!")
  ENDIF()
ENDFOREACH()
ENDFUNCTION()

FUNCTION(kokkoskernels_add_option SUFFIX DEFAULT TYPE DOCSTRING)
  SET(CAMEL_NAME KokkosKernels_${SUFFIX})
  STRING(TOUPPER ${CAMEL_NAME} UC_NAME)

  # Make sure this appears in the cache with the appropriate DOCSTRING
  SET(${CAMEL_NAME} ${DEFAULT} CACHE ${TYPE} ${DOCSTRING})

  #I don't love doing it this way because it's N^2 in number options, but cest la vie
  FOREACH(opt ${KOKKOSKERNELS_GIVEN_VARIABLES})
    STRING(TOUPPER ${opt} OPT_UC)
    IF ("${OPT_UC}" STREQUAL "${UC_NAME}")
      IF (NOT "${opt}" STREQUAL "${CAMEL_NAME}")
        MESSAGE(FATAL_ERROR "Matching option found for ${CAMEL_NAME} with the wrong case ${opt}. Please delete your CMakeCache.txt and change option to -D${CAMEL_NAME}=${${opt}}. This is now enforced to avoid hard-to-debug CMake cache inconsistencies.")
      ENDIF()
    ENDIF()
  ENDFOREACH()

  #okay, great, we passed the validation test - use the default
  IF (DEFINED ${CAMEL_NAME})
    SET(${UC_NAME} ${${CAMEL_NAME}} PARENT_SCOPE)
  ELSE()
    SET(${UC_NAME} ${DEFAULT} PARENT_SCOPE)
  ENDIF()

ENDFUNCTION()

MACRO(KOKKOSKERNELS_ADD_OPTION_AND_DEFINE USER_OPTION_NAME MACRO_DEFINE_NAME DOCSTRING DEFAULT_VALUE )
  KOKKOSKERNELS_ADD_OPTION(${USER_OPTION_NAME} ${DEFAULT_VALUE} BOOL ${DOCSTRING})
  IF (${KOKKOSKERNELS_${USER_OPTION_NAME}})
    SET(${MACRO_DEFINE_NAME} ON)
  ENDIF()
ENDMACRO()

MACRO(KOKKOSKERNELS_ADD_TPL_OPTION NAME DEFAULT_VALUE DOCSTRING)
  KOKKOSKERNELS_ADD_OPTION(ENABLE_TPL_${NAME} ${DEFAULT_VALUE} BOOL ${DOCSTRING})
  IF (DEFINED TPL_ENABLE_${NAME})
    IF (TPL_ENABLE_${NAME} AND NOT KOKKOSKERNELS_ENABLE_TPL_${NAME})
      MESSAGE(WARNING "Overriding KOKKOSKERNELS_ENABLE_TPL_${NAME}=OFF with TPL_ENABLE_${NAME}=ON") 
      SET(KOKKOSKERNELS_ENABLE_TPL_${NAME} ON)
    ELSEIF(NOT TPL_ENABLE_${NAME} AND KOKKOSKERNELS_ENABLE_TPL_${NAME})
      MESSAGE(WARNING "Overriding KOKKOSKERNELS_ENABLE_TPL_${NAME}=ON with TPL_ENABLE_${NAME}=OFF") 
      SET(KOKKOSKERNELS_ENABLE_TPL_${NAME} OFF)
    ENDIF()
  ENDIF()
ENDMACRO()

IF (NOT KOKKOSKERNELS_HAS_TRILINOS)
MACRO(APPEND_GLOB VAR)
  FILE(GLOB LOCAL_TMP_VAR ${ARGN})
  LIST(APPEND ${VAR} ${LOCAL_TMP_VAR})
ENDMACRO()

MACRO(GLOBAL_SET VARNAME)
  SET(${VARNAME} ${ARGN} CACHE INTERNAL "")
ENDMACRO()

FUNCTION(VERIFY_EMPTY CONTEXT)
IF(${ARGN})
 MESSAGE(FATAL_ERROR "Kokkos does not support all of Tribits. Unhandled arguments in ${CONTEXT}:\n${ARGN}")
ENDIF()
ENDFUNCTION()

MACRO(PREPEND_GLOBAL_SET VARNAME)
ASSERT_DEFINED(${VARNAME})
GLOBAL_SET(${VARNAME} ${ARGN} ${${VARNAME}})
ENDMACRO()

MACRO(PREPEND_TARGET_SET VARNAME TARGET_NAME TYPE)
IF(TYPE STREQUAL "REQUIRED")
  SET(REQUIRED TRUE)
ELSE()
  SET(REQUIRED FALSE)
ENDIF()
IF(TARGET ${TARGET_NAME})
  PREPEND_GLOBAL_SET(${VARNAME} ${TARGET_NAME})
ELSE()
  IF(REQUIRED)
    MESSAGE(FATAL_ERROR "Missing dependency ${TARGET_NAME}")
  ENDIF()
ENDIF()
ENDMACRO()
ENDIF(NOT KOKKOSKERNELS_HAS_TRILINOS)

FUNCTION(KOKKOSKERNELS_CONFIGURE_FILE  PACKAGE_NAME_CONFIG_FILE)
  if (KOKKOSKERNELS_HAS_TRILINOS)
    TRIBITS_CONFIGURE_FILE(${PACKAGE_NAME_CONFIG_FILE})
  else()
    # Configure the file
    CONFIGURE_FILE(
      ${PACKAGE_SOURCE_DIR}/cmake/${PACKAGE_NAME_CONFIG_FILE}.in
      ${CMAKE_CURRENT_BINARY_DIR}/${PACKAGE_NAME_CONFIG_FILE}
      )
  endif()
ENDFUNCTION(KOKKOSKERNELS_CONFIGURE_FILE)

MACRO(KOKKOSKERNELS_ADD_TEST_DIRECTORIES)
  if (KOKKOSKERNELS_HAS_TRILINOS)
    TRIBITS_ADD_TEST_DIRECTORIES(${ARGN})
  else()
    IF(${${PROJECT_NAME}_ENABLE_TESTS})
      FOREACH(TEST_DIR ${ARGN})
        ADD_SUBDIRECTORY(${TEST_DIR})
      ENDFOREACH()
    ENDIF()
  endif()
ENDMACRO(KOKKOSKERNELS_ADD_TEST_DIRECTORIES)

MACRO(KOKKOSKERNELS_ADD_EXAMPLE_DIRECTORIES)
  if (KOKKOSKERNELS_HAS_TRILINOS)
    TRIBITS_ADD_EXAMPLE_DIRECTORIES(${ARGN})
  else()
    IF(KOKKOSKERNELS_ENABLE_EXAMPLES)
      FOREACH(EXAMPLE_DIR ${ARGN})
        ADD_SUBDIRECTORY(${EXAMPLE_DIR})
      ENDFOREACH()
    ENDIF()
  endif()
ENDMACRO(KOKKOSKERNELS_ADD_EXAMPLE_DIRECTORIES)

MACRO(ADD_INTERFACE_LIBRARY LIB_NAME)
FILE(WRITE ${CMAKE_CURRENT_BINARY_DIR}/dummy.cpp "")
ADD_LIBRARY(${LIB_NAME} STATIC ${CMAKE_CURRENT_BINARY_DIR}/dummy.cpp)
SET_TARGET_PROPERTIES(${LIB_NAME} PROPERTIES INTERFACE TRUE)
ENDMACRO(ADD_INTERFACE_LIBRARY)

FUNCTION(KOKKOSKERNELS_ADD_EXECUTABLE EXE_NAME)
  IF (KOKKOSKERNELS_HAS_TRILINOS)
    TRIBITS_ADD_EXECUTABLE(${EXE_NAME} ${ARGN})
  ELSE()
    CMAKE_PARSE_ARGUMENTS(PARSE 
      "TESTONLY"
      ""
      "SOURCES;TESTONLYLIBS"
      ${ARGN})

    ADD_EXECUTABLE(${EXE_NAME} ${PARSE_SOURCES})
    IF (PARSE_TESTONLYLIBS)
      TARGET_LINK_LIBRARIES(${EXE_NAME} ${PARSE_TESTONLYLIBS})
    ENDIF()
    TARGET_LINK_LIBRARIES(${EXE_NAME} kokkoskernels)
    VERIFY_EMPTY(KOKKOSKERNELS_ADD_EXECUTABLE ${PARSE_UNPARSED_ARGUMENTS})
  ENDIF()
ENDFUNCTION(KOKKOSKERNELS_ADD_EXECUTABLE)

IF(NOT TARGET check)
  ADD_CUSTOM_TARGET(check COMMAND ${CMAKE_CTEST_COMMAND} -VV -C ${CMAKE_CFG_INTDIR})
ENDIF()

FUNCTION(KOKKOSKERNELS_ADD_TEST)
IF (KOKKOSKERNELS_HAS_TRILINOS)
  CMAKE_PARSE_ARGUMENTS(TEST 
    ""
    "EXE;NAME"
    ""
    ${ARGN})
  IF(TEST_EXE)
    SET(EXE_ROOT ${TEST_EXE})
  ELSE()
    SET(EXE_ROOT ${TEST_NAME})
  ENDIF()

  TRIBITS_ADD_TEST(
    ${EXE_ROOT}
    NAME ${TEST_NAME}
    ${ARGN} 
    COMM serial mpi
    NUM_MPI_PROCS 1
    ${TEST_UNPARSED_ARGUMENTS}
  )
ELSE()
  CMAKE_PARSE_ARGUMENTS(TEST 
    "WILL_FAIL"
    "FAIL_REGULAR_EXPRESSION;PASS_REGULAR_EXPRESSION;EXE;NAME"
    "CATEGORIES"
    ${ARGN})
  IF(TEST_EXE)
    SET(EXE ${TEST_EXE})
  ELSE()
    SET(EXE ${TEST_NAME})
  ENDIF()
  IF(WIN32)
    ADD_TEST(NAME ${TEST_NAME} WORKING_DIRECTORY ${LIBRARY_OUTPUT_PATH} COMMAND ${EXE}${CMAKE_EXECUTABLE_SUFFIX})
  ELSE()
    ADD_TEST(NAME ${TEST_NAME} COMMAND ${EXE})
  ENDIF()
  IF(TEST_WILL_FAIL)
    SET_TESTS_PROPERTIES(${TEST_NAME} PROPERTIES WILL_FAIL ${TEST_WILL_FAIL})
  ENDIF()
  IF(TEST_FAIL_REGULAR_EXPRESSION)
    SET_TESTS_PROPERTIES(${TEST_NAME} PROPERTIES FAIL_REGULAR_EXPRESSION ${TEST_FAIL_REGULAR_EXPRESSION})
  ENDIF()
  IF(TEST_PASS_REGULAR_EXPRESSION)
    SET_TESTS_PROPERTIES(${TEST_NAME} PROPERTIES PASS_REGULAR_EXPRESSION ${TEST_PASS_REGULAR_EXPRESSION})
  ENDIF()
  VERIFY_EMPTY(KOKKOSKERNELS_ADD_TEST ${TEST_UNPARSED_ARGUMENTS})
ENDIF()
ENDFUNCTION()

FUNCTION(KOKKOSKERNELS_ADD_ADVANCED_TEST)
  IF (KOKKOSKERNELS_HAS_TRILINOS)
    TRIBITS_ADD_ADVANCED_TEST(${ARGN})
  ELSE()
    # TODO WRITE THIS
  ENDIF()
ENDFUNCTION()

FUNCTION(KOKKOSKERNELS_ADD_EXECUTABLE_AND_TEST ROOT_NAME)
IF (KOKKOSKERNELS_HAS_TRILINOS)
  TRIBITS_ADD_EXECUTABLE_AND_TEST(
    ${ROOT_NAME} 
    TESTONLYLIBS kokkoskernels_gtest
    ${ARGN}
    NUM_MPI_PROCS 1
    COMM serial mpi
    FAIL_REGULAR_EXPRESSION "  FAILED  "
  )
ELSE()
  CMAKE_PARSE_ARGUMENTS(PARSE 
    ""
    ""
    "SOURCES;CATEGORIES"
    ${ARGN})
  VERIFY_EMPTY(KOKKOSKERNELS_ADD_EXECUTABLE_AND_TEST ${PARSE_UNPARSED_ARGUMENTS})
  SET(EXE_NAME ${PACKAGE_NAME}_${ROOT_NAME})
  KOKKOSKERNELS_ADD_TEST_EXECUTABLE(${EXE_NAME}
    SOURCES ${PARSE_SOURCES}
  )
  KOKKOSKERNELS_ADD_TEST(NAME ${ROOT_NAME}
    EXE ${EXE_NAME}
    FAIL_REGULAR_EXPRESSION "  FAILED  "
  )
ENDIF()
ENDFUNCTION(KOKKOSKERNELS_ADD_EXECUTABLE_AND_TEST)


MACRO(KOKKOSKERNELS_EXCLUDE_AUTOTOOLS_FILES)
  IF (KOKKOSKERNELS_HAS_TRILINOS)
    TRIBITS_EXCLUDE_AUTOTOOLS_FILES()
  ELSE()
    #DO nothing
  ENDIF()
ENDMACRO(KOKKOSKERNELS_EXCLUDE_AUTOTOOLS_FILES)

MACRO(KOKKOSKERNELS_ADD_TEST_EXECUTABLE EXE_NAME)
CMAKE_PARSE_ARGUMENTS(PARSE 
  ""
  ""
  "SOURCES"
  ${ARGN})
KOKKOSKERNELS_ADD_EXECUTABLE(${EXE_NAME}
  SOURCES ${PARSE_SOURCES}
  TESTONLYLIBS kokkos_gtest 
  ${PARSE_UNPARSED_ARGUMENTS}
)
ADD_DEPENDENCIES(check ${EXE_NAME})
ENDMACRO(KOKKOSKERNELS_ADD_TEST_EXECUTABLE)

FUNCTION(KOKKOSKERNELS_LIB_TYPE LIB RET)
GET_TARGET_PROPERTY(PROP ${LIB} TYPE)
IF (${PROP} STREQUAL "INTERFACE_LIBRARY")
  SET(${RET} "INTERFACE" PARENT_SCOPE)
ELSE()
  SET(${RET} "PUBLIC" PARENT_SCOPE)
ENDIF()
ENDFUNCTION(KOKKOSKERNELS_LIB_TYPE)

FUNCTION(KOKKOSKERNELS_ADD_TEST_LIBRARY NAME)
IF (KOKKOSKERNELS_HAS_TRILINOS)
  TRIBITS_ADD_LIBRARY(${NAME} ${ARGN} TESTONLY
   ADDED_LIB_TARGET_NAME_OUT ${NAME}
  )
ELSE()
  SET(oneValueArgs)
  SET(multiValueArgs HEADERS SOURCES)

  CMAKE_PARSE_ARGUMENTS(PARSE 
    "STATIC;SHARED"
    ""
    "HEADERS;SOURCES"
    ${ARGN})

  IF(PARSE_HEADERS)
    LIST(REMOVE_DUPLICATES PARSE_HEADERS)
  ENDIF()
  IF(PARSE_SOURCES)
    LIST(REMOVE_DUPLICATES PARSE_SOURCES)
  ENDIF()
  ADD_LIBRARY(${NAME} ${PARSE_SOURCES})
ENDIF()
ENDFUNCTION(KOKKOSKERNELS_ADD_TEST_LIBRARY)


FUNCTION(KOKKOSKERNELS_INCLUDE_DIRECTORIES)
IF(KOKKOSKERNELS_HAS_TRILINOS)
  TRIBITS_INCLUDE_DIRECTORIES(${ARGN})
ELSE()
  CMAKE_PARSE_ARGUMENTS(
    INC
    "REQUIRED_DURING_INSTALLATION_TESTING"
    ""
    ""
    ${ARGN}
  )
  INCLUDE_DIRECTORIES(${INC_UNPARSED_ARGUMENTS})
ENDIF()
ENDFUNCTION(KOKKOSKERNELS_INCLUDE_DIRECTORIES)


MACRO(PRINTALL)
GET_CMAKE_PROPERTY(_variableNames VARIABLES)
LIST (SORT _variableNames)
FOREACH (_variableName ${_variableNames})
  MESSAGE(STATUS "${_variableName}=${${_variableName}}")
ENDFOREACH()
ENDMACRO(PRINTALL)

MACRO(KOKKOSKERNELS_ADD_DEBUG_OPTION)
  IF(KOKKOSKERNELS_HAS_TRILINOS)
    TRIBITS_ADD_DEBUG_OPTION()
  ENDIF()
ENDMACRO()
