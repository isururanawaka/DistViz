cmake_minimum_required(VERSION 3.17)
project(DistViz)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# LocationByValue MPI
message("Finding MPI")
find_package(MPI REQUIRED)
message(STATUS "MPI include dir: ${MPI_CXX_INCLUDE_PATH}")
message(STATUS "MPI libs: ${MPI_CXX_LIBRARIES}")
message(STATUS "MPI executable: ${MPIEXEC_EXECUTABLE}")

#message("Finding MKL")
#find_package(MKL REQUIRED)

#message("Finding openBLAS")
#find_package(OpenBLAS REQUIRED)
#message(STATUS "OpenBLAS include dir: ${OpenBLAS_CXX_INCLUDE_PATH}")
#message(STATUS "OpenBLAS libs: ${OpenBLAS_CXX_LIBRARIES}")
#
#include_directories(${OpenBLAS_CXX_INCLUDE_PATH})


message("Finding OpenMP")
find_package(OpenMP REQUIRED)


#message("Finding PETSC")
#find_package(PETSC REQUIRED)
#include_directories(${PETSC_DIR}/include)
#message(STATUS "PETSC include dir: $ENV{PETSC_DIR}/include")
#message("Finding BLAS")
#find_package(BLAS REQUIRED)
#
#
#include_directories($ENV{COMBLAS_ROOT}/install/include $ENV{MKLROOT}/include $ENV{EIGEN_ROOT}/include)
#link_directories($ENV{COMBLAS_ROOT}/install/lib $ENV{MKLROOT}/lib/intel64)
#include_directories($ENV{PETSC_DIR}/include $ENV{PETSC_ARCH}/include)
#include_directories($ENV{SLEPC_DIR}/include $ENV{SLEPC_ARCH}/include)

include_directories($ENV{COMBLAS_ROOT}/install/include)
link_directories($ENV{COMBLAS_ROOT}/install/lib)


include_directories($ENV{PCG_ROOT}/include)
include_directories($ENV{EIGEN_ROOT})

message("CMAKE_BINARY_PATH ${CMAKE_BINARY_DIR}")
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)

#add_subdirectory(include)
#add_subdirectory(core)
#add_subdirectory(embedding)

#add_library(disviz_lib INTERFACE)
#target_include_directories(disviz_lib INTERFACE cpp/lib)

add_library(disviz_lib SHARED
             cpp/io/file_reader.hpp
             cpp/io/file_writer.hpp
#             cpp/lib/Mrpt.h
             cpp/lib/Mrpt_Sparse.h
             cpp/common/common.h
             cpp/common/common.cpp
             cpp/knng/global_tree_handler.hpp
             cpp/net/process_3D_grid.hpp
             cpp/net/process_3D_grid.cpp
             cpp/knng/math_operations.hpp
             cpp/common/mpi_type_creator.hpp
             cpp/common/json.hpp
             cpp/knng/knng_handler.hpp
             cpp/net/data_comm.hpp
             cpp/embedding/algo.hpp
             cpp/embedding/csr_local.hpp
             cpp/embedding/dense_mat.hpp
             cpp/embedding/distributed_mat.hpp
             cpp/embedding/sparse_mat.hpp
             cpp/embedding/partitioner.hpp
             cpp/embedding/partitioner.cpp)

SET(BUILD_TESTS "" ON)

#if (BUILD_TESTS)
#    add_subdirectory(tests)
#    message(STATUS "Building tests ....")
#endif ()

SET(SOURCES cpp/main.cpp)

add_executable(distviz ${SOURCES})



target_compile_options(distviz PRIVATE -std=c++11 -Ofast -march=native -Icpp -Icpp/lib -DMKL_ILP64  -m64 -fopenmp -I$ENV{MKLROOT}/include -lgomp)
target_link_options(distviz PRIVATE -L$ENV{MKLROOT}/lib/intel64 -Wl,--no-as-needed -lmkl_intel_ilp64 -lmkl_gnu_thread -lmkl_core -lgomp -lpthread -lm -ldl)
#target_link_options(distviz PRIVATE -L$ENV{PETSC_ARCH}/lib)
#target_link_options(distviz PRIVATE -L$ENV{SLEPC_ARCH}/lib)
target_link_libraries(distviz  disviz_lib)
target_link_libraries(distviz mkl_intel_lp64 mkl_sequential mkl_core)
target_link_libraries(distviz MPI::MPI_CXX)
target_link_libraries(distviz OpenMP::OpenMP_CXX)
target_link_libraries(distviz CombBLAS)
#target_link_libraries(distviz petsc.so.3.21)
#target_link_libraries(distviz slepc.so.3.21)
#target_link_libraries(distviz  )
#target_link_libraries(distembed CombBLAS)
#target_link_libraries(distblas $ENV{MKLROOT})

