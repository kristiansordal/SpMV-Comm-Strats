cmake_minimum_required(VERSION 3.17)

project(SpMV
    VERSION 0.0.1
    DESCRIPTION "Sparse Matrix Multiplication"
    LANGUAGES C
)

set(CMAKE_C_STANDARD 11)

# Use Release mode with optimizations
set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type")
set(CMAKE_C_FLAGS_RELEASE "-O3 -march=native")

# Find packages
find_package(MPI REQUIRED)
find_package(METIS REQUIRED)
find_package(OpenMP REQUIRED)

# Add include directories
include_directories(${CMAKE_SOURCE_DIR}/include)

# Common source files
set(COMMON_SOURCES
    src/p2.c
    src/p2.h
    src/mtx.c
    src/mtx.h
    src/spmv.c
    src/spmv.h
)

# Helper macro to avoid repeating the linking setup
function(add_spmv_executable name file)
    add_executable(${name} src/${file} ${COMMON_SOURCES})
    target_include_directories(${name} PRIVATE ${MPI_C_INCLUDE_PATH} ${METIS_INCLUDE_DIRS})
    target_link_libraries(${name} PRIVATE ${MPI_C_LIBRARIES} ${METIS_LIBRARIES} OpenMP::OpenMP_C m)
    target_compile_options(${name} PRIVATE -O3 -march=native)
endfunction()

# Create executables
add_spmv_executable(0a option0.c)
add_spmv_executable(1a option1a.c)
add_spmv_executable(1b option1b.c)
add_spmv_executable(1c option1c.c)
add_spmv_executable(1d option1d.c)

# Optional test target
# add_spmv_executable(test test_bcast.c)
