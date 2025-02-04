cmake_minimum_required(VERSION 3.17)

project(SpMV
    VERSION 0.0.1
    DESCRIPTION "Sparse Matrix Multiplication"
    LANGUAGES C
)

set(CMAKE_C_STANDARD 11)

# Find MPI
find_package(MPI REQUIRED)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Find METIS
find_package(METIS REQUIRED)
find_package(OpenMP REQUIRED)

add_executable(1e 
    src/main_dist.c
    src/spmv.c
    src/mtx.c
)

# Correct the typo for including the project's header files
include_directories(${CMAKE_SOURCE_DIR}/include)
target_include_directories(1e PRIVATE ${MPI_C_INCLUDE_PATH} ${METIS_INCLUDE_DIRS})
target_link_libraries(1e PRIVATE ${MPI_C_LIBRARIES} ${METIS_LIBRARIES} OpenMP::OpenMP_C)
target_compile_options(1e PRIVATE)
target_link_options(1e PRIVATE)
