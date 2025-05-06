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

# Find METIS and OpenMP
find_package(METIS REQUIRED)
find_package(OpenMP REQUIRED)

# Executables
add_executable(0a 
    src/option0.c
    src/p2.c
    src/p2.h
    src/mtx.c
    src/mtx.h
    src/spmv.c
    src/spmv.h
)

add_executable(1a 
    src/option1a.c
    src/p2.c
    src/p2.h
    src/mtx.c
    src/mtx.h
    src/spmv.c
    src/spmv.h
)

add_executable(1b 
    src/option1b.c
    src/p2.c
    src/p2.h
    src/mtx.c
    src/mtx.h
    src/spmv.c
    src/spmv.h
)

add_executable(1c 
    src/option1c.c
    src/p2.c
    src/p2.h
    src/mtx.c
    src/mtx.h
    src/spmv.c
    src/spmv.h
)

add_executable(1d 
    src/option1d.c
    src/p2.c
    src/p2.h
    src/mtx.c
    src/mtx.h
    src/spmv.c
    src/spmv.h
)

# include headers
include_directories(${CMAKE_SOURCE_DIR}/include)

# Target settings (linking and compile flags)
foreach(target 0a 1a 1b 1c 1d)
    target_include_directories(${target} PRIVATE ${MPI_C_INCLUDE_PATH} ${METIS_INCLUDE_DIRS})
    target_link_libraries(${target} PRIVATE ${MPI_C_LIBRARIES} ${METIS_LIBRARIES} OpenMP::OpenMP_C m)
    target_compile_options(${target} PRIVATE -O3 -march=native)
endforeach()
