cmake_minimum_required(VERSION 3.17)

project(SpMV
    VERSION 0.0.1
    DESCRIPTION "Sparse Matrix Multiplication"
    LANGUAGES C
)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3")

# Find MPI
find_package(MPI REQUIRED)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Find METIS
find_package(METIS REQUIRED)
find_package(OpenMP REQUIRED)

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

# add_executable(test 
#     src/test_bcast.c
# )

# Correct the typo for including the project's header files
include_directories(${CMAKE_SOURCE_DIR}/include)
target_include_directories(1a PRIVATE ${MPI_C_INCLUDE_PATH} ${METIS_INCLUDE_DIRS})
target_link_libraries(1a PRIVATE ${MPI_C_LIBRARIES} ${METIS_LIBRARIES} OpenMP::OpenMP_C m)
target_compile_options(1a PRIVATE)
target_link_options(1a PRIVATE)

target_include_directories(1b PRIVATE ${MPI_C_INCLUDE_PATH} ${METIS_INCLUDE_DIRS})
target_link_libraries(1b PRIVATE ${MPI_C_LIBRARIES} ${METIS_LIBRARIES} OpenMP::OpenMP_C m)
target_compile_options(1b PRIVATE)
target_link_options(1b PRIVATE)


target_include_directories(1c PRIVATE ${MPI_C_INCLUDE_PATH} ${METIS_INCLUDE_DIRS})
target_link_libraries(1c PRIVATE ${MPI_C_LIBRARIES} ${METIS_LIBRARIES} OpenMP::OpenMP_C m)
target_compile_options(1c PRIVATE)
target_link_options(1c PRIVATE)

target_include_directories(1d PRIVATE ${MPI_C_INCLUDE_PATH} ${METIS_INCLUDE_DIRS})
target_link_libraries(1d PRIVATE ${MPI_C_LIBRARIES} ${METIS_LIBRARIES} OpenMP::OpenMP_C m)
target_compile_options(1d PRIVATE)
target_link_options(1d PRIVATE)

target_include_directories(0a PRIVATE ${MPI_C_INCLUDE_PATH} ${METIS_INCLUDE_DIRS})
target_link_libraries(0a PRIVATE ${MPI_C_LIBRARIES} ${METIS_LIBRARIES} OpenMP::OpenMP_C m)
target_compile_options(0a PRIVATE)
target_link_options(0a PRIVATE)


# target_include_directories(test PRIVATE ${MPI_C_INCLUDE_PATH} ${METIS_INCLUDE_DIRS})
# target_link_libraries(test PRIVATE ${MPI_C_LIBRARIES} ${METIS_LIBRARIES} OpenMP::OpenMP_C m)
# target_compile_options(test PRIVATE)
# target_link_options(test PRIVATE)
