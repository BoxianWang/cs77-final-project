cmake_minimum_required(VERSION 3.20)
project(cuda CUDA)

set(CMAKE_CUDA_STANDARD 11)

add_executable(main.cu)

target_compile_options(cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-gencode arch=compute_37,code=sm_37>)

set_target_properties(cuda PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
