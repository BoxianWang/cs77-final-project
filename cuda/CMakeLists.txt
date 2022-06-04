cmake_minimum_required(VERSION 3.20)
project(cuda CUDA)

set(CMAKE_CUDA_STANDARD 11)

add_executable(cuda main.cu)
add_executable(inOneWeekend ./inOneWeekend/main.cu)
add_executable(inOneWeekendFloats ./inOneWeekendFloats/main.cu)
add_executable(theNextWeekTree ./theNextWeek/tree/main.cu)
add_executable(theNextWeekRest ./theNextWeek/rest/main.cu)
add_executable(theNextWeekSmarterTree ./theNextWeek/smarter-trees/main.cu)
add_executable(theNextWeekUnrolled ./theNextWeek/unrolled-supersampling/main.cu)
add_executable(trace ./advancedTracer/main.cu)

set(CMAKE_CUDA_FLAGS "--use_fast_math -O3 -Xptxas -O3, -Xcompiler -O3")
target_compile_options(cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-gencode arch=compute_37,code=sm_37>)
target_compile_options(inOneWeekend PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-gencode arch=compute_37,code=sm_37>)
target_compile_options(inOneWeekendFloats PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-gencode arch=compute_37,code=sm_37>)
target_compile_options(theNextWeekTree PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-gencode arch=compute_37,code=sm_37>)
target_compile_options(theNextWeekRest PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-gencode arch=compute_37,code=sm_37>)
target_compile_options(theNextWeekSmarterTree PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-gencode arch=compute_37,code=sm_37>)
target_compile_options(theNextWeekUnrolled PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-gencode arch=compute_37,code=sm_37>)
target_compile_options(trace PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-gencode arch=compute_37,code=sm_37>)

set_target_properties(cuda PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(inOneWeekend PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(inOneWeekendFloats PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(theNextWeekTree PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(theNextWeekRest PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(theNextWeekSmarterTree PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(theNextWeekUnrolled PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(trace PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)