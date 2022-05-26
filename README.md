# cs77-final-project
CUDA/C++ ray tracer

## ===========BENCHMARKS===========
All CPU benchmarks were performed on an 5990x. All GPU benchmarks were performed on an RTX 3080.

## Cover of "Raytracing In One Weekend" with 10 samples per pixel
### C++ (using source code of "Raytracing in One Weekend")
This executable is built in `./cpp/cmake-build-debug` and called `inOneWeekend`.
358.848 seconds (Nearly 6 minutes)

### C++ (using source code of "The Next Week")
This executable is built in `./cpp/cmake-build-debug` and called `theNextWeekBenchmark`
12.584 seconds

### CUDA (using "Raytracing In One Weekend")
This executable is built in `./cuda/cmake-build-debug` and called `inOneWeekend`.
1.57348 seconds

### CUDA (using "Raytracing In One Weekend" refactored to use only floats)
This executable is built in `./cuda/cmake-build-debug` and called `inOneWeekendFloat`.
0.424923 seconds (More than 3x faster than using mixed floats and doubles!)


### CUDA (using tree-based acceleration datastructures)
This executable is built in `./cuda/cmake-build-debug` and called `theNextWeekTree`
0.490523 seconds (A little slower...)
This is in large part because the simpler, non-recursive hit strategy required fewer registers and allowed for higher occupancy on the GPU (46% vs 38%)
