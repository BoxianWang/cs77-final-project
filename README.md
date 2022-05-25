# cs77-final-project
CUDA/C++ ray tracer

## ===========BENCHMARKS===========
All CPU benchmarks were performed on an 5990x. All GPU benchmarks were performed on an RTX 3080.

## Cover of "Raytracing In One Weekend"
### C++ (using source code of "Raytracing in One Weekend")
This executable is built in `./cpp/cmake-build-debug` and called `inOneWeekend`.
356.496 seconds (Nearly 6 minutes)

### C++ (using source code of "The Next Week")
This executable is built in `./cpp/cmake-build-debug` and called `theNextWeekBenchmark`
123.324 seconds (About 2 minutes)

### CUDA (using "Raytracing In One Weekend")
This executable is built in `./cuda/cmake-build-debug` and called `inOneWeekend`.
76.4971 seconds

### CUDA (using "Raytracing In One Weekend" refactored to use only floats)
This executable is built in `./cuda/cmake-build-debug` and called `inOneWeekendFloat`.
20.0741 seconds (More than 3x faster than using mixed floats and doubles!)
