# GROUP 7
## Elliot Potter, Boxian Wang, Eric (Spencer) Warezak
## CUDA/C++ ray tracer

## ===========CODE STRUCTURE============
1. All CUDA code is in the `/cuda` directory.
2. All C++ code is in the `cpp` directory
3. Compilation instructions for cuda can be found in the README of the `/cuda` directory.
4. CUDA executables are built into the `cuda/cmake-build-debug` directory
	a. `inOneWeekend`: basic rendering of the final scene of the first book
	b. `inOneWeekendFloat`: the same thing, at 1/3 the time
	c. `inOneWeekendTree`: implements motion blur and tree-based object collisions
	d. `inOneWeekendSmarterTree`: the same thing, except the tree builder attempts
		to minimize the volume of each layer of the tree.
5. Each CUDA executable is backed by a directory. They have the following structure:
	a. `materials` is a directory of classes that extend `material`
	b. `objects` is a directory of classes that can be collided with, though they 
		don't all extend "hittable"
	c. a `camera` class
	d. a `randoms` class
	e. a `ray` class
	d. main, which contains:
		i. Error handling functions
		ii. Functions to create and delete the world (in some cases creating
			the tree that backs the world)
		iii. Functions to create and delete the camera
		iv. Functions to actually manage tracing the ray
		v. The main function, which allocates memory and manages all the GPU-CPU
			interactions

## ==============WORK DONE==============
### Work that is entirely our own:
1. Benchmarking the various implementations
2. The differences between `cuda/theNextWeek/tree` and `cuda/theNextWeek/smarter-trees`.
These optimizations were surprisingly difficult, despite being simple.

### Work that is a mix of our own and other sources:
1. `cuda/inOneWeekend`. The first four sections (in the `/sections` subdirectory) is
almost entirely directly copy-pasted from the NVIDIA blog. Everything else was C++ in the
"Ray Tracing in One Weekend" book, which we CUDA-ified and debugged.
2. `cuda/inOneWeekendFloats`. The NVIDIA blog suggested using mixed precision would be
slower. They were right.
3. `cuda/theNextWeek/tree`. The C++ book provided the guidelines for implementing the
tree, but the majority of C++ operations did not have good analogs within CUDA
(this is partly because we chose to use pointer arrays rather than a CUDA vector analog).
Therefore, the construction of the trees is mostly our own. We used a sorting algorithm
from GeeksForGeeks in here as well.  

### Work that is entirely borrowed
1. The entire cpp directory

## ==============RESOURCES===============
### C++ BASICS AND RAYTRACING GUIDE
We followed Books 1 and 2 of "Ray Tracing in One Weekend". We generally read the code
in the books themselves, and copy-pasted functions for later modifications, but did
not use the GitHub extensively.
#### Book 1: https://raytracing.github.io/books/RayTracingInOneWeekend.html
#### Book 2: https://raytracing.github.io/books/RayTracingTheNextWeek.html 

### CUDA
We followed the NVIDIA blog on Book 1 of "Ray Tracing in One Weekend". Everything up
to about Chapter 8 of the Ray Tracing book is copy-pasted from this blog. Everything
past this point is 

### SORTING ALGORITHM
We slightly modified an insertion sort algorithm from GeeksForGeeks for use in the
tree-based CUDA program.
https://www.geeksforgeeks.org/insertion-sort/

### K-MEANS (TODO)

## ============PRESENTATION============
https://docs.google.com/presentation/d/1PKAS5HJpFJFtKRe6GA9VrKXgAXtUfojzbELVClpj-rg/edit?usp=sharing

## =============BENCHMARKS=============
All CPU benchmarks were performed on a Ryzen 5950x. All GPU benchmarks were performed on an RTX 3080.

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
Larger scene: 1.52617 seconds

### CUDA (using tree-based acceleration datastructures)
This executable is built in `./cuda/cmake-build-debug` and called `theNextWeekTree`
0.490523 seconds (A little slower...)
This is in large part because the simpler, non-recursive hit strategy required fewer registers and allowed for higher occupancy on the GPU (46% vs 38%)
Larger scene: 0.633864 seconds

### CUDA (using intelligent xyz trees)
0.390624 seconds (A little faster...)
Larger scene: 0.597435 seconds
