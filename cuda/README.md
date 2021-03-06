# CUDA CODE FOR GROUP 7
### Elliot Potter, Boxian Wang, Eric (Spencer) Warezak

## Connecting to Discovery
1. ssh <your_net_id>@discovery7.dartmouth.edu
2. type in your standard DartHub password

## Compilation Instructions
1. Make sure that you have a valid account on the Discovery cluster, or access to another Unix-based machine with a NVIDIA gpu.
2. We tested with cmake 3.21.2; higher versions also work.
3. We tested with CUDA 11.2.
4. On Discovery:
```
srun --nodes=1 --ntasks-per-node=1 --partition gpuq --gres=gpu:1 --pty /bin/bash
module load cmake/3.21
module load cuda/11.2
```
5. From the cmake-build-debug directory:
    A. run `cmake ..` This updates the makefile
    B. then run `make`

## Execution Instructions
1. Install eog or some other form of ppm viewing tool on your machine; if you're running a Mac, you can view the ppm directly
    in Preview.
2. Execute
```
cuda > out.ppm
eog out.ppm
```