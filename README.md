# Synopsis
**BFS3D**(Basic Flow Solver 3D) solves the incompressible Navier-Stokes equations via projection method. It uses the direct solver PittPack to solve the Poisson for pressure field and is suitable for extreme-scale computing with accelerators. <br/>
The main goal is to solve the system of incompressible Navier-Stokes equations, consisting of the momentum and continuity equations, on a directionally uniform Cartesian grid on CPU or GPU clusters. Derived from the base class PencilDcmp of the PittPack project, the flow solver class implements second-order explicit time marching for the prediction step, and calls the pittPack() routine provided in PittPack's PoissonCPU or PoissonGPU class to solve the pressure equation in the projection step. Currently, BFS3D supports the boundary conditions present in the lid-driven cavity problem, i.e. specified velocity or no-slip with Neumann condition for pressure on walls incident to the flow motion, and periodic conditions on walls normal to the spanwise direction.     

## Features
  * Uses direct Poisson solver PittPack suitable for extreme-scale computing
  * Hybrid MPI/OpenACC parallelization
  * User-friendly interface   

## Configuration 
BFS3D provides config.sh which can be used to automatically configure the code for Stampede2 (https://www.tacc.utexas.edu/systems/stampede2), Comet (https://www.sdsc.edu/services/hpc/hpc_systems.html) and Bridges (https://www.psc.edu/resources/computing/bridges) clusters 

## Linux 
```
source config.sh 
```
When prompted respond by entering 0 or 1    
* (0) : will configure BFS3D for CPU clusters 
* (1) : will configure BFS3D for GPU clusters



## Installation
BFS3D requires the following libraries
  * cmake 
  * MPI 
  * FFTW3
  * cuFFT
  * OpenACC (available through the PGI C++ compiler)  
  * HDF5 with Parallel IO

##  Build  
BFS3D uses *CMakeLists.txt* and *CMakeModules* folder to detect the library paths. <br/>
These two components are crucial for complilation of BFS3D.
Perform the following steps
```
  cd build
  cmake ..
  make 
```
The executable will be placed in the /bin folder


## Run
```
mpirun -np N ./bin/BFS3D nx ny nz Nsteps Stepsize Viscosity
```
  * N: Number of processes (square of integer number, N = n^2)
  * nx: Number of elements in X-direction on each single processor along x direction (total n\*nx)
  * ny: Number of elements in Y-direction on each single processor along y direction (total n\*ny)
  * nz: Number of elements in Z-direction of each chunk (total n\*nz)  
  * Nsteps: Total number of time steps
  * Stepsize: Time step size  
  * Viscosity: Kinematic viscosity of fluid
  
## Visualization
  * The outputs  are written to the /soln folder
  * Separate files for three velocity components and presure field
  * Paraview can be used to visualize the solution
  * Simply open the file ending with xdmf in soln/ 


## Directory structure
```
PittPack
│   README.md
│   CMakeLists.txt    
│   LICENSE
│   config.sh
│
└─── CMakeModules
│   │   FindPGI.cmake: cmake script to find PGI 
│   │   FindMYMPI.cmake: cmake script to find MPI
│   │   FindMYHDF5.cmake: cmake script to find HDF5
│   │   FindGOOGLETEST.cmake: cmake script to find Google Test
│   │   FindFFTW.cmake: cmake script to find FFTW
│
└─── src
│   │   chunkedArray.cpp:  Abstracts access patterns 
│   │   communicate.cpp:   MPI communications
│   │   mathFunction.cpp:  Defines math functions for kernel generation by PGI
│   │   signalProc.cpp:    Performs FFT transforms
│   │   poissonCPU.cpp:    Inherits from class PencilDcmp and specialized for CPU
│   │   poissonGPU.cpp:    Inherits from class PencilDcmp and specialized for GPU
│   │   pencilDcmp.cpp:    Incorporates Decomposition strategy and communication patterns
|   |   flowSolver.cpp:    Implements class for flow solver 
│   │   triDiag.cpp:       Class for tridiagonal solvers      
│   │   phdf5.cpp:         Class for handling IO with hdf5     
│   │  
│   └─── include (headers)
│       │   pittpack.h     
│       │   chunkedArray.hpp
│       │   communicate.hpp  
│       │   mathFunction.hpp
│       │   signalProc.hpp
│       │   poissonCPU.hpp
│       │   poissonGPU.hpp
│       │   pencilDcmp.hpp
|       |   flowSolver.hpp
│       │   triDiag.hpp 
│       │   phdf5.hpp
│       │   params.h 
│       │   definitions.h 
│   
└─── bin
│       PittPack (executable)  
│  
│
└─── build   
│       Will be populated by cmake   
│  
│
└─── soln 
│   │   Pxdmf3d1.h5: outputs the file in hdf5 format 
│   │   Pxdmf3d1.xmf: meta data to be used by ParaView
│   
└─── archives
 
```



## Notes 
We welcome any feedbacks by the users and developers <br/>
Please read the LICENSE file for how to use this software

## Acknowledgements
**BFS3D** is developed at the Department of Mechanical Engineering at University Pittsburgh, PA, USA. 


## Contributors
  * Jaber J. Hasbestan (jaber@pitt.edu)
  * Inanc Senocak (senocak@pitt.edu)
  * Cheng-Nian Xiao (chx33@pitt.edu)
