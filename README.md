# CUDACFD_out
CUDA C/C++ scripts for Computational Fluid Dynamics (CFD) for presentation purposes (that goes out)

## Abstract:

*20161228 update*:I implemented, from "soup to nuts", a 2-dimensional Navier-Stokes equations solver for viscous, incompressible fluids with finite difference methods, in CUDA C++11, along with using the thrust library, to run entirely in parallel on the GPU (graphics processing unit).  The implementation beats previous implementations in terms of speed, by taking advantage of vectors from thrust that reside entirely on the GPU, taking advantage of the asynchronous nature of CUDA threads in the Poisson solver for pressure, and removing the CPU-GPU communication bottleneck, such as in the case of Niemeyer and Sung (2013), by using the parallel reduce algorithm (thrust) computed directly on the GPU, for summations, and in terms of scalability, with a scalable memory access pattern in each of the 2-dimensions, based upon my previous work (available on my github repository, CUDACFD_out).  The lid-driven cavity is a classic computational fluid dynamics (CFD) problem as a validation test for CFD routines, where a fluid fills a square container and the so-called "lid" of the container, either at the "top" or "right" is given a constant velocity, thereby, setting the fluid in motion.  Contour plots of the components of the velocity vector fields were made with Python's pandas, matplotlib, and displayed in jupyter notebooks.  For the purposes of demonstrating real-time visualization (video), I modularized (i.e. wrote C++11 classes) Niemeyer and Sung's original code, with a red-black Gauss-Seidel method with successive over-relaxation (SOR) for the Poisson solver of the pressure, and used my previous OpenGL C++11 class to texture graphics for the components of the velocity vector field rendered entirely on the GPU.  In both cases, one, in terms of software architecture, I implemented the best practices with C++11 in terms of modularity (classes and objects) and using containers (e.g. vectors, arrays, both on the host CPU and device GPU).  Two, writing parallelized code in CUDA C++11 for the GPU allows for scalable grids to grid sizes (in my case, 512x512 and above) much larger than previous work (cf. Griebel et. al, Marchi, et. al.).  The big idea is that with these higher resolutions, we're able to resolve physical phenomenon literally unseen before, such as the small eddies that form around the large eddy in the cavity's center, as clearly shown in the video from 256x256 grid to 512x512 grid.  


- C++14 standard on host CPU code/C++11 standard on device GPU as of CUDA Toolkit 7.5 

### (Abridged) Table of Contents

| filename (or directory) | directory | Description | related YouTube link (if there is one) |
| :---------------------- | :-------: | :---------: | -------------------------------------: |
| `CUDACFD_writeup01.tex` | `./LaTeXandpdfs/` | LaTeX file for the writeup explaining in depth the theory, concepts, and implementation of finite difference methods on CUDA C/C++ | [presentation the writeup accompanies](https://youtu.be/xQnEQMrol5I)  |
| `CUDACFD_writeup01.pdf` | `./LaTeXandpdfs/` | Compiled pdf file for writeup | [presentation the writeup accompanies](https://youtu.be/xQnEQMrol5I) |
| `CUDACFD_slides01.tex`  | `./LaTeXandpdfs/` | LaTeX file for slide deck for the main YouTube presentation | [presentation that uses the slide deck](https://youtu.be/xQnEQMrol5I) |
| `CUDACFD_slides01.pdf`  | `./LaTeXandpdfs/` | Compiled pdf file for slide deck for the main YouTube presentation | [presentation that uses the slide deck](https://youtu.be/xQnEQMrol5I) |
| `convect1dupwind/`   | `./convect1dupwind`  | 1-dimensional convection according to mass conservation; initial condition for mass density is a gaussian distribution by "upwind" finite volume method | [screen capture of a convect1dupwind run](https://youtu.be/mRJGl0yfiH8) |
| `convect3dupwind/`   | `./convect3dupwind`  | 3-dimensional convection according to mass conservation; initial condition for mass density is a gaussian distribution by "upwind" finite volume method | [screen capture of a convect3dupwind run](https://youtu.be/s1H1zDkpwTQ) |
| `heat2d/`            | `./heat2d`           | 2-dimensional heat conduction according to the heat equation; CUDA C++ classes has been entirely factored to allow for finite difference method to any order of accuracy | [screen capture of heat2d run](https://youtu.be/oAkOcf8g_TQ)  |
| `heat3d/`            | `./heat3d`           | 3-dimensional heat conduction according to the heat equation; CUDA C++ classes has been entirely factored to allow for finite difference method to any order of accuracy | [screen capture of heat3d run](https://youtu.be/LTl9eGs_oIA)  |
| `convect2dfinitedifftex/`   | `./convect2dfinitedifftex`  | 2-dimensional convection according to mass conservation by finite difference method with OpenGL 2-dimensional texture graphics; initial condition for mass density is a gaussian distribution | [screen capture of a convect2dfinitedifftex run](https://youtu.be/RGQOIX70jvg) |
| `convect3dfinitedifftex/`   | `./convect3dfinitedifftex`  | 3-dimensional convection according to mass conservation by finite difference method with OpenGL 2-dimensional texture graphics; initial condition for mass density is a gaussian distribution | [screen capture of a convect3dfinitedifftex run](https://youtu.be/mhXK6xtMz44) |
| `Euler2d/`   | `./Euler2d`  | Numerical computation of 2-dimensional Euler equations by finite difference method with OpenGL 2-dimensional texture graphics; initial condition for mass density is a gaussian distribution | [screen capture of a Euler2d run](https://youtu.be/KLIfUj3V8pY) |
| `lid-driven-cavity_gpu-gfx/' | './lid-driven-cavity_gpu-gfx' | *2-dim. Navier-Stokes equations solver for viscous, incompressible fluids with finite difference* methods, in CUDA C++11, with initial parameters set for the **lid-driven cavity** problem. I modularized ((i.e. wrote C++11 classes) Niemeyer and Sung's original code, with a red-black Gauss-Seidel method with successive over-relaxation (SOR) for the Poisson solver of the pressure, and used my previous OpenGL C++11 class to texture graphics for the components of the velocity vector field rendered entirely on the GPU. | [Lid-driven cavity CFD,512x512,Re=100000, CUDA C++11, w/ finite diff for incompress. Nav.-Stokes eq](https://youtu.be/6ciU1YiKPC0), [Lid-driven cavity CFD,256x256,Re=100000, CUDA C++11, w/ finite diff for incompress. Nav.-Stokes eq](https://youtu.be/_E33hmzK3Pw) |
|
| `NavSt2DIncompFiniteDiff/' | './NavSt2DIncompFiniteDiff/' | I implemented, from "soup to nuts", a 2-dimensional Navier-Stokes equations solver for viscous, incompressible fluids with finite difference methods, in CUDA C++11, along with using the thrust library, to run entirely in parallel on the GPU (graphics processing unit), with the GPU global memory access patterns for 2-dim. stencil used in this here github repository, [CUDACFD_out](https://github.com/ernestyalumni/CUDACFD_out)   | |

### Descriptions following each of the output

These descriptions for each of the output are also on the YouTube video descriptions and I reiterate them here, with some editing.

#### [`convect1dupwind`](https://youtu.be/mRJGl0yfiH8)

[convect1dupwind - on CUDA C/C++, 1-dim. convection according to mass conservation](https://github.com/ernestyalumni/CUDACFD_out/tree/master/convect1dupwind)

1-dimensional convection according to mass conservation; initial condition for mass density is a Gaussian distribution.  The velocity vector field is set to be uniformly positive 1.0 m/s and so the Gaussian mass moves to the right.  

OpenGL bitmap graphics were used and is entirely rendered directly on the device GPU, NVIDIA GeForce GTX 980 Ti.  The cudaGraphicsResource class allows for CUDA C/C++ to point to the same bitmap memory address as OpenGL's bitmap pointer (pun intended).  

The 1-dimensional grid on which the so-called “upwind” scheme for finite volume method is computed on, directly on the device GPU, is of size 400, 400 threads being computed in real-time.  The other “y-direction”, of size 400 is used to represent the magnitude of the mass density, so both OpenGL and cudaGraphicsResource is utilizing a 400x400 bitmap, on the GPU.  

This video was used in the main presentation “Finite Difference methods with CUDA C/C++ for GPGPU to solve inviscid (i.e. nonviscous), Compressible Fluid (gas) Flow”, starting at about minute [08:09](https://youtu.be/xQnEQMrol5I?t=8m9s)

#### [`convect3dupwind`](https://youtu.be/s1H1zDkpwTQ)

[convect3dupwind - on CUDA C/C++, 3-dim. convection according to mass conservation](https://github.com/ernestyalumni/CUDACFD_out/tree/master/convect3dupwind)

3-dimensional convection according to mass conservation; initial condition for mass density is a Gaussian distribution.  The velocity vector field is set to be uniformly (25 m/s, 25 m/s, 12 m/s) in the x, y, and z directions, respectively, and so the Gaussian mass moves diagonally up and to the right, and, since we are seeing a 2-dimensional “slice in the middle” of a 3-dimensional grid block, we see the Gaussian mass “shrink” as it moves away from us.  

OpenGL bitmap graphics were used and is entirely rendered directly on the device GPU, NVIDIA GeForce GTX 980 Ti.  The cudaGraphicsResource class allows for CUDA C/C++ to point to the same bitmap memory address as OpenGL's bitmap pointer (pun intended).  

The 3-dimensional grid block on which the so-called “upwind” scheme for finite volume method is computed on, directly on the device GPU, is of dim3 dimensions (1920, 1920, 32), 117964800 threads being computed in real-time.  

This video was used in the main presentation “Finite Difference methods with CUDA C/C++ for GPGPU to solve inviscid (i.e. nonviscous), Compressible Fluid (gas) Flow”, starting at about minute [08:51](https://youtu.be/xQnEQMrol5I?t=8m51s)


#### [`heat2d`](https://youtu.be/oAkOcf8g_TQ)

[heat2d - on CUDA C/C++, 2-dim. heat conduction according to heat equation by finite difference](https://github.com/ernestyalumni/CUDACFD_out/tree/master/heat2d)

2-dimensional heat conduction according to heat equation; a circular heat source (temperature is set to a constant of a high temperature represented by 255 of an arbitrary scale) can be moved around by the mouse, the bottom of a low temperature represented by 0 of an arbitrary scale is the boundary condition at y=0, representing “ground”, the upper, diagonal “sides” are of constant temperature 70 of an arbitrary scale, representing the sides exposed to outside “air”.  

Keep in mind, that the CUDA C++ classes have been factored to allow for the implementation of finite difference in full generality, to any order of accuracy desired: in fact, in this video, the stencil size is of 2, 2 adjacent grid cells in each direction of the point of interest.  

OpenGL texture graphics were used and is entirely rendered directly on the device GPU, NVIDIA GeForce GTX 980 Ti.  The cudaGraphicsResource class allows for CUDA C/C++ to point to the same texture memory address as OpenGL's texture memory pointer (pun intended).  Of note, the implementation of OpenGL texture graphics, inherently a C API, was factored in such a way to be separate from the physics (heat equation) that's being implemented, using C++11 functional library.  CUDA's nvcc, as of CUDA Toolkit 7.5, does not find the device memory where a C++ class that encapsulates OpenGL would be.   

The 2-dimensional grid on which finite difference method is computed on for the heat equation, directly on the device GPU, is of size 640 x 640, 409600 threads being computed in real-time, with threads in blocks of size 32 x 32 threads in a single block.  Of note is that shared memory on this block is used, according to the so-called “tiling” scheme.

This video was used in the main presentation “Finite Difference methods with CUDA C/C++ for GPGPU to solve inviscid (i.e. nonviscous), Compressible Fluid (gas) Flow”, starting at about minute [25:54](https://youtu.be/xQnEQMrol5I?t=25m54s)

#### [`heat3d`](https://youtu.be/LTl9eGs_oIA)

[heat3d - on CUDA C/C++, 3-dim. heat conduction according to heat equation by finite difference](https://github.com/ernestyalumni/CUDACFD_out/tree/master/heat3d)

3-dimensional heat conduction according to heat equation; a spherical heat source (temperature is set to a constant of a high temperature represented by 212 of an arbitrary scale) can be moved around by the mouse, the “left” or (x=0) boundary condition is set at a relatively high temperature of 150 of an arbitrary scale to represent inlet conditions coming in of a high temperature gas, the boundary conditions on the “sides” of this cubic block is set at constant temperature 10 to represent a low temperature for the surroundings of this cubic block.  No boundary conditions was set at x = L_x, the “exit” of this cubic block, representing that we obtain the temperature, in real-time out of this cubic block.  

Keep in mind, that the CUDA C++ classes have been factored to allow for the implementation of finite difference in full generality, to any order of accuracy desired: in fact, in this video, the stencil size is of 2, 2 adjacent grid cells in each direction of the point of interest.  

OpenGL texture graphics were used and is entirely rendered directly on the device GPU, NVIDIA GeForce GTX 980 Ti.  The cudaGraphicsResource class allows for CUDA C/C++ to point to the same texture memory address as OpenGL's texture pointer (pun intended).  Of note, the implementation of OpenGL texture graphics, inherently a C API, was factored in such a way to be separate from the physics (heat equation) that's being implemented, using C++11 functional library.  CUDA's nvcc, as of CUDA Toolkit 7.5, does not find the device memory where a C++ class that encapsulates OpenGL would be.   

The 3-dimensional grid on which finite difference method is computed on for the heat equation, directly on the device GPU, is of size 480 x 480 x 288,  66355200 threads being computed in real-time, with threads in blocks of size 16 x 16 x 4 threads in a single block.  Of note is that shared memory on this block is used, according to the so-called “tiling” scheme.

To emphasize, the heat equation is being computed directly on the device GPU for a 3-dimensional grid in real-time and it also has to render the texture graphics in real-time, and so the animation is slower than in the 2-dimensional case; watch the iteration count at the top bar and see how heat conduction is occurring, as the area where the spherical heat source was grows colder (more blue).

This video was used in the main presentation “Finite Difference methods with CUDA C/C++ for GPGPU to solve inviscid (i.e. nonviscous), Compressible Fluid (gas) Flow”, starting at about minute [29:28](https://youtu.be/xQnEQMrol5I?t=29m28s)

#### [`convect2dfinitedifftex`](https://youtu.be/RGQOIX70jvg)

[convect2dfinitedifftex - on CUDA C/C++, 2-dim. convection by finite difference with OpenGL texture](https://github.com/ernestyalumni/CUDACFD_out/tree/master/convect2dfinitedifftex)

2-dimensional convection according to mass conservation; initial condition for mass density is a Gaussian distribution.  The velocity vector field is set to be uniformly (25.0 m/s, 25.0 m/s) in the x and y-directions, respectively, and so the Gaussian mass moves down and to the right (OpenGL texture graphics (setting the vertex) flips the y direction).  

OpenGL texture graphics were used and is entirely rendered directly on the device GPU, NVIDIA GeForce GTX 980 Ti.  The cudaGraphicsResource class allows for CUDA C/C++ to point to the same texture memory address as OpenGL's texture graphics pointer (pun intended).  

The 2-dimensional grid on which finite difference method is computed on, directly on the device GPU, is of size 680 x 680, 462400 threads being computed in real-time.  For a single block, the blocks were chosen to be of size 2 x 2 threads in a single block (i.e. blockDim).  
Finite difference was computed with a stencil of size 2, i.e. 2 adjacent grid cells in each direction of the point of interest.

This video was used in the main presentation “Finite Difference methods with CUDA C/C++ for GPGPU to solve inviscid (i.e. nonviscous), Compressible Fluid (gas) Flow”, starting at about minute [32:17](https://youtu.be/xQnEQMrol5I?t=32m17s)


#### [`convect3dfinitedifftex`](https://youtu.be/mhXK6xtMz44)

[convect3dfinitedifftex - on CUDA C/C++, 3-dim. convection by finite difference with OpenGL texture](https://github.com/ernestyalumni/CUDACFD_out/tree/master/convect3dfinitedifftex)

3-dimensional convection according to mass conservation; initial condition for mass density is a Gaussian distribution.  The velocity vector field is set to be uniformly (25.0 m/s, 25.0 m/s, 12.0 m/s) in the x, y, and z-directions, respectively, and so the Gaussian mass moves down and to the right (OpenGL texture graphics (setting the vertex) flips the y direction), and “shrinks” because we are observing a 2-dimensional x-y slice “in the middle” of the 3-dimensional grid block, as the Gaussian mass moves away from us.  

OpenGL texture graphics were used and is entirely rendered directly on the device GPU, NVIDIA GeForce GTX 980 Ti.  The cudaGraphicsResource class allows for CUDA C/C++ to point to the same texture memory address as OpenGL's texture pointer (pun intended).  

The 3-dimensional grid on which finite difference method is computed on, directly on the device GPU, is of size 680 x 680 x 320, 14796800 threads being computed in real-time.  For a single block, the blocks were chosen to be of size 2 x 2 x 2 threads in a single block (i.e. blockDim).  

Finite difference was computed with a stencil of size 2, i.e. 2 adjacent grid cells in each direction of the point of interest.

This video was used in the main presentation “Finite Difference methods with CUDA C/C++ for GPGPU to solve inviscid (i.e. nonviscous), Compressible Fluid (gas) Flow”, starting at about minute [32:39](https://youtu.be/xQnEQMrol5I?t=32m39s)

#### [`Euler2d`](https://youtu.be/KLIfUj3V8pY)

##### [Euler2dfinitedifftex - on CUDA C/C++, 2-dim. Euler equations by finite difference with OpenGL texture (my attempt)](https://github.com/ernestyalumni/CUDACFD_out/tree/master/Euler2d)

This is my first attempt at the numerical computation of the 2-dimensional Euler equations; initial condition for mass density is a Gaussian distribution.  The velocity vector field is set initially to be uniformly (25.0 m/s, 25.0, 12.0 m/s) in the x, y and z-directions, respectively, and is dynamically evolved according to the momentum flux.    

OpenGL texture graphics were used and is entirely rendered directly on the device GPU, NVIDIA GeForce GTX 980 Ti.  The cudaGraphicsResource class allows for CUDA C/C++ to point to the same texture memory address as OpenGL's texture pointer (pun intended).  

The 2-dimensional grid on which finite difference method is computed on, directly on the device GPU, is of size 640 x 640, 409600 threads being computed in real-time.  For a single block, the blocks were chosen to be of size 2 x 2 threads in a single block (i.e. blockDim).  Keep in mind that the mass density, 2-dimensional velocity vector field, 2-dimensional momentum flux field, and total energy at each grid point had to be stored in global memory on the device GPU, each represented by float, float2, float2, and float values, respectively.  

Finite difference was computed with a stencil of size 2, i.e. 2 adjacent grid cells in each direction of the point of interest.

This video was used in the main presentation “Finite Difference methods with CUDA C/C++ for GPGPU to solve inviscid (i.e. nonviscous), Compressible Fluid (gas) Flow”, starting at about minute [33:55](https://youtu.be/xQnEQMrol5I?t=33m55s)

##### Conclusion 
While the energy (conservation) equation and the mass conservation equation were easily calculated with CUDA C/C++ for each grid point, the dynamical evolution of the momentum flux and hence the velocity vector field wasn't well-resolved, as can be seen by the eventual “destruction” or nonsensical output after a few iterations.  As far as I know, in previous engineering efforts for CFD (Computational Fluid Dynamics) a uniform velocity vector field was fixed to be time-independent – I was attempting here at a full time-dependent velocity vector field.  Understanding this dynamical evolution from a theory standpoint is also needed in the future.  

##### Remarks on Future Developments for (direct) numerical computation of Euler equations 

Fixing the appropriate initial (other than a Gaussian distribution for the mass density) and boundary conditions, especially conditions directly related to real world conditions, could help to resolve what exactly needs to be calculated.  

What I found was that the so-called “tiling” scheme for shared memory did not compile with nvcc for these Euler equations, even with only mass conservation convection.  This is strange in that this tiling scheme worked and scaled for the heat equation.  Developing a method that uses shared memory for the Euler equations could help.  

Also, it could be just that either the cudaGraphicsResource class pointer or the OpenGL class pointer isn't well “cleared”-out at each iteration or that the appropriate synchronization of all the threads on the GPU is done.   I've done my best to add in the appropriate functions such as __syncthreads(); to synchronize all the threads before each time step – again, my code is on github:  https://github.com/ernestyalumni/CUDACFD_out/tree/master/Euler2d 

Finite difference methods had been shown to be stable, convergent, and produce sensible output for the heat equation; other methods, finite volume and finite element, methods, computed directly on the GPU could possibly be explored for the Euler equations.  
