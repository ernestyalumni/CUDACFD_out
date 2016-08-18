# CUDACFD_out
CUDA C/C++ scripts for Computational Fluid Dynamics (CFD) for presentation purposes (that goes out)

- C++14 standard on host CPU code/C++11 standard on device GPU as of CUDA Toolkit 7.5 

### (Abridged) Table of Contents

| filename (or directory) | directory | Description | related YouTube link (if there is one) |
| :---------------------- | :-------: | :---------: | -------------------------------------: |
| `CUDACFD_writeup01.tex` | `./LaTeXandpdfs/` | LaTeX file for the writeup explaining in depth the theory, concepts, and implementation of finite difference methods on CUDA C/C++ | [presentation the writeup accompanies](https://youtu.be/xQnEQMrol5I)  |
| `CUDACFD_writeup01.pdf` | `./LaTeXandpdfs/` | Compiled pdf file for writeup | [presentation the writeup accompanies](https://youtu.be/xQnEQMrol5I) |
| `CUDACFD_slides01.tex`  | `./LaTeXandpdfs/` | LaTeX file for slide deck for the main YouTube presentation | [presentation that uses the slide deck](https://youtu.be/xQnEQMrol5I) |
| `CUDACFD_slides01.pdf`  | `./LaTeXandpdfs/` | Compiled pdf file for slide deck for the main YouTube presentation | [presentation that uses the slide deck](https://youtu.be/xQnEQMrol5I) |
| `/convect1dupwind`   | `./convect1dupwind`  | 1-dimensional convection according to mass conservation; initial condition for mass density is a gaussian distribution | [screen capture of a convect1dupwind run](https://youtu.be/mRJGl0yfiH8) |
| `/convect3dupwind`   | `./convect3dupwind`  | 3-dimensional convection according to mass conservation; initial condition for mass density is a gaussian distribution | [screen capture of a convect3dupwind run](https://youtu.be/s1H1zDkpwTQ) |
| `/heat2d`            | `./heat2d`           | 2-dimensional heat conduction according to the heat equation; CUDA C++ classes has been entirely factored to allow for finite difference method to any order of accuracy | [screen capture of heat2d run](https://youtu.be/oAkOcf8g_TQ)  |
| `/heat3d`            | `./heat3d`           | 3-dimensional heat conduction according to the heat equation; CUDA C++ classes has been entirely factored to allow for finite difference method to any order of accuracy | [screen capture of heat3d run](https://youtu.be/LTl9eGs_oIA)  |
