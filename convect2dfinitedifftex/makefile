#  Ernest Yeung  ernestyalumni@gmail.com
#  20160704
# GFX stands for "graphics"; OpenGL flags
GFXFLAGS += -lglut -lGL -lGLU

NVCC = nvcc
NVCC_FLAGS = -std=c++11 -dc

objects = main.o finitediff.o tex_anim2d.o convect.o R2grid.o dev_R2grid.o 


all: main.exe

main.exe: $(objects)
	$(NVCC) $(GFXFLAGS) $(objects) -o $@
	
	
main.o : main.cu
	$(NVCC) $(NVCC_FLAGS) $(GFXFLAGS) $< -o $@


finitediff.o : ./commonlib/finitediff.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@


tex_anim2d.o : ./commonlib/tex_anim2d.cu
	$(NVCC) $(NVCC_FLAGS) $(GFXFLAGS) $< -o $@

# physics

convect.o : ./physlib/convect.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

R2grid.o: ./physlib/R2grid.cpp
	$(NVCC) $(NVCC_FLAGS) -x cu $< -o $@

dev_R2grid.o: ./physlib/dev_R2grid.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@




clean:
	rm -f *.o main.exe
