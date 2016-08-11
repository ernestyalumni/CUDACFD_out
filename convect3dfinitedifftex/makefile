#  Ernest Yeung  ernestyalumni@gmail.com
#  20160704
# GFX stands for "graphics"; OpenGL flags
GFXFLAGS += -lglut -lGL -lGLU

NVCC = nvcc
NVCC_FLAGS = -std=c++11 -dc

objects = main.o finitediff.o tex_anim2d.o convect.o R3grid.o dev_R3grid.o sharedmem.o


all: main.exe

main.exe: $(objects)
	$(NVCC) $(GFXFLAGS) $(objects) -o $@
	
	
main.o : main.cu
	$(NVCC) $(NVCC_FLAGS) $(GFXFLAGS) $< -o $@

sharedmem.o : ./commonlib/sharedmem.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

finitediff.o : ./commonlib/finitediff.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@


tex_anim2d.o : ./commonlib/tex_anim2d.cu
	$(NVCC) $(NVCC_FLAGS) $(GFXFLAGS) $< -o $@

# physics

convect.o : ./physlib/convect.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

R3grid.o: ./physlib/R3grid.cpp
	$(NVCC) $(NVCC_FLAGS) -x cu $< -o $@

dev_R3grid.o: ./physlib/dev_R3grid.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@




clean:
	rm -f *.o main.exe
