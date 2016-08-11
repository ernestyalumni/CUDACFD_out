#  Ernest Yeung  ernestyalumni@gmail.com
#  20160704
# GFX stands for "graphics"; OpenGL flags
GFXFLAGS += -lglut -lGL

NVCC = nvcc
NVCC_FLAGS = -std=c++11 -dc

objects = convect3dupwind.o R3grid.o dev_R3grid.o

convect3dupwind: $(objects)
	$(NVCC) $(GFXFLAGS) $(objects) -o convect3dupwind
	
	
convect3dupwind.o : convect3dupwind.cu
	$(NVCC) $(NVCC_FLAGS) $(GFXFLAGS) $< -o $@

R3grid.o: ./physlib/R3grid.cpp
	$(NVCC) $(NVCC_FLAGS) $< -o $@


dev_R3grid.o: ./physlib/dev_R3grid.cu
	nvcc $(NVCC_FLAGS) $< -o $@

clean:
	rm -f *.o convect3dupwind
