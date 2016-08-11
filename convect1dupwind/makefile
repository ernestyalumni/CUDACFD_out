#  Ernest Yeung  ernestyalumni@gmail.com
#  20160704
# GFX stands for "graphics"; OpenGL flags
GFXFLAGS += -lglut -lGL

NVCC = nvcc
NVCC_FLAGS = -std=c++11

all: convect1dupwind

convect1dupwind : convect1dupwind.cu
	$(NVCC) $(NVCC_FLAGS) $(GFXFLAGS) $< -o $@

clean:
	rm -f *.o convect1dupwind
