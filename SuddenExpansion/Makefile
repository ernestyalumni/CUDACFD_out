# Ernest Yeung
# ernestyalumni@gmail.com
# 20161116
NVCC = nvcc
NVCC_FLAGS = -std=c++11

GPP = g++
GPP_FLAGS = -std=c++11

GFX_FLAGS = -lglut -lGL -lGLU 

physobjects    = dev_R2grid.o R2grid.o 
commonobjects  = surfObj2d.o
gfxobjects     = tex_anim2d.o

all: main.exe

main.exe: main.o $(physobjects) $(commonobjects) $(gfxobjects)
	$(NVCC) $(NVCC_FLAGS) $^ -o $@ $(GFX_FLAGS)
		
main.o: main.cu 
	$(NVCC) $(NVCC_FLAGS) -dc $< -o $@

# commonlib
surfObj2d.o: ./commonlib/surfObj2d.cu
	$(NVCC) $(NVCC_FLAGS) -dc $< -o $@
	

# physics
dev_R2grid.o: ./physlib/dev_R2grid.cu
	$(NVCC) $(NVCC_FLAGS) -dc $< -o $@

R2grid.o: ./physlib/R2grid.cpp
#	$(GPP) $(GPP_FLAGS) -c $< -o $@  # I need float2 and so I had to use the nvcc compiler
	$(NVCC) -x cu $(NVCC_FLAGS) -dc $< -o $@  

# graphics (gfx)
tex_anim2d.o: ./commonlib/tex_anim2d.cu
	$(NVCC) $(NVCC_FLAGS) $(GFX_FLAGS) -dc $< -o $@
		
clean:
		rm -f *.o *.exe
