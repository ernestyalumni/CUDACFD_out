LDFLAGS += -L/usr/local/cuda/samples/common/lib/linux/x86_64
LDFLAGS += -lglut -lGL -lGLU

NVCC = /usr/local/cuda/bin/nvcc
NVCC_FLAGS = -g -G -Xcompiler "-Wall -Wno-deprecated-declarations"
INC = -I/usr/local/cuda/samples/common/inc

gfxobjects  = tex_anim2d.o
mathobjects = finitediff.o sharedmem.o
physobjects = dev_phys_param.o dev_R2grid.o R2grid.o convect.o Euler2d.o

all: main.exe

main.exe: main.o $(gfxobjects) $(mathobjects) $(physobjects)
	$(NVCC) -std=c++11 $^ -o $@ $(LDFLAGS)
		
main.o: main.cu 
	$(NVCC) -std=c++11 $(NVCC_FLAGS) $(INC) -dc $< -o $@

# graphics (gfx)
tex_anim2d.o: ./commonlib/tex_anim2d.cu 
		$(NVCC) -std=c++11 $(NVCC_FLAGS) $(LDFLAGS) -dc $< -o $@

# math
sharedmem.o: ./commonlib/sharedmem.cu
	$(NVCC) -std=c++11 $(NVCC_FLAGS) -dc $< -o $@
	
finitediff.o: ./commonlib/finitediff.cu
	$(NVCC) -std=c++11 $(NVCC_FLAGS) -dc $< -o $@
	


# physics
dev_phys_param.o: ./physlib/dev_phys_param.cu
	$(NVCC) -std=c++11 $(NVCC_FLAGS) -dc $< -o $@

dev_R2grid.o: ./physlib/dev_R2grid.cu
	$(NVCC) -std=c++11 $(NVCC_FLAGS) -dc $< -o $@

R2grid.o: ./physlib/R2grid.cu
	$(NVCC) -std=c++11 $(NVCC_FLAGS) -dc $< -o $@

convect.o: ./physlib/convect.cu
	$(NVCC) -std=c++11 $(NVCC_FLAGS) -dc $< -o $@
	
Euler2d.o: ./physlib/Euler2d.cu
	$(NVCC) -std=c++11 $(NVCC_FLAGS) -dc $< -o $@

		
clean:
		rm -f *.o *.exe
