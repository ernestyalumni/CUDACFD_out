#  Ernest Yeung  ernestyalumni@gmail.com
#  20161029
# 
LDFLAGS += -L/usr/local/cuda/samples/common/lib/linux/x86_64

NVCC = /usr/local/cuda/bin/nvcc
NVCC_FLAGS = -g -G -Xcompiler "-Wall -Wno-deprecated-declarations"
INC = -I/usr/local/cuda/samples/common/inc

physobjects = dev_R1grid.o R1grid.o 
compobjects = MacCor1d.o kernelLaunch.o

all: main.exe

main.exe: main.o $(physobjects) $(compobjects)
	$(NVCC) -std=c++11 $^ -o $@ $(LDFLAGS)
	
main.o: main.cu
	$(NVCC) -std=c++11 $(NVCC_FLAGS) $(LDFLAGS) -dc $< -o $@
	
# physics

dev_R1grid.o: ./physlib/dev_R1grid.cu
	$(NVCC) -std=c++11 $(NVCC_FLAGS) -dc $< -o $@

R1grid.o: ./physlib/R1grid.cpp
	$(NVCC) -std=c++11 -x cu $(NVCC_FLAGS) -dc $< -o $@
	
# computational

MacCor1d.o: ./commonlib/MacCor1d.cu
	$(NVCC) -std=c++11 $(NVCC_FLAGS) -dc $< -o $@

kernelLaunch.o: ./commonlib/kernelLaunch.cu 
	$(NVCC) -std=c++11 $(NVCC_FLAGS) -dc $< -o $@
	
	
clean:
	rm -f *.o *.exe
