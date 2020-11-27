CXX=nvcc
CXXFLAGS=-arch=sm_70 -rdc=true

# XXX: If you have difficulties compiling on Cori,
#      make sure you've sourced the modules.sh file
#      with `source modules.sh`.

all: segscan

multiblock_scan: segscan.cu serial.hpp util.hpp
	$(CXX) segscan.cu -o segscan -std=c++14 -O3 $(CXXFLAGS)

clean:
	@rm -fv segscan
