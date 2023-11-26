# CUDA Compiler
NVCC := nvcc

# Compiler flags
CXXFLAGS := -std=c++11

# CUDA flags and libraries
NVCCFLAGS := -arch=sm_70
LIBS := -lcudart

# Source file and executable name
SRC := vector_allocation.cu
EXE := vector_allocation

all: $(EXE)

$(EXE): $(SRC)
	$(NVCC) $(CXXFLAGS) $(NVCCFLAGS) $^ -o $@ $(LIBS)

clean:
	rm -f $(EXE)

