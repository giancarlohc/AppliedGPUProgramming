# CUDA Compiler
NVCC := nvcc

# Compiler flags
CXXFLAGS := -std=c++11

# CUDA flags and libraries
NVCCFLAGS := -arch=sm_70
LIBS := -lcudart

# Source file and executable name
SRC := lab3_ex3.cu
EXE := lab3_ex3

all: $(EXE)

$(EXE): $(SRC)
	$(NVCC) $(CXXFLAGS) $(NVCCFLAGS) $^ -o $@ $(LIBS)

clean:
	rm -f $(EXE)

