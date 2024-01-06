# CUDA Compiler
NVCC := nvcc

# Compiler flags
CXXFLAGS := -std=c++11

# CUDA flags and libraries
NVCCFLAGS := -arch=sm_70
LIBS := -lcudart -lcusolver -lcublas -lcusparse

# Source file and executable name
SRC := lab4-ex4-template.cu
EXE := lab4-ex4-template

all: $(EXE)

$(EXE): $(SRC)
	$(NVCC) $(CXXFLAGS) $(NVCCFLAGS) $^ -o $@ $(LIBS)

clean:
	rm -f $(EXE)

