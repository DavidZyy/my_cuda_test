NAME  = test_cuda_gemm
# SRC_DIR  = src/parallel/cuda
SRC_DIR  =

# SRC_FILE = $(SRC_DIR)$(NAME).cpp
# SRC_FILE = $(wildcard $(SRC_DIR)/*) 
SRC_FILE = main.cpp gemm.cu gemm.cpp

BUILD = build/
OBJ = $(BUILD)$(SRC_DIR)$(NAME)

# CXX = g++
CXX = nvcc

# CXXFLAGS = -std=c++98 -Wall -Wextra -g
# CXXFLAGS = -Wall -Wextra -g -std=c++20
# CXXFLAGS = -lcublas -ccbin clang-14
CXXFLAGS = -lcublas -lopenblas -ccbin g++


# $(error $(OBJ))
# $(error $(SRC_FILE))

run:
# @mkdir -p $(BUILD)$(SRC_DIR)
# @$(CXX) $(CXXFLAGS) -o $(OBJ) $(SRC_FILE)
# @$(OBJ)
	mkdir -p $(BUILD)$(SRC_DIR)
	$(CXX) $(CXXFLAGS) -o $(OBJ) $(SRC_FILE)
	$(OBJ)

clean:
	rm -rf $(BUILD)