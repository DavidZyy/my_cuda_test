NAME  = test_cuda_gemm
SRC_DIR  =

SRC_FILE = Sgemm.cu

BUILD = build/
OBJ = $(BUILD)$(SRC_DIR)$(NAME)

CXX = nvcc

CXXFLAGS = -lcublas -ccbin g++ -std=c++17


run:
	mkdir -p $(BUILD)$(SRC_DIR)
	$(CXX) $(CXXFLAGS) -o $(OBJ) $(SRC_FILE)
	$(OBJ)

clean:
	rm -rf $(BUILD)