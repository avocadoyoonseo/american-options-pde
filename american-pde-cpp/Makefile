CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -O3 -I./include
DEBUG_FLAGS = -g -DDEBUG

# Directories
SRC_DIR = src
INCLUDE_DIR = include
TEST_DIR = tests
OBJ_DIR = obj
BIN_DIR = bin

# Create directories if they don't exist
$(shell mkdir -p $(OBJ_DIR) $(BIN_DIR))

# Source files
CN_PSOR_SRC = $(SRC_DIR)/main_cn_psor.cpp
PENALTY_SRC = $(SRC_DIR)/main_penalty.cpp
AMR_SRC = $(SRC_DIR)/main_amr.cpp

# Test source files
TEST_TRIDIAG_SRC = $(TEST_DIR)/test_tridiag.cpp
TEST_PSOR_SRC = $(TEST_DIR)/test_psor.cpp

# Object files
CN_PSOR_OBJ = $(OBJ_DIR)/main_cn_psor.o
PENALTY_OBJ = $(OBJ_DIR)/main_penalty.o
AMR_OBJ = $(OBJ_DIR)/main_amr.o
TEST_TRIDIAG_OBJ = $(OBJ_DIR)/test_tridiag.o
TEST_PSOR_OBJ = $(OBJ_DIR)/test_psor.o

# Executables
CN_PSOR_BIN = $(BIN_DIR)/cn_psor
PENALTY_BIN = $(BIN_DIR)/penalty
AMR_BIN = $(BIN_DIR)/amr
TEST_TRIDIAG_BIN = $(BIN_DIR)/test_tridiag
TEST_PSOR_BIN = $(BIN_DIR)/test_psor

# All executables
EXECS = $(CN_PSOR_BIN) $(PENALTY_BIN) $(AMR_BIN)
TESTS = $(TEST_TRIDIAG_BIN) $(TEST_PSOR_BIN)

# Default target
all: $(EXECS)

# Test target
test: $(TESTS)
	@echo "Running tests..."
	@./$(TEST_TRIDIAG_BIN)
	@./$(TEST_PSOR_BIN)

# Compile main executables
$(CN_PSOR_BIN): $(CN_PSOR_OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $^

$(PENALTY_BIN): $(PENALTY_OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $^

$(AMR_BIN): $(AMR_OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $^

# Compile test executables
$(TEST_TRIDIAG_BIN): $(TEST_TRIDIAG_OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $^

$(TEST_PSOR_BIN): $(TEST_PSOR_OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $^

# Compile object files
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $<

$(OBJ_DIR)/%.o: $(TEST_DIR)/%.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $<

# Debug versions
debug: CXXFLAGS += $(DEBUG_FLAGS)
debug: all

# Clean
clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR)

.PHONY: all test clean debug
