#!/bin/bash

# Create required directories
mkdir -p bin data obj

# Determine compiler (prefer clang++ on macOS)
if command -v clang++ &> /dev/null; then
    CXX=clang++
else
    CXX=g++
fi

# Set compiler flags
CXXFLAGS="-std=c++17 -O3 -Wall -Wextra -I./include"
DEBUG_FLAGS="-g -DDEBUG"

# Parse command line arguments
DEBUG=0
CLEAN=0
TEST=0
TARGET="all"

for arg in "$@"; do
    if [ "$arg" == "debug" ]; then
        DEBUG=1
    elif [ "$arg" == "clean" ]; then
        CLEAN=1
    elif [ "$arg" == "test" ]; then
        TEST=1
    else
        TARGET="$arg"
    fi
done

# Clean if requested
if [ $CLEAN -eq 1 ]; then
    echo "Cleaning build files..."
    rm -rf obj/* bin/*
    if [ "$TARGET" == "clean" ]; then
        exit 0
    fi
fi

# Add debug flags if requested
if [ $DEBUG -eq 1 ]; then
    CXXFLAGS="$CXXFLAGS $DEBUG_FLAGS"
    echo "Building in debug mode..."
fi

# Build test executables if requested
if [ $TEST -eq 1 ]; then
    echo "Building tests..."
    
    echo "Compiling test_tridiag..."
    $CXX $CXXFLAGS tests/test_tridiag.cpp -o bin/test_tridiag
    
    echo "Compiling test_psor..."
    $CXX $CXXFLAGS tests/test_psor.cpp -o bin/test_psor
    
    # Run tests if build succeeded
    if [ $? -eq 0 ]; then
        echo "Running tests..."
        bin/test_tridiag
        bin/test_psor
    fi
    
    exit $?
fi

# Build main executables
if [ "$TARGET" == "all" ] || [ "$TARGET" == "cn_psor" ]; then
    echo "Compiling main_cn_psor..."
    $CXX $CXXFLAGS src/main_cn_psor.cpp -o bin/cn_psor
fi

if [ "$TARGET" == "all" ] || [ "$TARGET" == "penalty" ]; then
    echo "Compiling main_penalty..."
    $CXX $CXXFLAGS src/main_penalty.cpp -o bin/penalty
fi

if [ "$TARGET" == "all" ] || [ "$TARGET" == "amr" ]; then
    echo "Compiling main_amr..."
    $CXX $CXXFLAGS src/main_amr.cpp -o bin/amr
fi

if [ $? -eq 0 ]; then
    echo "Build successful!"
    echo "Run the executables with: ./bin/[executable_name] [options]"
    echo "Examples:"
    echo "  ./bin/cn_psor --S0 100 --K 100 --r 0.05 --q 0.02 --sigma 0.2 --T 1.0"
    echo "  ./bin/penalty --S0 100 --K 100 --r 0.05 --q 0.02 --sigma 0.2 --T 1.0 --penalty 1e6"
    echo "  ./bin/amr --S0 100 --K 100 --r 0.05 --q 0.02 --sigma 0.2 --T 1.0 --refine 3"
fi
