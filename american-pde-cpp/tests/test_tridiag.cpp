#include "../include/tridiag.hpp"
#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>

// Simple test framework
#define TEST(name) void name()
#define ASSERT_NEAR(a, b, tol) { \
    if (std::abs((a) - (b)) > (tol)) { \
        std::cerr << "FAILED: " << #a << " (" << (a) << ") != " << #b << " (" << (b) << ") within tolerance " << (tol) << std::endl; \
        assert(false); \
    } \
}
#define RUN_TEST(name) std::cout << "Running " << #name << "..." << std::endl; name(); std::cout << "PASSED" << std::endl

// Test the tridiagonal solver with a simple system
TEST(test_tridiag_simple) {
    // Simple 3x3 system:
    // [2 -1  0] [x1]   [1]
    // [-1 2 -1] [x2] = [2]
    // [0 -1  2] [x3]   [3]
    // Expected solution: [3/4, 3/2, 9/4]
    
    std::vector<double> a = {-1.0, -1.0};  // Lower diagonal
    std::vector<double> b = {2.0, 2.0, 2.0};  // Main diagonal
    std::vector<double> c = {-1.0, -1.0};  // Upper diagonal
    std::vector<double> d = {1.0, 2.0, 3.0};  // RHS
    
    std::vector<double> expected = {0.75, 1.5, 2.25};
    
    std::vector<double> result = fin::TridiagonalSolver::solve(a, b, c, d);
    
    ASSERT_NEAR(result[0], expected[0], 1e-10);
    ASSERT_NEAR(result[1], expected[1], 1e-10);
    ASSERT_NEAR(result[2], expected[2], 1e-10);
}

// Test the tridiagonal solver with a larger system
TEST(test_tridiag_larger) {
    int n = 100;
    
    // Create a system representing the 1D Poisson equation:
    // -u''(x) = f(x) with u(0)=u(1)=0
    // Discretized with central difference
    
    std::vector<double> a(n-1, -1.0);  // Lower diagonal
    std::vector<double> b(n, 2.0);     // Main diagonal
    std::vector<double> c(n-1, -1.0);  // Upper diagonal
    
    // Set boundary conditions
    b[0] = 1.0;
    b[n-1] = 1.0;
    
    // Create RHS representing f(x) = sin(pi*x)
    std::vector<double> d(n, 0.0);
    double h = 1.0 / (n - 1);
    for (int i = 1; i < n-1; i++) {
        double x = i * h;
        d[i] = h*h * std::sin(M_PI * x);
    }
    
    // Solve system
    std::vector<double> result = fin::TridiagonalSolver::solve(a, b, c, d);
    
    // Check against analytical solution u(x) = sin(pi*x) / (pi*pi)
    for (int i = 0; i < n; i++) {
        double x = i * h;
        double expected = std::sin(M_PI * x) / (M_PI * M_PI);
        ASSERT_NEAR(result[i], expected, 1e-4);  // Lower precision due to discretization error
    }
}

// Test the solver with reuse
TEST(test_tridiag_reuse) {
    std::vector<double> a = {-1.0, -1.0};  // Lower diagonal
    std::vector<double> b = {2.0, 2.0, 2.0};  // Main diagonal
    std::vector<double> c = {-1.0, -1.0};  // Upper diagonal
    
    fin::TridiagonalSolver solver(a, b, c);
    
    // First system
    std::vector<double> d1 = {1.0, 2.0, 3.0};
    std::vector<double> expected1 = {0.75, 1.5, 2.25};
    std::vector<double> result1 = solver.solve(d1);
    
    ASSERT_NEAR(result1[0], expected1[0], 1e-10);
    ASSERT_NEAR(result1[1], expected1[1], 1e-10);
    ASSERT_NEAR(result1[2], expected1[2], 1e-10);
    
    // Second system with same coefficients but different RHS
    std::vector<double> d2 = {3.0, 2.0, 1.0};
    std::vector<double> expected2 = {2.25, 1.5, 0.75};
    std::vector<double> result2 = solver.solve(d2);
    
    ASSERT_NEAR(result2[0], expected2[0], 1e-10);
    ASSERT_NEAR(result2[1], expected2[1], 1e-10);
    ASSERT_NEAR(result2[2], expected2[2], 1e-10);
    
    // Update coefficients and solve again
    std::vector<double> a2 = {-2.0, -2.0};
    std::vector<double> b2 = {4.0, 4.0, 4.0};
    std::vector<double> c2 = {-2.0, -2.0};
    
    solver.updateDiagonals(a2, b2, c2);
    std::vector<double> d3 = {4.0, 8.0, 12.0};
    std::vector<double> expected3 = {1.5, 3.0, 4.5};  // Same solution as first case
    std::vector<double> result3 = solver.solve(d3);
    
    ASSERT_NEAR(result3[0], expected3[0], 1e-10);
    ASSERT_NEAR(result3[1], expected3[1], 1e-10);
    ASSERT_NEAR(result3[2], expected3[2], 1e-10);
}

int main() {
    std::cout << "Running tridiagonal solver tests..." << std::endl;
    
    RUN_TEST(test_tridiag_simple);
    RUN_TEST(test_tridiag_larger);
    RUN_TEST(test_tridiag_reuse);
    
    std::cout << "All tridiagonal solver tests passed!" << std::endl;
    return 0;
}
