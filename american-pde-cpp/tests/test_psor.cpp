#include "../include/psor.hpp"
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

// Test PSOR solver with a simple unconstrained system (should behave like regular SOR)
TEST(test_psor_unconstrained) {
    // Simple 3x3 system:
    // [2 -1  0] [x1]   [1]
    // [-1 2 -1] [x2] = [2]
    // [0 -1  2] [x3]   [3]
    // Expected solution: [3/4, 3/2, 9/4]
    
    std::vector<double> a = {-1.0, -1.0};  // Lower diagonal
    std::vector<double> b = {2.0, 2.0, 2.0};  // Main diagonal
    std::vector<double> c = {-1.0, -1.0};  // Upper diagonal
    std::vector<double> rhs = {1.0, 2.0, 3.0};  // RHS
    
    // Set constraints far below the solution to make the problem effectively unconstrained
    std::vector<double> constraints = {-100.0, -100.0, -100.0};
    
    // Create PSOR solver with relaxation parameter 1.0 (Gauss-Seidel)
    fin::PSORSolver solver(a, b, c, 1.0, 1e-10, 1000);
    
    // Solve system
    std::vector<double> result = solver.solve(rhs, constraints);
    
    // Expected solution
    std::vector<double> expected = {0.75, 1.5, 2.25};
    
    // Check result
    ASSERT_NEAR(result[0], expected[0], 1e-6);
    ASSERT_NEAR(result[1], expected[1], 1e-6);
    ASSERT_NEAR(result[2], expected[2], 1e-6);
}

// Test PSOR solver with constraints that affect the solution
TEST(test_psor_constrained) {
    // Simple 3x3 system:
    // [2 -1  0] [x1]   [1]
    // [-1 2 -1] [x2] = [2]
    // [0 -1  2] [x3]   [3]
    // With constraints x1 >= 1, x2 >= 2, x3 >= 0
    // Expected solution: [1, 2, 2.5]
    
    std::vector<double> a = {-1.0, -1.0};  // Lower diagonal
    std::vector<double> b = {2.0, 2.0, 2.0};  // Main diagonal
    std::vector<double> c = {-1.0, -1.0};  // Upper diagonal
    std::vector<double> rhs = {1.0, 2.0, 3.0};  // RHS
    
    // Set constraints
    std::vector<double> constraints = {1.0, 2.0, 0.0};
    
    // Create PSOR solver with relaxation parameter 1.5
    fin::PSORSolver solver(a, b, c, 1.5, 1e-10, 1000);
    
    // Solve system
    std::vector<double> result = solver.solve(rhs, constraints);
    
    // Without constraints, solution would be [0.75, 1.5, 2.25]
    // With constraints x1 >= 1, x2 >= 2, we expect [1, 2, 2.5]
    
    // Check that constraints are satisfied
    ASSERT_NEAR(result[0], 1.0, 1e-6);  // Constrained to x1 >= 1
    ASSERT_NEAR(result[1], 2.0, 1e-6);  // Constrained to x2 >= 2
    ASSERT_NEAR(result[2], 2.5, 1e-6);  // Updated due to constraints on x1 and x2
}

// Test convergence with different relaxation parameters
TEST(test_psor_relaxation) {
    // Create a larger system for better testing of convergence properties
    int n = 50;
    std::vector<double> a(n-1, -1.0);  // Lower diagonal
    std::vector<double> b(n, 2.0);     // Main diagonal
    std::vector<double> c(n-1, -1.0);  // Upper diagonal
    
    // Create RHS
    std::vector<double> rhs(n, 0.0);
    for (int i = 0; i < n; ++i) {
        rhs[i] = std::sin(i * M_PI / (n-1));
    }
    
    // Set constraints far below solution
    std::vector<double> constraints(n, -100.0);
    
    // Try different relaxation parameters and check iterations
    std::vector<double> omegas = {0.5, 1.0, 1.5, 1.9};
    std::vector<int> iterations;
    
    for (double omega : omegas) {
        fin::PSORSolver solver(a, b, c, omega, 1e-10, 1000);
        std::vector<double> result = solver.solve(rhs, constraints);
        iterations.push_back(solver.getLastIterationCount());
    }
    
    // Verify that moderate over-relaxation (omega around 1.5-1.9) converges faster
    // than under-relaxation (omega < 1) or no relaxation (omega = 1)
    std::cout << "PSOR iterations with different omega values:" << std::endl;
    for (size_t i = 0; i < omegas.size(); ++i) {
        std::cout << "  omega = " << omegas[i] << ": " << iterations[i] << " iterations" << std::endl;
    }
    
    // We expect omega=1.5 or omega=1.9 to require fewer iterations than omega=0.5 or omega=1.0
    assert(iterations[2] < iterations[0]);  // omega=1.5 better than omega=0.5
    assert(iterations[3] < iterations[1]);  // omega=1.9 better than omega=1.0
}

// Test PSOR with an initial guess
TEST(test_psor_initial_guess) {
    // Simple 3x3 system
    std::vector<double> a = {-1.0, -1.0};  // Lower diagonal
    std::vector<double> b = {2.0, 2.0, 2.0};  // Main diagonal
    std::vector<double> c = {-1.0, -1.0};  // Upper diagonal
    std::vector<double> rhs = {1.0, 2.0, 3.0};  // RHS
    std::vector<double> constraints = {0.0, 0.0, 0.0};  // Non-negative constraints
    
    // Create PSOR solver
    fin::PSORSolver solver(a, b, c, 1.5, 1e-10, 1000);
    
    // Solve system without initial guess
    std::vector<double> result1 = solver.solve(rhs, constraints);
    int iter1 = solver.getLastIterationCount();
    
    // Solve system with good initial guess (close to solution)
    std::vector<double> initialGuess = {0.7, 1.4, 2.2};  // Close to [0.75, 1.5, 2.25]
    std::vector<double> result2 = solver.solve(rhs, constraints, &initialGuess);
    int iter2 = solver.getLastIterationCount();
    
    // Check solutions match
    for (size_t i = 0; i < result1.size(); ++i) {
        ASSERT_NEAR(result1[i], result2[i], 1e-6);
    }
    
    // Good initial guess should converge in fewer iterations
    std::cout << "PSOR iterations: without initial guess = " << iter1 
              << ", with good initial guess = " << iter2 << std::endl;
    assert(iter2 <= iter1);
}

int main() {
    std::cout << "Running PSOR solver tests..." << std::endl;
    
    RUN_TEST(test_psor_unconstrained);
    RUN_TEST(test_psor_constrained);
    RUN_TEST(test_psor_relaxation);
    RUN_TEST(test_psor_initial_guess);
    
    std::cout << "All PSOR solver tests passed!" << std::endl;
    return 0;
}
