#include "../include/grid.hpp"
#include "../include/cn.hpp"
#include "../include/utils.hpp"
#include <iostream>
#include <iomanip>
#include <vector>

int main() {
    // Test parameters
    double S0 = 100.0, K = 100.0, r = 0.05, q = 0.02, sigma = 0.2, T = 1.0;
    int nx = 50;  // Small grid for debugging
    int nt = 50;
    bool isCall = false; // Put option
    
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "=== PDE Solver Debug ===" << std::endl;
    std::cout << "Parameters: S0=" << S0 << ", K=" << K << ", r=" << r 
              << ", q=" << q << ", sigma=" << sigma << ", T=" << T << std::endl;
    std::cout << "Grid: nx=" << nx << ", nt=" << nt << std::endl << std::endl;
    
    // Create grid
    fin::Grid grid = fin::Grid::createOptionGrid(S0, K, sigma, T, nx);
    std::vector<double> timePoints = fin::Grid::createTimeGrid(T, nt);
    
    // Print grid info
    std::cout << "Grid boundaries:" << std::endl;
    std::cout << "  S_min = " << grid.point(0) << std::endl;
    std::cout << "  S_max = " << grid.point(nx-1) << std::endl;
    std::cout << "  Grid size = " << grid.size() << std::endl;
    
    // Find where S0 is in the grid
    int idx_S0 = -1;
    for (size_t i = 0; i < grid.size(); ++i) {
        if (grid.point(i) >= S0) {
            idx_S0 = i;
            break;
        }
    }
    std::cout << "  S0=" << S0 << " is between grid points " << (idx_S0-1) << " and " << idx_S0 << std::endl;
    if (idx_S0 > 0) {
        std::cout << "    S[" << (idx_S0-1) << "] = " << grid.point(idx_S0-1) << std::endl;
        std::cout << "    S[" << idx_S0 << "] = " << grid.point(idx_S0) << std::endl;
    }
    std::cout << std::endl;
    
    // Print first few and last few grid points
    std::cout << "First 10 grid points:" << std::endl;
    for (int i = 0; i < std::min(10, nx); ++i) {
        std::cout << "  S[" << i << "] = " << grid.point(i) << std::endl;
    }
    std::cout << "Last 10 grid points:" << std::endl;
    for (int i = std::max(0, nx-10); i < nx; ++i) {
        std::cout << "  S[" << i << "] = " << grid.point(i) << std::endl;
    }
    std::cout << std::endl;
    
    // Create solver
    fin::CrankNicolson solver(grid, timePoints, r, q, sigma, isCall);
    
    // Check initial condition (payoff at maturity)
    std::cout << "Initial condition (payoff at T):" << std::endl;
    std::cout << "  At S=" << grid.point(idx_S0-1) << ": payoff = " << std::max(0.0, K - grid.point(idx_S0-1)) << std::endl;
    std::cout << "  At S=" << grid.point(idx_S0) << ": payoff = " << std::max(0.0, K - grid.point(idx_S0)) << std::endl;
    std::cout << std::endl;
    
    // Solve
    std::vector<double> solution = solver.solve(K);
    
    // Print solution at key points
    std::cout << "Solution at key points:" << std::endl;
    std::cout << "  V(S=0) = " << solution[0] << " (expected: K*exp(-r*T) = " 
              << K*std::exp(-r*T) << ")" << std::endl;
    std::cout << "  V(S=S_max) = " << solution[nx-1] << " (expected: 0)" << std::endl;
    
    // Interpolate to S0
    double price_at_S0 = solver.priceAt(S0);
    std::cout << "  V(S=" << S0 << ") = " << price_at_S0 << std::endl;
    
    // Compare with Black-Scholes
    double bs_price = fin::BlackScholes::price(S0, K, r, q, sigma, T, isCall);
    std::cout << std::endl;
    std::cout << "Black-Scholes price: " << bs_price << std::endl;
    std::cout << "PDE solver price: " << price_at_S0 << std::endl;
    std::cout << "Error: " << (price_at_S0 - bs_price) << " (" 
              << std::abs(price_at_S0 - bs_price) / bs_price * 100.0 << "%)" << std::endl;
    
    // Print solution around S0
    std::cout << std::endl << "Solution around S0:" << std::endl;
    if (idx_S0 > 0) {
        for (int i = std::max(0, idx_S0-5); i <= std::min(nx-1, idx_S0+5); ++i) {
            double S = grid.point(i);
            double V = solution[i];
            double payoff = std::max(0.0, K - S);
            std::cout << "  S[" << i << "] = " << S << ", V = " << V 
                      << ", Payoff = " << payoff << std::endl;
        }
    }
    
    return 0;
}
