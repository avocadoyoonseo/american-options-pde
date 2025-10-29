#include "../include/grid.hpp"
#include "../include/cn.hpp"
#include "../include/penalty.hpp"
#include "../include/american.hpp"
#include "../include/utils.hpp"
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <fstream>
#include <chrono>

int main(int argc, char* argv[]) {
    // Default parameters
    double S0 = 100.0;        // Initial stock price
    double K = 100.0;         // Strike price
    double r = 0.05;          // Risk-free rate
    double q = 0.02;          // Dividend yield
    double sigma = 0.2;       // Volatility
    double T = 1.0;           // Time to maturity
    bool isCall = false;      // Put option by default
    
    // Grid parameters
    double xmin = K * 0.1;    // Lower bound for grid
    double xmax = K * 3.0;    // Upper bound for grid
    int nx = 200;             // Number of spatial grid points
    int nt = 100;             // Number of time steps
    double omega = 1.5;       // PSOR relaxation parameter (for comparison)
    double penaltyCoeff = 1e6; // Penalty coefficient
    
    // Process command line arguments
    for (int i = 1; i < argc; i += 2) {
        std::string arg = argv[i];
        if (i + 1 < argc) {
            if (arg == "--S0") S0 = std::stod(argv[i + 1]);
            else if (arg == "--K") K = std::stod(argv[i + 1]);
            else if (arg == "--r") r = std::stod(argv[i + 1]);
            else if (arg == "--q") q = std::stod(argv[i + 1]);
            else if (arg == "--sigma") sigma = std::stod(argv[i + 1]);
            else if (arg == "--T") T = std::stod(argv[i + 1]);
            else if (arg == "--type") isCall = (std::string(argv[i + 1]) == "call");
            else if (arg == "--nx") nx = std::stoi(argv[i + 1]);
            else if (arg == "--nt") nt = std::stoi(argv[i + 1]);
            else if (arg == "--omega") omega = std::stod(argv[i + 1]);
            else if (arg == "--penalty") penaltyCoeff = std::stod(argv[i + 1]);
        }
    }
    
    std::cout << "=====================================================================" << std::endl;
    std::cout << "American Option Pricing with Penalty Method vs PSOR" << std::endl;
    std::cout << "=====================================================================" << std::endl;
    std::cout << "Parameters: " << fin::OptionUtils::formatOptionParams(S0, K, r, q, sigma, T, isCall) << std::endl;
    std::cout << "Grid: nx=" << nx << ", nt=" << nt << ", omega=" << omega << std::endl;
    std::cout << "Penalty coefficient: " << penaltyCoeff << std::endl;
    std::cout << "=====================================================================" << std::endl;
    
    // Create spatial grid
    fin::Grid grid = fin::Grid::createOptionGrid(S0, K, sigma, T, nx, 0.1, 3.0);
    
    // Create time grid
    std::vector<double> timePoints = fin::Grid::createTimeGrid(T, nt);
    
    // Price American option using binomial tree (reference)
    int nCRR = 1000; // Number of steps for binomial tree
    auto [crr, crrTime] = fin::OptionUtils::timeExecution([&]() {
        return fin::CRRBinomialTree::priceAmerican(S0, K, r, q, sigma, T, isCall, nCRR);
    });
    
    std::cout << "CRR Binomial price (American with n=" << nCRR << "): " << crr << std::endl;
    std::cout << "Execution time: " << crrTime << " seconds" << std::endl;
    std::cout << "--------------------------------------------------------------------" << std::endl;
    
    // Price American option using Crank-Nicolson with PSOR
    auto [psorSolution, psorTime] = fin::OptionUtils::timeExecution([&]() {
        fin::AmericanOption am(grid, timePoints, r, q, sigma, isCall, omega);
        return am.solve(K);
    });
    double psorPrice = grid.interpolate(psorSolution, S0);
    
    std::cout << "American price (CN+PSOR): " << psorPrice << std::endl;
    std::cout << "Error vs CRR (bps): " << fin::OptionUtils::errorInBps(psorPrice, crr) << std::endl;
    std::cout << "Execution time: " << psorTime << " seconds" << std::endl;
    std::cout << "--------------------------------------------------------------------" << std::endl;
    
    // Price American option using Crank-Nicolson with Penalty method
    auto [penaltySolution, penaltyTime] = fin::OptionUtils::timeExecution([&]() {
        fin::PenaltyMethod pm(grid, timePoints, r, q, sigma, isCall, penaltyCoeff);
        return pm.solve(K);
    });
    double penaltyPrice = grid.interpolate(penaltySolution, S0);
    
    std::cout << "American price (Penalty): " << penaltyPrice << std::endl;
    std::cout << "Error vs CRR (bps): " << fin::OptionUtils::errorInBps(penaltyPrice, crr) << std::endl;
    std::cout << "Execution time: " << penaltyTime << " seconds" << std::endl;
    std::cout << "--------------------------------------------------------------------" << std::endl;
    
    // Compare PSOR and Penalty methods
    std::cout << "PSOR vs Penalty comparison:" << std::endl;
    std::cout << "Price difference: " << std::abs(psorPrice - penaltyPrice) << std::endl;
    std::cout << "Relative difference (bps): " << fin::OptionUtils::errorInBps(penaltyPrice, psorPrice) << std::endl;
    
    double speedup = psorTime / penaltyTime;
    std::cout << "Penalty method speedup: " << speedup << "x" << std::endl;
    std::cout << "=====================================================================" << std::endl;
    
    // Study effect of penalty coefficient
    std::cout << "Studying effect of penalty coefficient..." << std::endl;
    std::vector<double> penalties = {1e2, 1e3, 1e4, 1e5, 1e6, 1e7};
    std::vector<double> penaltyPrices;
    std::vector<double> penaltyTimes;
    std::vector<double> penaltyErrors;
    
    for (double pc : penalties) {
        auto [sol, time] = fin::OptionUtils::timeExecution([&]() {
            fin::PenaltyMethod pm(grid, timePoints, r, q, sigma, isCall, pc);
            return pm.solve(K);
        });
        
        double price = grid.interpolate(sol, S0);
        penaltyPrices.push_back(price);
        penaltyTimes.push_back(time);
        penaltyErrors.push_back(fin::OptionUtils::errorInBps(price, crr));
        
        std::cout << "Penalty = " << pc << ": Price = " << price 
                  << ", Error (bps) = " << penaltyErrors.back()
                  << ", Time = " << time << " s" << std::endl;
    }
    
    // Find optimal penalty coefficient
    auto minErrorIt = std::min_element(penaltyErrors.begin(), penaltyErrors.end());
    int minIndex = std::distance(penaltyErrors.begin(), minErrorIt);
    std::cout << "Optimal penalty coefficient: " << penalties[minIndex] 
              << " (Error: " << penaltyErrors[minIndex] << " bps)" << std::endl;
    std::cout << "=====================================================================" << std::endl;
    
    // Calculate early exercise boundary for both methods
    fin::AmericanOption amOption(grid, timePoints, r, q, sigma, isCall, omega);
    amOption.solve(K);
    std::vector<double> psorBoundary = amOption.getExerciseBoundary();
    
    fin::PenaltyMethod pmOption(grid, timePoints, r, q, sigma, isCall, penaltyCoeff);
    pmOption.solve(K);
    std::vector<double> penaltyBoundary = pmOption.getExerciseBoundary();
    
    // Save early exercise boundaries to file
    std::ofstream boundaryFile("data/exercise_boundary_comparison.csv");
    boundaryFile << "Time,PSOR_Boundary,Penalty_Boundary\n";
    
    for (size_t i = 0; i < timePoints.size(); ++i) {
        boundaryFile << timePoints[i] << "," << psorBoundary[i] << "," << penaltyBoundary[i] << "\n";
    }
    boundaryFile.close();
    
    std::cout << "Early exercise boundaries saved to data/exercise_boundary_comparison.csv" << std::endl;
    
    return 0;
}
