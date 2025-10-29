#include "../include/grid.hpp"
#include "../include/cn.hpp"
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
    double omega = 1.5;       // PSOR relaxation parameter
    
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
        }
    }
    
    std::cout << "=====================================================================" << std::endl;
    std::cout << "European & American Option Pricing with Crank-Nicolson + PSOR" << std::endl;
    std::cout << "=====================================================================" << std::endl;
    std::cout << "Parameters: " << fin::OptionUtils::formatOptionParams(S0, K, r, q, sigma, T, isCall) << std::endl;
    std::cout << "Grid: nx=" << nx << ", nt=" << nt << ", omega=" << omega << std::endl;
    std::cout << "=====================================================================" << std::endl;
    
    // Create spatial grid with proper boundaries for option pricing
    // Use a wider range to capture important price movements (3-4 standard deviations)
    double stdDev = sigma * std::sqrt(T);
    double xMin = std::max(0.01 * K, S0 * std::exp(-4.0 * stdDev)); // Avoid S=0 numerical issues
    double xMax = S0 * std::exp(4.0 * stdDev);
    
    // Create grid with concentration around strike price and current stock price
    fin::Grid grid = fin::Grid::createOptionGrid(S0, K, sigma, T, nx);
    
    // Create time grid
    std::vector<double> timePoints = fin::Grid::createTimeGrid(T, nt);
    
    // Price European option using analytical Black-Scholes
    double bsPrice = fin::BlackScholes::price(S0, K, r, q, sigma, T, isCall);
    std::cout << "Black-Scholes price (European): " << bsPrice << std::endl;
    
    // Price European option using Crank-Nicolson
    auto [cnSolution, cnTime] = fin::OptionUtils::timeExecution([&]() {
        fin::CrankNicolson cn(grid, timePoints, r, q, sigma, isCall);
        return cn.solve(K);
    });
    double cnPrice = grid.interpolate(cnSolution, S0);
    
    std::cout << "Crank-Nicolson price (European): " << cnPrice << std::endl;
    std::cout << "Error (bps): " << fin::OptionUtils::errorInBps(cnPrice, bsPrice) << std::endl;
    std::cout << "Execution time: " << cnTime << " seconds" << std::endl;
    std::cout << "--------------------------------------------------------------------" << std::endl;
    
    // Price American option using binomial tree (reference)
    int nCRR = 1000; // Number of steps for binomial tree
    auto [crr, crrTime] = fin::OptionUtils::timeExecution([&]() {
        return fin::CRRBinomialTree::priceAmerican(S0, K, r, q, sigma, T, isCall, nCRR);
    });
    
    std::cout << "CRR Binomial price (American with n=" << nCRR << "): " << crr << std::endl;
    std::cout << "Execution time: " << crrTime << " seconds" << std::endl;
    std::cout << "--------------------------------------------------------------------" << std::endl;
    
    // Price American option using Crank-Nicolson with PSOR
    auto [amSolution, amTime] = fin::OptionUtils::timeExecution([&]() {
        fin::AmericanOption am(grid, timePoints, r, q, sigma, isCall, omega);
        return am.solve(K);
    });
    double amPrice = grid.interpolate(amSolution, S0);
    
    std::cout << "American price (CN+PSOR): " << amPrice << std::endl;
    std::cout << "Error vs CRR (bps): " << fin::OptionUtils::errorInBps(amPrice, crr) << std::endl;
    std::cout << "Execution time: " << amTime << " seconds" << std::endl;
    
    // Compare European vs American prices
    double earlyExercisePremium = amPrice - cnPrice;
    std::cout << "Early exercise premium: " << earlyExercisePremium << std::endl;
    std::cout << "=====================================================================" << std::endl;
    
    // Perform convergence study with different grid sizes
    std::cout << "Running convergence study..." << std::endl;
    
    std::vector<int> gridSizes = {50, 100, 200, 400, 800};
    std::vector<double> europeanValues;
    std::vector<double> americanValues;
    std::vector<double> europeanTimes;
    std::vector<double> americanTimes;
    
    for (int n : gridSizes) {
        // Create refined grid
        fin::Grid refGrid = fin::Grid::createOptionGrid(S0, K, sigma, T, n, 0.1, 3.0);
        std::vector<double> refTimePoints = fin::Grid::createTimeGrid(T, n);
        
        // European option
        auto [euSol, euTime] = fin::OptionUtils::timeExecution([&]() {
            fin::CrankNicolson cn(refGrid, refTimePoints, r, q, sigma, isCall);
            return cn.solve(K);
        });
        europeanValues.push_back(refGrid.interpolate(euSol, S0));
        europeanTimes.push_back(euTime);
        
        // American option
        auto [amSol, amTime] = fin::OptionUtils::timeExecution([&]() {
            fin::AmericanOption am(refGrid, refTimePoints, r, q, sigma, isCall, omega);
            return am.solve(K);
        });
        americanValues.push_back(refGrid.interpolate(amSol, S0));
        americanTimes.push_back(amTime);
        
        std::cout << "Grid size " << n << " - European: " << europeanValues.back() 
                  << ", American: " << americanValues.back() << std::endl;
    }
    
    // Save convergence data to file
    fin::OptionUtils::saveConvergenceTable("data/european_convergence.csv", gridSizes, europeanValues, bsPrice, "CN");
    fin::OptionUtils::saveConvergenceTable("data/american_convergence.csv", gridSizes, americanValues, crr, "CN+PSOR");
    
    std::cout << "Convergence data saved to data/european_convergence.csv and data/american_convergence.csv" << std::endl;
    
    // Study the effect of omega parameter on PSOR convergence
    std::cout << "=====================================================================" << std::endl;
    std::cout << "Studying effect of omega parameter on PSOR convergence..." << std::endl;
    
    std::vector<double> omegas = {1.0, 1.2, 1.4, 1.6, 1.8, 1.9, 1.95};
    std::vector<double> omegaTimes;
    std::vector<int> omegaIters;
    
    for (double w : omegas) {
        fin::AmericanOption am(grid, timePoints, r, q, sigma, isCall, w);
        
        auto start = std::chrono::high_resolution_clock::now();
        am.solve(K);
        auto end = std::chrono::high_resolution_clock::now();
        
        std::chrono::duration<double> elapsed = end - start;
        omegaTimes.push_back(elapsed.count());
        
        // Note: in a real implementation, we would track iterations and residuals
        // from the PSOR solver and report them here
        
        std::cout << "omega = " << w << ": Time = " << omegaTimes.back() << " s" << std::endl;
    }
    
    // Find optimal omega
    auto minTimeIt = std::min_element(omegaTimes.begin(), omegaTimes.end());
    int minIndex = std::distance(omegaTimes.begin(), minTimeIt);
    std::cout << "Optimal omega: " << omegas[minIndex] << " (Time: " << omegaTimes[minIndex] << " s)" << std::endl;
    
    return 0;
}
