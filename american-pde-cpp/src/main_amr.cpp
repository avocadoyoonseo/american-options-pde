#include "../include/grid.hpp"
#include "../include/amr.hpp"
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
    int nxInitial = 50;       // Initial number of spatial grid points
    int ntInitial = 25;       // Initial number of time steps
    int maxRefinementLevel = 3; // Maximum number of refinement levels
    double refinementThreshold = 0.005; // Threshold for refinement
    bool usePenalty = true;   // Use penalty method (true) or PSOR (false)
    
    // Solver parameters
    double omega = 1.5;       // PSOR relaxation parameter
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
            else if (arg == "--nx") nxInitial = std::stoi(argv[i + 1]);
            else if (arg == "--nt") ntInitial = std::stoi(argv[i + 1]);
            else if (arg == "--refine") maxRefinementLevel = std::stoi(argv[i + 1]);
            else if (arg == "--threshold") refinementThreshold = std::stod(argv[i + 1]);
            else if (arg == "--method") usePenalty = (std::string(argv[i + 1]) == "penalty");
            else if (arg == "--omega") omega = std::stod(argv[i + 1]);
            else if (arg == "--penalty") penaltyCoeff = std::stod(argv[i + 1]);
        }
    }
    
    std::cout << "=====================================================================" << std::endl;
    std::cout << "American Option Pricing with Adaptive Mesh Refinement" << std::endl;
    std::cout << "=====================================================================" << std::endl;
    std::cout << "Parameters: " << fin::OptionUtils::formatOptionParams(S0, K, r, q, sigma, T, isCall) << std::endl;
    std::cout << "Initial grid: nx=" << nxInitial << ", nt=" << ntInitial << std::endl;
    std::cout << "Refinement: max_levels=" << maxRefinementLevel << ", threshold=" << refinementThreshold << std::endl;
    std::cout << "Solver: " << (usePenalty ? "Penalty method" : "PSOR") << std::endl;
    std::cout << "=====================================================================" << std::endl;
    
    // Create initial spatial grid
    fin::Grid initialGrid = fin::Grid::createOptionGrid(S0, K, sigma, T, nxInitial, 0.1, 3.0);
    
    // Create time grid
    std::vector<double> timePoints = fin::Grid::createTimeGrid(T, ntInitial);
    
    // Price American option using binomial tree (reference)
    int nCRR = 1000; // Number of steps for binomial tree
    auto [crr, crrTime] = fin::OptionUtils::timeExecution([&]() {
        return fin::CRRBinomialTree::priceAmerican(S0, K, r, q, sigma, T, isCall, nCRR);
    });
    
    std::cout << "CRR Binomial price (reference): " << crr << std::endl;
    std::cout << "Execution time: " << crrTime << " seconds" << std::endl;
    std::cout << "--------------------------------------------------------------------" << std::endl;
    
    // Use Longstaff-Schwartz MC as additional reference
    int nPaths = 10000;
    int nSteps = 100;
    auto [lsmc, lsmcTime] = fin::OptionUtils::timeExecution([&]() {
        return fin::LongstaffSchwartzMC::price(S0, K, r, q, sigma, T, isCall, nSteps, nPaths);
    });
    
    std::cout << "Longstaff-Schwartz MC price (n=" << nPaths << " paths): " << lsmc << std::endl;
    std::cout << "Execution time: " << lsmcTime << " seconds" << std::endl;
    std::cout << "--------------------------------------------------------------------" << std::endl;
    
    // Solve with AMR
    auto [amrSolution, amrTime] = fin::OptionUtils::timeExecution([&]() {
        fin::AdaptiveMeshRefinement amr(initialGrid, timePoints, r, q, sigma, isCall, 
                                       usePenalty, maxRefinementLevel, refinementThreshold);
        return amr.solve(K, omega, 1e-6, 1000, penaltyCoeff);
    });
    
    // Calculate price at S0
    fin::AdaptiveMeshRefinement amr(initialGrid, timePoints, r, q, sigma, isCall, 
                                   usePenalty, maxRefinementLevel, refinementThreshold);
    amr.solve(K, omega, 1e-6, 1000, penaltyCoeff);
    double amrPrice = amr.priceAt(S0);
    
    // Get the grid hierarchy
    const std::vector<fin::Grid>& gridHierarchy = amr.getGridHierarchy();
    const std::vector<std::vector<double>>& solutionHierarchy = amr.getSolutionHierarchy();
    
    std::cout << "AMR price: " << amrPrice << std::endl;
    std::cout << "Error vs CRR (bps): " << fin::OptionUtils::errorInBps(amrPrice, crr) << std::endl;
    std::cout << "Error vs LSMC (bps): " << fin::OptionUtils::errorInBps(amrPrice, lsmc) << std::endl;
    std::cout << "Execution time: " << amrTime << " seconds" << std::endl;
    std::cout << "Number of refinement levels: " << gridHierarchy.size() << std::endl;
    
    // Print grid sizes at each level
    std::cout << "Grid sizes:" << std::endl;
    for (size_t i = 0; i < gridHierarchy.size(); ++i) {
        std::cout << "  Level " << i << ": " << gridHierarchy[i].size() << " points" << std::endl;
    }
    std::cout << "--------------------------------------------------------------------" << std::endl;
    
    // Analyze efficiency compared to uniform grid refinement
    std::cout << "Efficiency analysis:" << std::endl;
    
    // For comparison, solve on a uniform fine grid with equivalent number of points
    size_t totalPointsAMR = 0;
    for (const auto& grid : gridHierarchy) {
        totalPointsAMR += grid.size();
    }
    
    int nUniform = static_cast<int>(std::sqrt(totalPointsAMR));
    fin::Grid uniformGrid = fin::Grid::createOptionGrid(S0, K, sigma, T, nUniform, 0.1, 3.0);
    std::vector<double> uniformTimePoints = fin::Grid::createTimeGrid(T, nUniform);
    
    double uniformPrice;
    double uniformTime;
    
    if (usePenalty) {
        auto [sol, time] = fin::OptionUtils::timeExecution([&]() {
            fin::PenaltyMethod pm(uniformGrid, uniformTimePoints, r, q, sigma, isCall, penaltyCoeff);
            std::vector<double> solution = pm.solve(K);
            return uniformGrid.interpolate(solution, S0);
        });
        uniformPrice = sol;
        uniformTime = time;
    } else {
        auto [sol, time] = fin::OptionUtils::timeExecution([&]() {
            fin::AmericanOption am(uniformGrid, uniformTimePoints, r, q, sigma, isCall, omega);
            std::vector<double> solution = am.solve(K);
            return uniformGrid.interpolate(solution, S0);
        });
        uniformPrice = sol;
        uniformTime = time;
    }
    
    std::cout << "Uniform grid (" << nUniform << " points): Price = " << uniformPrice 
              << ", Time = " << uniformTime << " s" << std::endl;
    std::cout << "AMR efficiency gain: " << uniformTime / amrTime << "x faster" << std::endl;
    std::cout << "AMR error vs uniform (bps): " 
              << fin::OptionUtils::errorInBps(amrPrice, uniformPrice) << std::endl;
    
    // Save grid points and option values to visualize mesh refinement
    std::ofstream meshFile("data/adaptive_mesh.csv");
    meshFile << "Level,GridPoint,OptionValue\n";
    
    for (size_t level = 0; level < gridHierarchy.size(); ++level) {
        const auto& grid = gridHierarchy[level];
        const auto& solution = solutionHierarchy[level];
        
        for (size_t i = 0; i < grid.size(); ++i) {
            meshFile << level << "," << grid.point(i) << "," << solution[i] << "\n";
        }
    }
    meshFile.close();
    
    std::cout << "Adaptive mesh data saved to data/adaptive_mesh.csv" << std::endl;
    std::cout << "=====================================================================" << std::endl;
    
    // Study impact of refinement threshold
    std::cout << "Studying impact of refinement threshold..." << std::endl;
    
    std::vector<double> thresholds = {0.1, 0.05, 0.01, 0.005, 0.001};
    std::vector<double> thresholdPrices;
    std::vector<double> thresholdTimes;
    std::vector<double> thresholdErrors;
    std::vector<int> thresholdPoints;
    
    for (double thresh : thresholds) {
        auto [solution, time] = fin::OptionUtils::timeExecution([&]() {
            fin::AdaptiveMeshRefinement amr(initialGrid, timePoints, r, q, sigma, isCall, 
                                           usePenalty, maxRefinementLevel, thresh);
            return amr.solve(K, omega, 1e-6, 1000, penaltyCoeff);
        });
        
        fin::AdaptiveMeshRefinement amr(initialGrid, timePoints, r, q, sigma, isCall, 
                                       usePenalty, maxRefinementLevel, thresh);
        amr.solve(K, omega, 1e-6, 1000, penaltyCoeff);
        double price = amr.priceAt(S0);
        
        // Count total grid points across all levels
        int totalPoints = 0;
        for (const auto& grid : amr.getGridHierarchy()) {
            totalPoints += grid.size();
        }
        
        thresholdPrices.push_back(price);
        thresholdTimes.push_back(time);
        thresholdErrors.push_back(fin::OptionUtils::errorInBps(price, crr));
        thresholdPoints.push_back(totalPoints);
        
        std::cout << "Threshold = " << thresh << ": Price = " << price 
                  << ", Error (bps) = " << thresholdErrors.back()
                  << ", Points = " << totalPoints
                  << ", Time = " << time << " s" << std::endl;
    }
    
    // Save threshold study data
    std::ofstream thresholdFile("data/threshold_study.csv");
    thresholdFile << "Threshold,Price,Error(bps),Points,Time\n";
    
    for (size_t i = 0; i < thresholds.size(); ++i) {
        thresholdFile << thresholds[i] << "," << thresholdPrices[i] << ","
                      << thresholdErrors[i] << "," << thresholdPoints[i] << ","
                      << thresholdTimes[i] << "\n";
    }
    thresholdFile.close();
    
    std::cout << "Threshold study data saved to data/threshold_study.csv" << std::endl;
    
    return 0;
}
