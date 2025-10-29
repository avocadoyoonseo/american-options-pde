#ifndef AMR_HPP
#define AMR_HPP

#include "grid.hpp"
#include "american.hpp"
#include "penalty.hpp"
#include <vector>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <memory>
#include <functional>

namespace fin {

/**
 * AdaptiveMeshRefinement class for American option pricing
 * Uses adaptive mesh refinement to concentrate grid points near the exercise boundary
 */
class AdaptiveMeshRefinement {
public:
    /**
     * Constructor for AdaptiveMeshRefinement
     * 
     * @param initialGrid Initial spatial grid for the stock price
     * @param timePoints Vector of time points (in ascending order, starting from 0)
     * @param r Risk-free interest rate
     * @param q Dividend yield
     * @param sigma Volatility
     * @param isCall Whether the option is a call (true) or put (false)
     * @param usePenalty Whether to use penalty method (true) or PSOR (false)
     * @param maxRefinementLevel Maximum number of refinement levels
     * @param refinementThreshold Threshold for refinement (based on residual or option delta)
     */
    AdaptiveMeshRefinement(const Grid& initialGrid, 
                          const std::vector<double>& timePoints,
                          double r, double q, double sigma, 
                          bool isCall = false,
                          bool usePenalty = true,
                          int maxRefinementLevel = 3,
                          double refinementThreshold = 0.01)
        : m_initialGrid(initialGrid), m_timePoints(timePoints), 
          m_r(r), m_q(q), m_sigma(sigma), m_isCall(isCall),
          m_usePenalty(usePenalty), m_maxRefinementLevel(maxRefinementLevel),
          m_refinementThreshold(refinementThreshold) {
        
        // Validate inputs
        if (timePoints.size() < 2) {
            throw std::invalid_argument("Time grid must have at least 2 points");
        }
        if (maxRefinementLevel < 0) {
            throw std::invalid_argument("Maximum refinement level must be non-negative");
        }
        if (refinementThreshold <= 0.0) {
            throw std::invalid_argument("Refinement threshold must be positive");
        }
    }

    /**
     * Solve for the American option price using adaptive mesh refinement
     * 
     * @param K Strike price
     * @param psorOmega PSOR relaxation parameter (if PSOR is used)
     * @param psorTol PSOR convergence tolerance (if PSOR is used)
     * @param psorMaxIter Maximum PSOR iterations (if PSOR is used)
     * @param penaltyCoeff Penalty coefficient (if penalty method is used)
     * @return Final solution vector (option prices at t=0 for all grid points on finest grid)
     */
    std::vector<double> solve(double K, 
                             double psorOmega = 1.5,
                             double psorTol = 1e-6,
                             int psorMaxIter = 1000,
                             double penaltyCoeff = 1e6) {
        // Initial solution on coarse grid
        Grid currentGrid = m_initialGrid;
        std::vector<double> currentSolution;
        std::vector<double> exerciseBoundary;
        
        // Solve on initial grid
        if (m_usePenalty) {
            PenaltyMethod solver(currentGrid, m_timePoints, m_r, m_q, m_sigma, m_isCall, penaltyCoeff);
            currentSolution = solver.solve(K);
            exerciseBoundary = solver.getExerciseBoundary();
        } else {
            AmericanOption solver(currentGrid, m_timePoints, m_r, m_q, m_sigma, m_isCall, 
                                 psorOmega, psorTol, psorMaxIter);
            currentSolution = solver.solve(K);
            exerciseBoundary = solver.getExerciseBoundary();
        }
        
        // Store initial solution
        m_gridHierarchy.push_back(currentGrid);
        m_solutionHierarchy.push_back(currentSolution);
        
        // Refinement loop
        for (int level = 0; level < m_maxRefinementLevel; ++level) {
            // Detect regions needing refinement
            std::vector<std::pair<double, double>> regionsToRefine = 
                detectRefinementRegions(currentGrid, currentSolution, exerciseBoundary, K);
            
            if (regionsToRefine.empty()) {
                break; // No regions to refine
            }
            
            // Create refined grid
            Grid refinedGrid = createRefinedGrid(currentGrid, regionsToRefine);
            
            // Interpolate current solution to new grid
            std::vector<double> initialGuess(refinedGrid.size());
            for (size_t i = 0; i < refinedGrid.size(); ++i) {
                initialGuess[i] = currentGrid.interpolate(currentSolution, refinedGrid.point(i));
            }
            
            // Solve on refined grid
            if (m_usePenalty) {
                PenaltyMethod solver(refinedGrid, m_timePoints, m_r, m_q, m_sigma, m_isCall, penaltyCoeff);
                currentSolution = solver.solve(K);
                exerciseBoundary = solver.getExerciseBoundary();
            } else {
                AmericanOption solver(refinedGrid, m_timePoints, m_r, m_q, m_sigma, m_isCall, 
                                     psorOmega, psorTol, psorMaxIter);
                
                // If we have a good initial guess, we can use it
                if (!initialGuess.empty() && initialGuess.size() == refinedGrid.size()) {
                    // Custom solve with initial guess could be implemented
                    // For now, we just use the standard solve
                    currentSolution = solver.solve(K);
                } else {
                    currentSolution = solver.solve(K);
                }
                
                exerciseBoundary = solver.getExerciseBoundary();
            }
            
            // Store refined grid and solution
            m_gridHierarchy.push_back(refinedGrid);
            m_solutionHierarchy.push_back(currentSolution);
            
            // Update current grid
            currentGrid = refinedGrid;
            
            // Check if further refinement would exceed desired accuracy
            if (estimateError() < m_refinementThreshold) {
                break;
            }
        }
        
        // Return the solution on the finest grid
        return m_solutionHierarchy.back();
    }
    
    /**
     * Get the finest grid from the hierarchy
     * 
     * @return Finest grid used in the AMR process
     */
    const Grid& getFinestGrid() const {
        if (m_gridHierarchy.empty()) {
            throw std::runtime_error("No grid available, solve method must be called first");
        }
        return m_gridHierarchy.back();
    }
    
    /**
     * Get the option price on the finest grid at a specific stock price
     * 
     * @param S Stock price
     * @return Interpolated option price
     */
    double priceAt(double S) const {
        if (m_gridHierarchy.empty() || m_solutionHierarchy.empty()) {
            throw std::runtime_error("No solution available, solve method must be called first");
        }
        
        const Grid& finestGrid = m_gridHierarchy.back();
        const std::vector<double>& finestSolution = m_solutionHierarchy.back();
        
        return finestGrid.interpolate(finestSolution, S);
    }
    
    /**
     * Get the hierarchy of grids used in the AMR process
     * 
     * @return Vector of grids
     */
    const std::vector<Grid>& getGridHierarchy() const {
        return m_gridHierarchy;
    }
    
    /**
     * Get the hierarchy of solutions computed in the AMR process
     * 
     * @return Vector of solution vectors
     */
    const std::vector<std::vector<double>>& getSolutionHierarchy() const {
        return m_solutionHierarchy;
    }

private:
    Grid m_initialGrid;                     // Initial coarse grid
    std::vector<double> m_timePoints;       // Time discretization points
    double m_r;                             // Risk-free rate
    double m_q;                             // Dividend yield
    double m_sigma;                         // Volatility
    bool m_isCall;                          // Whether option is call (true) or put (false)
    bool m_usePenalty;                      // Use penalty method or PSOR
    int m_maxRefinementLevel;               // Maximum refinement levels
    double m_refinementThreshold;           // Error threshold for refinement
    
    // Hierarchy of grids and solutions
    std::vector<Grid> m_gridHierarchy;
    std::vector<std::vector<double>> m_solutionHierarchy;
    
    /**
     * Detect regions that need refinement based on option delta or residual
     * 
     * @param grid Current grid
     * @param solution Current solution
     * @param exerciseBoundary Early exercise boundary
     * @param K Strike price
     * @return Vector of regions (min, max) that need refinement
     */
    std::vector<std::pair<double, double>> detectRefinementRegions(
            const Grid& grid, 
            const std::vector<double>& solution,
            const std::vector<double>& exerciseBoundary,
            double K) const {
        
        std::vector<std::pair<double, double>> regions;
        
        // Calculate option delta (first derivative) as indicator
        std::vector<double> delta(grid.size() - 1);
        for (size_t i = 0; i < delta.size(); ++i) {
            delta[i] = (solution[i+1] - solution[i]) / (grid.point(i+1) - grid.point(i));
        }
        
        // Find points with high delta (near exercise boundary or strike)
        std::vector<size_t> highDeltaIndices;
        double maxDelta = *std::max_element(delta.begin(), delta.end(), 
                                           [](double a, double b) { return std::abs(a) < std::abs(b); });
        
        double deltaThreshold = 0.3 * std::abs(maxDelta); // Refinement threshold as fraction of max delta
        
        for (size_t i = 0; i < delta.size(); ++i) {
            if (std::abs(delta[i]) > deltaThreshold) {
                highDeltaIndices.push_back(i);
            }
        }
        
        // Also add points near the strike price
        size_t strikeIndex = grid.findNearestIndex(K);
        highDeltaIndices.push_back(strikeIndex > 0 ? strikeIndex - 1 : 0);
        highDeltaIndices.push_back(strikeIndex);
        highDeltaIndices.push_back(strikeIndex < grid.size() - 1 ? strikeIndex + 1 : grid.size() - 1);
        
        // Add points near exercise boundary for each time step
        for (double boundary : exerciseBoundary) {
            size_t boundaryIndex = grid.findNearestIndex(boundary);
            highDeltaIndices.push_back(boundaryIndex > 0 ? boundaryIndex - 1 : 0);
            highDeltaIndices.push_back(boundaryIndex);
            highDeltaIndices.push_back(boundaryIndex < grid.size() - 1 ? boundaryIndex + 1 : grid.size() - 1);
        }
        
        // Sort and remove duplicates
        std::sort(highDeltaIndices.begin(), highDeltaIndices.end());
        highDeltaIndices.erase(std::unique(highDeltaIndices.begin(), highDeltaIndices.end()), 
                              highDeltaIndices.end());
        
        // Group consecutive indices into regions
        if (highDeltaIndices.empty()) {
            return regions;
        }
        
        size_t start = highDeltaIndices[0];
        size_t end = start;
        
        for (size_t i = 1; i < highDeltaIndices.size(); ++i) {
            if (highDeltaIndices[i] == end + 1) {
                // Extend current region
                end = highDeltaIndices[i];
            } else {
                // Create region and start new one
                double regionMin = grid.point(start > 0 ? start - 1 : 0);
                double regionMax = grid.point(end + 1 < grid.size() ? end + 1 : grid.size() - 1);
                regions.push_back(std::make_pair(regionMin, regionMax));
                
                start = highDeltaIndices[i];
                end = start;
            }
        }
        
        // Add final region
        double regionMin = grid.point(start > 0 ? start - 1 : 0);
        double regionMax = grid.point(end + 1 < grid.size() ? end + 1 : grid.size() - 1);
        regions.push_back(std::make_pair(regionMin, regionMax));
        
        return regions;
    }
    
    /**
     * Create a refined grid based on identified regions
     * 
     * @param currentGrid Current grid
     * @param regionsToRefine Regions that need refinement
     * @return New refined grid
     */
    Grid createRefinedGrid(const Grid& currentGrid, 
                         const std::vector<std::pair<double, double>>& regionsToRefine) const {
        // If no regions to refine, return the original grid
        if (regionsToRefine.empty()) {
            return currentGrid;
        }
        
        // Start with points from the current grid
        std::vector<double> refinedPoints = currentGrid.points();
        
        // For each region, add additional points
        for (const auto& region : regionsToRefine) {
            double min = region.first;
            double max = region.second;
            
            // Find points already in this region
            std::vector<double> pointsInRegion;
            for (double p : refinedPoints) {
                if (p >= min && p <= max) {
                    pointsInRegion.push_back(p);
                }
            }
            
            // Add new points between existing points in the region
            if (pointsInRegion.size() >= 2) {
                std::vector<double> newPoints;
                for (size_t i = 0; i < pointsInRegion.size() - 1; ++i) {
                    double midpoint = (pointsInRegion[i] + pointsInRegion[i+1]) / 2.0;
                    newPoints.push_back(midpoint);
                }
                
                // Add new points to refined points
                refinedPoints.insert(refinedPoints.end(), newPoints.begin(), newPoints.end());
            }
        }
        
        // Sort and remove any duplicates
        std::sort(refinedPoints.begin(), refinedPoints.end());
        refinedPoints.erase(std::unique(refinedPoints.begin(), refinedPoints.end(),
                                      [](double a, double b) { return std::abs(a - b) < 1e-10; }),
                          refinedPoints.end());
        
        return Grid(refinedPoints);
    }
    
    /**
     * Estimate error between successive refinement levels
     * 
     * @return Estimated error
     */
    double estimateError() const {
        if (m_solutionHierarchy.size() < 2) {
            return std::numeric_limits<double>::max();
        }
        
        // Get the two finest grids and solutions
        const Grid& fineGrid = m_gridHierarchy.back();
        const std::vector<double>& fineSolution = m_solutionHierarchy.back();
        
        const Grid& coarseGrid = m_gridHierarchy[m_gridHierarchy.size() - 2];
        const std::vector<double>& coarseSolution = m_solutionHierarchy[m_solutionHierarchy.size() - 2];
        
        // Compute relative error at common points
        double maxRelError = 0.0;
        
        for (size_t i = 0; i < coarseGrid.size(); ++i) {
            double S = coarseGrid.point(i);
            double coarseValue = coarseSolution[i];
            double fineValue = fineGrid.interpolate(fineSolution, S);
            
            if (std::abs(coarseValue) > 1e-10) {
                double relError = std::abs((fineValue - coarseValue) / coarseValue);
                maxRelError = std::max(maxRelError, relError);
            }
        }
        
        return maxRelError;
    }
};

} // namespace fin

#endif // AMR_HPP
