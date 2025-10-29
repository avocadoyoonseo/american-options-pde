#ifndef GRID_HPP
#define GRID_HPP

#include <vector>
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <iostream>

namespace fin {

/**
 * Grid class for spatial and temporal discretization
 * Handles both uniform and non-uniform grids
 */
class Grid {
public:
    // Create a uniform grid
    static Grid createUniform(double xMin, double xMax, int nPoints) {
        if (nPoints < 2) {
            throw std::invalid_argument("Grid must have at least 2 points");
        }
        if (xMin >= xMax) {
            throw std::invalid_argument("xMin must be less than xMax");
        }

        std::vector<double> points(nPoints);
        double dx = (xMax - xMin) / (nPoints - 1);
        
        for (int i = 0; i < nPoints; ++i) {
            points[i] = xMin + i * dx;
        }
        
        return Grid(points);
    }

    // Create a non-uniform grid with exponential spacing
    static Grid createNonUniform(double xMin, double xMax, int nPoints, double concentration, double center) {
        if (nPoints < 2) {
            throw std::invalid_argument("Grid must have at least 2 points");
        }
        if (xMin >= xMax) {
            throw std::invalid_argument("xMin must be less than xMax");
        }
        if (concentration <= 0) {
            throw std::invalid_argument("Concentration parameter must be positive");
        }

        std::vector<double> points(nPoints);
        
        // Map to [-1,1] interval with concentration at center
        for (int i = 0; i < nPoints; ++i) {
            double t = -1.0 + 2.0 * i / (nPoints - 1.0);
            double s = std::sinh(concentration * (t - center)) / std::sinh(concentration);
            points[i] = xMin + (xMax - xMin) * (s + 1) / 2.0;
        }
        
        return Grid(points);
    }

    // Create a grid specifically for option pricing with concentration near strike
    static Grid createOptionGrid(double S0, double K, double sigma, double T, 
                                int nPoints, double leftMult = 0.1, double rightMult = 5.0) {
        // Use simple uniform grid for now to debug
        double xMin = K * 0.1;   // 10% of strike
        double xMax = K * 3.0;    // 300% of strike
        
        // Create uniform grid
        return createUniform(xMin, xMax, nPoints);
    }

    // Create uniform time grid
    static std::vector<double> createTimeGrid(double T, int nSteps) {
        if (nSteps < 1) {
            throw std::invalid_argument("Time grid must have at least 1 step");
        }
        if (T <= 0) {
            throw std::invalid_argument("Maturity T must be positive");
        }

        std::vector<double> timePoints(nSteps + 1);
        double dt = T / nSteps;
        
        for (int i = 0; i <= nSteps; ++i) {
            timePoints[i] = i * dt;
        }
        
        return timePoints;
    }

    // Constructor with vector of grid points
    Grid(const std::vector<double>& gridPoints) : m_points(gridPoints) {
        if (gridPoints.size() < 2) {
            throw std::invalid_argument("Grid must have at least 2 points");
        }
        
        // Validate monotonicity
        for (size_t i = 1; i < m_points.size(); ++i) {
            if (m_points[i] <= m_points[i-1]) {
                throw std::invalid_argument("Grid points must be strictly increasing");
            }
        }
        
        // Precompute step sizes and midpoints
        m_dx.resize(m_points.size() - 1);
        m_midpoints.resize(m_points.size() - 1);
        
        for (size_t i = 0; i < m_points.size() - 1; ++i) {
            m_dx[i] = m_points[i+1] - m_points[i];
            m_midpoints[i] = (m_points[i+1] + m_points[i]) / 2.0;
        }
    }

    // Access grid properties
    size_t size() const { return m_points.size(); }
    const std::vector<double>& points() const { return m_points; }
    const std::vector<double>& stepSizes() const { return m_dx; }
    const std::vector<double>& midpoints() const { return m_midpoints; }
    
    // Access individual points
    double point(size_t i) const { 
        if (i >= m_points.size()) {
            throw std::out_of_range("Grid index out of range");
        }
        return m_points[i]; 
    }
    
    // Access step size at index i
    double dx(size_t i) const { 
        if (i >= m_dx.size()) {
            throw std::out_of_range("Grid step size index out of range");
        }
        return m_dx[i]; 
    }

    // Return min and max values
    double xMin() const { return m_points.front(); }
    double xMax() const { return m_points.back(); }
    
    // Find closest grid index for a given x value
    size_t findNearestIndex(double x) const {
        if (x <= m_points.front()) {
            return 0;
        }
        if (x >= m_points.back()) {
            return m_points.size() - 1;
        }
        
        auto it = std::lower_bound(m_points.begin(), m_points.end(), x);
        size_t idx = std::distance(m_points.begin(), it);
        
        // Check which of the two neighboring points is closer
        if (idx > 0 && std::abs(x - m_points[idx-1]) < std::abs(x - m_points[idx])) {
            return idx - 1;
        }
        return idx;
    }
    
    // Linear interpolation to get value at arbitrary point
    double interpolate(const std::vector<double>& values, double x) const {
        if (values.size() != m_points.size()) {
            throw std::invalid_argument("Values vector size must match grid size");
        }
        
        if (x <= m_points.front()) {
            return values.front();
        }
        if (x >= m_points.back()) {
            return values.back();
        }
        
        size_t i = findNearestIndex(x);
        if (m_points[i] == x) {
            return values[i];
        }
        
        // Ensure we have a valid interval for interpolation
        if (i == m_points.size() - 1) {
            i--;
        }
        
        double t = (x - m_points[i]) / (m_points[i+1] - m_points[i]);
        return (1.0 - t) * values[i] + t * values[i+1];
    }

private:
    std::vector<double> m_points;    // Grid points
    std::vector<double> m_dx;        // Step sizes between points
    std::vector<double> m_midpoints; // Midpoints between grid points
};

} // namespace fin

#endif // GRID_HPP
