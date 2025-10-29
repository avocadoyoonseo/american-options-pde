#ifndef AMERICAN_HPP
#define AMERICAN_HPP

#include "grid.hpp"
#include "tridiag.hpp"
#include "psor.hpp"
#include <vector>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <iostream>

namespace fin {

/**
 * AmericanOption class for American option pricing
 * Solves the Black-Scholes LCP using Crank-Nicolson and PSOR
 */
class AmericanOption {
public:
    /**
     * Constructor for the American option pricer
     * 
     * @param grid Spatial grid for the stock price
     * @param timePoints Vector of time points (in ascending order, starting from 0)
     * @param r Risk-free interest rate
     * @param q Dividend yield
     * @param sigma Volatility
     * @param isCall Whether the option is a call (true) or put (false)
     * @param omega PSOR relaxation parameter (1 < omega < 2 for over-relaxation)
     * @param psorTol PSOR convergence tolerance
     * @param psorMaxIter Maximum PSOR iterations
     */
    AmericanOption(const Grid& grid, 
                  const std::vector<double>& timePoints,
                  double r, double q, double sigma, 
                  bool isCall = false,
                  double omega = 1.5,
                  double psorTol = 1e-6,
                  int psorMaxIter = 1000)
        : m_grid(grid), m_timePoints(timePoints), 
          m_r(r), m_q(q), m_sigma(sigma), m_isCall(isCall),
          m_omega(omega), m_psorTol(psorTol), m_psorMaxIter(psorMaxIter) {
        
        // Validate inputs
        if (timePoints.size() < 2) {
            throw std::invalid_argument("Time grid must have at least 2 points");
        }
        if (timePoints[0] != 0.0) {
            throw std::invalid_argument("Time grid must start at t=0");
        }
        for (size_t i = 1; i < timePoints.size(); ++i) {
            if (timePoints[i] <= timePoints[i-1]) {
                throw std::invalid_argument("Time points must be strictly increasing");
            }
        }
        if (r < 0.0) {
            throw std::invalid_argument("Risk-free rate must be non-negative");
        }
        if (q < 0.0) {
            throw std::invalid_argument("Dividend yield must be non-negative");
        }
        if (sigma <= 0.0) {
            throw std::invalid_argument("Volatility must be positive");
        }
        
        // Pre-allocate solution arrays
        m_solution.resize(grid.size());
        m_exerciseBoundary.resize(timePoints.size());
        
        // Initialize diagonal coefficient vectors
        size_t n = grid.size();
        m_alpha.resize(n-1);
        m_beta.resize(n);
        m_gamma.resize(n-1);
        m_alphaExplicit.resize(n-1);
        m_betaExplicit.resize(n);
        m_gammaExplicit.resize(n-1);
    }

    /**
     * Set up the finite difference coefficients for the Crank-Nicolson scheme
     * This builds the tridiagonal system matrices for the implicit and explicit steps
     */
    void setupCoefficients(double dt) {
        size_t n = m_grid.size();
        double theta = 0.5; // Crank-Nicolson
        
        // Set up coefficients for the tridiagonal system
        for (size_t i = 1; i < n-1; ++i) {
            double S = m_grid.point(i);
            double dS_minus = m_grid.point(i) - m_grid.point(i-1);
            double dS_plus = m_grid.point(i+1) - m_grid.point(i);
            
            // Standard finite difference coefficients for non-uniform grid
            // Second derivative: d²V/dS²
            double alpha_2 = 2.0 / (dS_minus * (dS_minus + dS_plus));
            double beta_2 = -2.0 / (dS_minus * dS_plus);
            double gamma_2 = 2.0 / (dS_plus * (dS_minus + dS_plus));
            
            // First derivative: dV/dS (central difference)
            double alpha_1 = -dS_plus / (dS_minus * (dS_minus + dS_plus));
            double beta_1 = (dS_plus - dS_minus) / (dS_minus * dS_plus);
            double gamma_1 = dS_minus / (dS_plus * (dS_minus + dS_plus));
            
            // Black-Scholes PDE coefficients
            double sigma2 = m_sigma * m_sigma;
            double coeff_2 = 0.5 * sigma2 * S * S; // diffusion
            double coeff_1 = (m_r - m_q) * S;      // drift
            double coeff_0 = -m_r;                  // discount
            
            // Combine to get operator L coefficients
            double L_alpha = coeff_2 * alpha_2 + coeff_1 * alpha_1;
            double L_beta = coeff_2 * beta_2 + coeff_1 * beta_1 + coeff_0;
            double L_gamma = coeff_2 * gamma_2 + coeff_1 * gamma_1;
            
            // Implicit part (LHS): (I - θ*dt*L)V^{n+1}
            m_alpha[i-1] = -theta * dt * L_alpha;
            m_beta[i] = 1.0 - theta * dt * L_beta;
            m_gamma[i] = -theta * dt * L_gamma;
            
            // Explicit part (RHS): (I + (1-θ)*dt*L)V^{n}
            m_alphaExplicit[i-1] = (1.0 - theta) * dt * L_alpha;
            m_betaExplicit[i] = 1.0 + (1.0 - theta) * dt * L_beta;
            m_gammaExplicit[i] = (1.0 - theta) * dt * L_gamma;
        }
        
        // Dirichlet boundaries
        m_beta[0] = 1.0;
        m_gamma[0] = 0.0;
        m_betaExplicit[0] = 1.0;
        m_gammaExplicit[0] = 0.0;
        
        m_alpha[n-2] = 0.0;
        m_beta[n-1] = 1.0;
        m_alphaExplicit[n-2] = 0.0;
        m_betaExplicit[n-1] = 1.0;
    }

    /**
     * Apply boundary conditions for the given time
     * 
     * @param t Current time
     * @param K Strike price
     * @param rhs Right-hand side vector to update with boundary values
     */
    void applyBoundaryConditions(double t, double K, std::vector<double>& rhs) {
        size_t n = m_grid.size();
        double tau = m_timePoints.back() - t; // Time to maturity
        double S0 = m_grid.point(0);
        double Sn = m_grid.point(n-1);
        
        // Apply boundary conditions
        if (m_isCall) {
            // Call option boundary conditions
            rhs[0] = 0.0; // S -> 0, call option value -> 0
            
            // S -> infinity, call option value -> S - K*exp(-r*tau)
            double discountedStrike = K * std::exp(-m_r * tau);
            rhs[n-1] = Sn - discountedStrike;
        } else {
            // Put option boundary conditions
            rhs[0] = K * std::exp(-m_r * tau); // S -> 0, put option value -> K*exp(-r*tau)
            rhs[n-1] = 0.0; // S -> infinity, put option value -> 0
        }
    }
    
    /**
     * Calculate the payoff function for the option
     * 
     * @param K Strike price
     * @return Vector of payoff values for each grid point
     */
    std::vector<double> calculatePayoff(double K) const {
        size_t n = m_grid.size();
        std::vector<double> payoff(n);
        
        for (size_t i = 0; i < n; ++i) {
            double S = m_grid.point(i);
            if (m_isCall) {
                payoff[i] = std::max(0.0, S - K); // Call payoff
            } else {
                payoff[i] = std::max(0.0, K - S); // Put payoff
            }
        }
        
        return payoff;
    }

    /**
     * Solve for the American option price using Crank-Nicolson and PSOR
     * 
     * @param K Strike price
     * @return Final solution vector (option prices at t=0 for all grid points)
     */
    std::vector<double> solve(double K) {
        size_t n = m_grid.size();
        size_t numTimeSteps = m_timePoints.size() - 1;
        
        // Initialize solution at maturity with payoff
        std::vector<double> payoff = calculatePayoff(K);
        m_solution = payoff;
        
        // Record initial exercise boundary (at maturity)
        m_exerciseBoundary[numTimeSteps] = findExerciseBoundary(m_solution, payoff);
        
        // Time-stepping loop (backwards from maturity to t=0)
        for (size_t t = numTimeSteps; t > 0; --t) {
            double dt = m_timePoints[t] - m_timePoints[t-1];
            double currentTime = m_timePoints[t-1];
            
            // Setup coefficients for this time step
            setupCoefficients(dt);
            
            // Build right-hand side vector for current time step
            std::vector<double> rhs(n);
            for (size_t i = 1; i < n-1; ++i) {
                rhs[i] = m_alphaExplicit[i-1] * m_solution[i-1] +
                         m_betaExplicit[i] * m_solution[i] +
                         m_gammaExplicit[i] * m_solution[i+1];
            }
            
            // Copy boundary values
            rhs[0] = m_betaExplicit[0] * m_solution[0] + m_gammaExplicit[0] * m_solution[1];
            rhs[n-1] = m_alphaExplicit[n-2] * m_solution[n-2] + m_betaExplicit[n-1] * m_solution[n-1];
            
            // Apply boundary conditions
            applyBoundaryConditions(currentTime, K, rhs);
            
            // Create PSOR solver with updated coefficients and solve LCP
            PSORSolver psorSolver(m_alpha, m_beta, m_gamma, m_omega, m_psorTol, m_psorMaxIter);
            m_solution = psorSolver.solve(rhs, payoff, &m_solution);
            
            // Record exercise boundary
            m_exerciseBoundary[t-1] = findExerciseBoundary(m_solution, payoff);
        }
        
        return m_solution;
    }
    
    /**
     * Get option price at a specific stock price by interpolation
     * 
     * @param S Stock price
     * @return Interpolated option price
     */
    double priceAt(double S) const {
        return m_grid.interpolate(m_solution, S);
    }
    
    /**
     * Get the entire solution vector
     * 
     * @return Vector of option prices for all grid points
     */
    const std::vector<double>& getSolution() const {
        return m_solution;
    }
    
    /**
     * Get the grid used for discretization
     * 
     * @return Reference to the spatial grid
     */
    const Grid& getGrid() const {
        return m_grid;
    }
    
    /**
     * Get the exercise boundary S*(t) for all time points
     * 
     * @return Vector of exercise boundary values
     */
    const std::vector<double>& getExerciseBoundary() const {
        return m_exerciseBoundary;
    }

private:
    Grid m_grid;                       // Spatial grid
    std::vector<double> m_timePoints;  // Time discretization points
    double m_r;                        // Risk-free rate
    double m_q;                        // Dividend yield
    double m_sigma;                    // Volatility
    bool m_isCall;                     // Whether option is call (true) or put (false)
    
    // PSOR parameters
    double m_omega;                    // Relaxation parameter
    double m_psorTol;                  // PSOR tolerance
    int m_psorMaxIter;                 // Max PSOR iterations
    
    // Solution vector and exercise boundary
    std::vector<double> m_solution;
    std::vector<double> m_exerciseBoundary;
    
    // Tridiagonal system coefficients for implicit step
    std::vector<double> m_alpha;       // Lower diagonal
    std::vector<double> m_beta;        // Main diagonal
    std::vector<double> m_gamma;       // Upper diagonal
    
    // Tridiagonal system coefficients for explicit step
    std::vector<double> m_alphaExplicit; // Lower diagonal
    std::vector<double> m_betaExplicit;  // Main diagonal
    std::vector<double> m_gammaExplicit; // Upper diagonal
    
    /**
     * Find the early exercise boundary S*(t) at a given time step
     * by finding the largest S where option value equals payoff
     * 
     * @param values Option values at current time step
     * @param payoff Payoff values
     * @return Exercise boundary stock price
     */
    double findExerciseBoundary(const std::vector<double>& values, 
                               const std::vector<double>& payoff) const {
        size_t n = m_grid.size();
        
        if (m_isCall) {
            // For call options, find the smallest S where option > payoff
            for (size_t i = n-1; i > 0; --i) {
                if (std::abs(values[i] - payoff[i]) < 1e-10) {
                    return m_grid.point(i);
                }
            }
        } else {
            // For put options, find the largest S where option = payoff
            for (size_t i = 0; i < n; ++i) {
                if (std::abs(values[i] - payoff[i]) < 1e-10) {
                    return m_grid.point(i);
                }
            }
        }
        
        // If no exercise boundary found
        return m_isCall ? m_grid.xMax() : m_grid.xMin();
    }
};

} // namespace fin

#endif // AMERICAN_HPP
