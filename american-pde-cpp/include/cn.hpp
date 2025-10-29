#ifndef CN_HPP
#define CN_HPP

#include "grid.hpp"
#include "tridiag.hpp"
#include <vector>
#include <cmath>
#include <algorithm>
#include <stdexcept>

namespace fin {

/**
 * CrankNicolson class for European option pricing
 * Solves the Black-Scholes PDE using the Crank-Nicolson finite difference scheme
 */
class CrankNicolson {
public:
    /**
     * Constructor for the Crank-Nicolson solver
     * 
     * @param grid Spatial grid for the stock price
     * @param timePoints Vector of time points (in ascending order, starting from 0)
     * @param r Risk-free interest rate
     * @param q Dividend yield
     * @param sigma Volatility
     * @param isCall Whether the option is a call (true) or put (false)
     */
    CrankNicolson(const Grid& grid, 
                 const std::vector<double>& timePoints,
                 double r, double q, double sigma, 
                 bool isCall = false)
        : m_grid(grid), m_timePoints(timePoints), 
          m_r(r), m_q(q), m_sigma(sigma), m_isCall(isCall) {
        
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
        double Sn = m_grid.point(n-1);
        
        // Apply Dirichlet boundary conditions - only set RHS values
        // The coefficient matrices are already set correctly in setupCoefficients
        if (m_isCall) {
            // Call option boundary conditions
            rhs[0] = 0.0;  // At S=0, call value = 0
            
            // At S=Smax, call value ~ S - K*exp(-r*tau)
            double discountedStrike = K * std::exp(-m_r * tau);
            rhs[n-1] = Sn - discountedStrike;
            
        } else {
            // Put option boundary conditions
            double discountedStrike = K * std::exp(-m_r * tau);
            rhs[0] = discountedStrike;  // At S=0, put value = K*exp(-r*tau)
            rhs[n-1] = 0.0;  // At S=Smax, put value = 0
        }
    }

    /**
     * Solve the Black-Scholes PDE using Crank-Nicolson method
     * 
     * @param K Strike price
     * @return Final solution vector (option prices at t=0 for all grid points)
     */
    std::vector<double> solve(double K) {
        size_t n = m_grid.size();
        size_t numTimeSteps = m_timePoints.size() - 1;
        
        // Initialize solution at maturity with payoff
        for (size_t i = 0; i < n; ++i) {
            double S = m_grid.point(i);
            if (m_isCall) {
                m_solution[i] = std::max(0.0, S - K); // Call payoff
            } else {
                m_solution[i] = std::max(0.0, K - S); // Put payoff
            }
        }
        
        // Time-stepping loop (backwards from maturity to t=0)
        for (size_t t = numTimeSteps; t > 0; --t) {
            double dt = m_timePoints[t] - m_timePoints[t-1];
            double currentTime = m_timePoints[t-1];
            
            // Recompute CN coefficients for this dt
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
            
            // Create solver with updated coefficients and solve
            TridiagonalSolver solver(m_alpha, m_beta, m_gamma);
            m_solution = solver.solve(rhs);
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

private:
    Grid m_grid;                       // Spatial grid
    std::vector<double> m_timePoints;  // Time discretization points
    double m_r;                        // Risk-free rate
    double m_q;                        // Dividend yield
    double m_sigma;                    // Volatility
    bool m_isCall;                     // Whether option is call (true) or put (false)
    
    // Solution vector
    std::vector<double> m_solution;
    
    // Tridiagonal system coefficients for implicit step
    std::vector<double> m_alpha;       // Lower diagonal
    std::vector<double> m_beta;        // Main diagonal
    std::vector<double> m_gamma;       // Upper diagonal
    
    // Tridiagonal system coefficients for explicit step
    std::vector<double> m_alphaExplicit; // Lower diagonal
    std::vector<double> m_betaExplicit;  // Main diagonal
    std::vector<double> m_gammaExplicit; // Upper diagonal
};

} // namespace fin

#endif // CN_HPP
