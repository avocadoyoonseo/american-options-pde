#ifndef PSOR_HPP
#define PSOR_HPP

#include <vector>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <iostream>

namespace fin {

/**
 * PSOR solver for the Linear Complementarity Problem (LCP)
 * Used for American option pricing to enforce the early exercise constraint
 * Solves: A*x >= b, x >= c, (A*x - b)^T * (x - c) = 0
 */
class PSORSolver {
public:
    /**
     * Constructor for PSORSolver
     * 
     * @param a Lower diagonal of the tridiagonal matrix A
     * @param b Main diagonal of the tridiagonal matrix A
     * @param c Upper diagonal of the tridiagonal matrix A
     * @param omega Relaxation parameter (1 < omega < 2 for over-relaxation)
     * @param tol Convergence tolerance
     * @param maxIter Maximum number of iterations
     */
    PSORSolver(const std::vector<double>& a, 
              const std::vector<double>& b, 
              const std::vector<double>& c,
              double omega = 1.5,
              double tol = 1e-6,
              int maxIter = 1000)
        : m_a(a), m_b(b), m_c(c), 
          m_omega(omega), m_tol(tol), m_maxIter(maxIter) {
        
        size_t n = b.size();
        if (n < 2) {
            throw std::invalid_argument("System size must be at least 2");
        }
        
        if (a.size() != n-1 || c.size() != n-1) {
            throw std::invalid_argument("Diagonal sizes are inconsistent");
        }
        
        if (omega <= 0.0 || omega >= 2.0) {
            throw std::invalid_argument("Relaxation parameter must be between 0 and 2");
        }
        
        if (tol <= 0.0) {
            throw std::invalid_argument("Tolerance must be positive");
        }
    }
    
    /**
     * Solve the LCP: A*x >= b, x >= c, (A*x - b)^T * (x - c) = 0
     * using the Projected SOR method
     * 
     * @param b Right-hand side vector
     * @param c Lower bound constraints (typically option payoffs)
     * @param initialGuess Initial solution guess (optional)
     * @return Solution vector x
     */
    std::vector<double> solve(const std::vector<double>& b, 
                             const std::vector<double>& c,
                             const std::vector<double>* initialGuess = nullptr) {
        size_t n = m_b.size();
        
        if (b.size() != n || c.size() != n) {
            throw std::invalid_argument("RHS vector or constraint size does not match system size");
        }
        
        // Initialize solution
        std::vector<double> x(n);
        if (initialGuess && initialGuess->size() == n) {
            x = *initialGuess;
        } else {
            x = c; // Start with payoff as initial guess
        }
        
        // Compute initial residual
        double residual = computeResidual(x, b);
        double initialResidual = residual;
        
        // PSOR iteration
        int iter = 0;
        while (iter < m_maxIter && residual > m_tol * initialResidual) {
            double maxChange = 0.0;
            
            // Forward sweep
            for (size_t i = 0; i < n; ++i) {
                double sum = b[i];
                
                // Lower diagonal contribution
                if (i > 0) {
                    sum -= m_a[i-1] * x[i-1];
                }
                
                // Upper diagonal contribution
                if (i < n-1) {
                    sum -= m_c[i] * x[i+1];
                }
                
                // Update solution with relaxation
                double xOld = x[i];
                double xNew = xOld + m_omega * (sum / m_b[i] - xOld);
                
                // Project onto constraint
                x[i] = std::max(xNew, c[i]);
                
                // Track maximum change
                maxChange = std::max(maxChange, std::abs(x[i] - xOld));
            }
            
            // Backward sweep (optional, for symmetric Gauss-Seidel)
            for (int i = n-1; i >= 0; --i) {
                double sum = b[i];
                
                // Lower diagonal contribution
                if (i > 0) {
                    sum -= m_a[i-1] * x[i-1];
                }
                
                // Upper diagonal contribution
                if (i < n-1) {
                    sum -= m_c[i] * x[i+1];
                }
                
                // Update solution with relaxation
                double xOld = x[i];
                double xNew = xOld + m_omega * (sum / m_b[i] - xOld);
                
                // Project onto constraint
                x[i] = std::max(xNew, c[i]);
                
                // Track maximum change
                maxChange = std::max(maxChange, std::abs(x[i] - xOld));
            }
            
            // Update residual and iteration counter
            residual = computeResidual(x, b);
            ++iter;
            
            // Early termination if convergence is stalled
            if (maxChange < m_tol) {
                break;
            }
        }
        
        m_lastIter = iter;
        m_lastResidual = residual;
        
        return x;
    }
    
    /**
     * Get the number of iterations from the last solve
     * 
     * @return Number of iterations
     */
    int getLastIterationCount() const {
        return m_lastIter;
    }
    
    /**
     * Get the final residual from the last solve
     * 
     * @return Final residual
     */
    double getLastResidual() const {
        return m_lastResidual;
    }
    
    /**
     * Set the relaxation parameter omega
     * 
     * @param omega Relaxation parameter (1 < omega < 2 for over-relaxation)
     */
    void setOmega(double omega) {
        if (omega <= 0.0 || omega >= 2.0) {
            throw std::invalid_argument("Relaxation parameter must be between 0 and 2");
        }
        m_omega = omega;
    }
    
    /**
     * Set the convergence tolerance
     * 
     * @param tol Convergence tolerance
     */
    void setTolerance(double tol) {
        if (tol <= 0.0) {
            throw std::invalid_argument("Tolerance must be positive");
        }
        m_tol = tol;
    }
    
    /**
     * Set the maximum number of iterations
     * 
     * @param maxIter Maximum number of iterations
     */
    void setMaxIterations(int maxIter) {
        if (maxIter <= 0) {
            throw std::invalid_argument("Maximum iterations must be positive");
        }
        m_maxIter = maxIter;
    }
    
    /**
     * Update the matrix diagonals
     * 
     * @param a Lower diagonal
     * @param b Main diagonal
     * @param c Upper diagonal
     */
    void updateMatrix(const std::vector<double>& a, 
                      const std::vector<double>& b, 
                      const std::vector<double>& c) {
        size_t n = b.size();
        
        if (m_b.size() != n || a.size() != n-1 || c.size() != n-1) {
            throw std::invalid_argument("New matrix dimensions inconsistent with original");
        }
        
        m_a = a;
        m_b = b;
        m_c = c;
    }

private:
    std::vector<double> m_a;  // Lower diagonal
    std::vector<double> m_b;  // Main diagonal
    std::vector<double> m_c;  // Upper diagonal
    double m_omega;           // Relaxation parameter
    double m_tol;             // Convergence tolerance
    int m_maxIter;            // Maximum number of iterations
    int m_lastIter;           // Iterations from last solve
    double m_lastResidual;    // Residual from last solve
    
    /**
     * Compute the residual ||A*x - b||_2 for the current solution
     * 
     * @param x Current solution vector
     * @param b Right-hand side vector
     * @return L2 norm of the residual
     */
    double computeResidual(const std::vector<double>& x, const std::vector<double>& b) const {
        size_t n = x.size();
        double residualNorm = 0.0;
        
        for (size_t i = 0; i < n; ++i) {
            double sum = m_b[i] * x[i];
            
            if (i > 0) {
                sum += m_a[i-1] * x[i-1];
            }
            
            if (i < n-1) {
                sum += m_c[i] * x[i+1];
            }
            
            double residual = sum - b[i];
            residualNorm += residual * residual;
        }
        
        return std::sqrt(residualNorm);
    }
};

} // namespace fin

#endif // PSOR_HPP
