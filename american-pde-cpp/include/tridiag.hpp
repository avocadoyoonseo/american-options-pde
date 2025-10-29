#ifndef TRIDIAG_HPP
#define TRIDIAG_HPP

#include <vector>
#include <stdexcept>
#include <cmath>

namespace fin {

/**
 * Tridiagonal matrix solver class
 * Implements Thomas algorithm for efficiently solving Ax = d where A is tridiagonal
 */
class TridiagonalSolver {
public:
    /**
     * Constructor for TridiagonalSolver
     * 
     * @param a Lower diagonal elements (size n-1, indexed from 0 to n-2)
     * @param b Main diagonal elements (size n, indexed from 0 to n-1)
     * @param c Upper diagonal elements (size n-1, indexed from 0 to n-2)
     */
    TridiagonalSolver(const std::vector<double>& a, 
                      const std::vector<double>& b, 
                      const std::vector<double>& c) 
        : m_a(a), m_b(b), m_c(c) {
        
        size_t n = b.size();
        if (n < 2) {
            throw std::invalid_argument("System size must be at least 2");
        }
        
        if (a.size() != n-1 || c.size() != n-1) {
            throw std::invalid_argument("Diagonal sizes are inconsistent");
        }
        
        // Precompute coefficients for Thomas algorithm
        m_cPrime.resize(n-1);
        m_dPrime.resize(n);
    }
    
    /**
     * Solve the tridiagonal system Ax = d using Thomas algorithm
     * 
     * @param d Right-hand side vector (size n)
     * @return Solution vector x (size n)
     */
    std::vector<double> solve(const std::vector<double>& d) {
        size_t n = m_b.size();
        
        if (d.size() != n) {
            throw std::invalid_argument("RHS vector size does not match system size");
        }
        
        std::vector<double> x(n);
        
        // Forward elimination phase
        m_cPrime[0] = m_c[0] / m_b[0];
        m_dPrime[0] = d[0] / m_b[0];
        
        for (size_t i = 1; i < n-1; i++) {
            double denominator = m_b[i] - m_a[i-1] * m_cPrime[i-1];
            if (std::abs(denominator) < 1e-15) {
                throw std::runtime_error("Singular matrix detected during tridiagonal solve");
            }
            
            m_cPrime[i] = m_c[i] / denominator;
            m_dPrime[i] = (d[i] - m_a[i-1] * m_dPrime[i-1]) / denominator;
        }
        
        // Last row
        m_dPrime[n-1] = (d[n-1] - m_a[n-2] * m_dPrime[n-2]) / 
                        (m_b[n-1] - m_a[n-2] * m_cPrime[n-2]);
        
        // Back substitution phase
        x[n-1] = m_dPrime[n-1];
        
        for (int i = n-2; i >= 0; i--) {
            x[i] = m_dPrime[i] - m_cPrime[i] * x[i+1];
        }
        
        return x;
    }
    
    /**
     * Update diagonal elements of the matrix
     * Useful when reusing the same structure with different coefficients
     * 
     * @param a New lower diagonal
     * @param b New main diagonal
     * @param c New upper diagonal
     */
    void updateDiagonals(const std::vector<double>& a, 
                        const std::vector<double>& b, 
                        const std::vector<double>& c) {
        size_t n = b.size();
        
        if (n != m_b.size() || a.size() != n-1 || c.size() != n-1) {
            throw std::invalid_argument("New diagonals have inconsistent sizes");
        }
        
        m_a = a;
        m_b = b;
        m_c = c;
    }
    
    /**
     * Solve tridiagonal system with periodic boundary conditions
     * This is a more complex algorithm than the standard Thomas algorithm
     * 
     * @param a Lower diagonal elements (size n, with a[0] linking last and first equations)
     * @param b Main diagonal elements (size n)
     * @param c Upper diagonal elements (size n, with c[n-1] linking last and first equations)
     * @param d Right-hand side vector (size n)
     * @return Solution vector x (size n)
     */
    static std::vector<double> solvePeriodic(
            const std::vector<double>& a, 
            const std::vector<double>& b, 
            const std::vector<double>& c, 
            const std::vector<double>& d) {
        
        size_t n = d.size();
        if (a.size() != n || b.size() != n || c.size() != n) {
            throw std::invalid_argument("Inconsistent matrix dimensions for periodic system");
        }
        
        if (n <= 2) {
            throw std::invalid_argument("Periodic system size must be greater than 2");
        }
        
        // Special handling for periodic boundary conditions
        std::vector<double> u(n, 0.0);
        std::vector<double> z(n, 0.0);
        
        // Set up modified system
        std::vector<double> bb = b;
        std::vector<double> aa(n-1), cc(n-1), dd(n);
        
        double gamma = -b[0]; // Modified value for the corners
        
        // Copy values for modified system
        for (size_t i = 0; i < n-1; i++) {
            aa[i] = a[i+1];
            cc[i] = c[i];
        }
        for (size_t i = 0; i < n; i++) {
            dd[i] = d[i];
        }
        
        // Set up the vectors u and z
        u[0] = gamma;
        u[n-1] = a[0];
        
        z[0] = 1.0;
        z[n-1] = c[n-1] / gamma;
        
        // Solve the modified system with standard tridiagonal solver
        TridiagonalSolver solver(aa, bb, cc);
        std::vector<double> y = solver.solve(dd);
        std::vector<double> q = solver.solve(u);
        
        // Compute the solution
        double fact = (y[0] + c[n-1] * y[n-1] / gamma) / 
                     (1.0 + q[0] + c[n-1] * q[n-1] / gamma);
        
        std::vector<double> x(n);
        for (size_t i = 0; i < n; i++) {
            x[i] = y[i] - fact * q[i];
        }
        
        return x;
    }
    
    /**
     * Static function to directly solve a tridiagonal system
     * Convenience method when only one solution is needed
     * 
     * @param a Lower diagonal elements (size n-1)
     * @param b Main diagonal elements (size n)
     * @param c Upper diagonal elements (size n-1)
     * @param d Right-hand side vector (size n)
     * @return Solution vector x (size n)
     */
    static std::vector<double> solve(
            const std::vector<double>& a, 
            const std::vector<double>& b, 
            const std::vector<double>& c, 
            const std::vector<double>& d) {
        
        TridiagonalSolver solver(a, b, c);
        return solver.solve(d);
    }

private:
    std::vector<double> m_a;      // Lower diagonal (indexed 0..n-2)
    std::vector<double> m_b;      // Main diagonal (indexed 0..n-1)
    std::vector<double> m_c;      // Upper diagonal (indexed 0..n-2)
    std::vector<double> m_cPrime; // Modified upper diagonal for Thomas algorithm
    std::vector<double> m_dPrime; // Modified RHS for Thomas algorithm
};

} // namespace fin

#endif // TRIDIAG_HPP
