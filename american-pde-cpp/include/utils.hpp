#ifndef UTILS_HPP
#define UTILS_HPP

#include <vector>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <string>
#include <memory>
#include <chrono>
#include <random>

namespace fin {

/**
 * Black-Scholes analytical formulas for European options
 */
class BlackScholes {
public:
    /**
     * Calculate the Black-Scholes price for a European option
     * 
     * @param S Stock price
     * @param K Strike price
     * @param r Risk-free interest rate
     * @param q Dividend yield
     * @param sigma Volatility
     * @param T Time to maturity
     * @param isCall Whether the option is a call (true) or put (false)
     * @return European option price
     */
    static double price(double S, double K, double r, double q, double sigma, double T, bool isCall) {
        if (T <= 0.0) {
            // At maturity, return payoff
            return isCall ? std::max(0.0, S - K) : std::max(0.0, K - S);
        }
        
        double d1 = (std::log(S/K) + (r - q + 0.5*sigma*sigma)*T) / (sigma * std::sqrt(T));
        double d2 = d1 - sigma * std::sqrt(T);
        
        if (isCall) {
            return S * std::exp(-q*T) * normalCDF(d1) - K * std::exp(-r*T) * normalCDF(d2);
        } else {
            return K * std::exp(-r*T) * normalCDF(-d2) - S * std::exp(-q*T) * normalCDF(-d1);
        }
    }
    
    /**
     * Calculate the Black-Scholes delta for a European option
     * 
     * @param S Stock price
     * @param K Strike price
     * @param r Risk-free interest rate
     * @param q Dividend yield
     * @param sigma Volatility
     * @param T Time to maturity
     * @param isCall Whether the option is a call (true) or put (false)
     * @return European option delta
     */
    static double delta(double S, double K, double r, double q, double sigma, double T, bool isCall) {
        if (T <= 0.0) {
            // At maturity, delta is either 0 or 1
            if (isCall) {
                return S > K ? 1.0 : 0.0;
            } else {
                return S < K ? -1.0 : 0.0;
            }
        }
        
        double d1 = (std::log(S/K) + (r - q + 0.5*sigma*sigma)*T) / (sigma * std::sqrt(T));
        
        if (isCall) {
            return std::exp(-q*T) * normalCDF(d1);
        } else {
            return std::exp(-q*T) * (normalCDF(d1) - 1.0);
        }
    }
    
private:
    /**
     * Standard normal cumulative distribution function
     * 
     * @param x Input value
     * @return Probability N(x)
     */
    static double normalCDF(double x) {
        // Use the complementary error function for accurate CDF
        return 0.5 * std::erfc(-x / std::sqrt(2.0));
    }
};

/**
 * Cox-Ross-Rubinstein (CRR) binomial tree model for option pricing
 */
class CRRBinomialTree {
public:
    /**
     * Calculate the price of an American option using the CRR binomial tree
     * 
     * @param S0 Initial stock price
     * @param K Strike price
     * @param r Risk-free interest rate
     * @param q Dividend yield
     * @param sigma Volatility
     * @param T Time to maturity
     * @param isCall Whether the option is a call (true) or put (false)
     * @param n Number of time steps in the tree
     * @return American option price
     */
    static double priceAmerican(double S0, double K, double r, double q, double sigma, double T, 
                               bool isCall, int n) {
        if (n < 1) {
            throw std::invalid_argument("Number of time steps must be positive");
        }
        
        double dt = T / n;
        double u = std::exp(sigma * std::sqrt(dt));
        double d = 1.0 / u;
        double p = (std::exp((r - q) * dt) - d) / (u - d);
        double discount = std::exp(-r * dt);
        
        // Initialize terminal stock prices and option values
        std::vector<double> S(n+1);
        std::vector<double> V(n+1);
        
        for (int i = 0; i <= n; ++i) {
            S[i] = S0 * std::pow(u, i) * std::pow(d, n-i);
            V[i] = isCall ? std::max(0.0, S[i] - K) : std::max(0.0, K - S[i]);
        }
        
        // Backward induction
        for (int step = n-1; step >= 0; --step) {
            for (int i = 0; i <= step; ++i) {
                // Calculate stock price at this node
                S[i] = S0 * std::pow(u, i) * std::pow(d, step-i);
                
                // Calculate continuation value
                double continuation = discount * (p * V[i+1] + (1.0 - p) * V[i]);
                
                // Calculate exercise value
                double exercise = isCall ? std::max(0.0, S[i] - K) : std::max(0.0, K - S[i]);
                
                // Option value is max of continuation and exercise
                V[i] = std::max(continuation, exercise);
            }
        }
        
        return V[0];
    }
    
    /**
     * Calculate the price of a European option using the CRR binomial tree
     * 
     * @param S0 Initial stock price
     * @param K Strike price
     * @param r Risk-free interest rate
     * @param q Dividend yield
     * @param sigma Volatility
     * @param T Time to maturity
     * @param isCall Whether the option is a call (true) or put (false)
     * @param n Number of time steps in the tree
     * @return European option price
     */
    static double priceEuropean(double S0, double K, double r, double q, double sigma, double T, 
                               bool isCall, int n) {
        if (n < 1) {
            throw std::invalid_argument("Number of time steps must be positive");
        }
        
        double dt = T / n;
        double u = std::exp(sigma * std::sqrt(dt));
        double d = 1.0 / u;
        double p = (std::exp((r - q) * dt) - d) / (u - d);
        double discount = std::exp(-r * dt);
        
        // Initialize terminal stock prices and option values
        std::vector<double> V(n+1);
        
        for (int i = 0; i <= n; ++i) {
            double ST = S0 * std::pow(u, i) * std::pow(d, n-i);
            V[i] = isCall ? std::max(0.0, ST - K) : std::max(0.0, K - ST);
        }
        
        // Backward induction
        for (int step = n-1; step >= 0; --step) {
            for (int i = 0; i <= step; ++i) {
                // For European options, no early exercise check
                V[i] = discount * (p * V[i+1] + (1.0 - p) * V[i]);
            }
        }
        
        return V[0];
    }
};

/**
 * Longstaff-Schwartz Monte Carlo method for American option pricing
 */
class LongstaffSchwartzMC {
public:
    /**
     * Calculate the price of an American option using Longstaff-Schwartz Monte Carlo
     * 
     * @param S0 Initial stock price
     * @param K Strike price
     * @param r Risk-free interest rate
     * @param q Dividend yield
     * @param sigma Volatility
     * @param T Time to maturity
     * @param isCall Whether the option is a call (true) or put (false)
     * @param nSteps Number of time steps
     * @param nPaths Number of Monte Carlo paths
     * @param nBasis Number of basis functions for regression
     * @return American option price
     */
    static double price(double S0, double K, double r, double q, double sigma, double T, 
                       bool isCall, int nSteps = 50, int nPaths = 10000, int nBasis = 3) {
        // Generate paths
        std::vector<std::vector<double>> paths = generatePaths(S0, r, q, sigma, T, nSteps, nPaths);
        
        double dt = T / nSteps;
        double discount = std::exp(-r * dt);
        
        // Initialize cash flows at maturity
        std::vector<double> cashFlows(nPaths);
        for (int i = 0; i < nPaths; ++i) {
            cashFlows[i] = isCall ? std::max(0.0, paths[i][nSteps] - K) : std::max(0.0, K - paths[i][nSteps]);
        }
        
        // Backward induction through time steps
        for (int t = nSteps - 1; t >= 0; --t) {
            // Identify in-the-money paths
            std::vector<int> itm;
            for (int i = 0; i < nPaths; ++i) {
                if ((isCall && paths[i][t] > K) || (!isCall && paths[i][t] < K)) {
                    itm.push_back(i);
                }
            }
            
            // Skip if no in-the-money paths
            if (itm.empty()) {
                continue;
            }
            
            // Extract in-the-money stock prices and cash flows
            int nItm = itm.size();
            std::vector<double> X(nItm);
            std::vector<double> Y(nItm);
            
            for (int j = 0; j < nItm; ++j) {
                int i = itm[j];
                X[j] = paths[i][t];
                Y[j] = cashFlows[i] * discount; // Discounted future cash flow
            }
            
            // Perform least squares regression
            std::vector<double> beta = leastSquaresRegression(X, Y, nBasis);
            
            // Compare continuation value with immediate exercise
            for (int j = 0; j < nItm; ++j) {
                int i = itm[j];
                double continuation = evaluatePolynomial(beta, X[j]);
                double immediate = isCall ? std::max(0.0, X[j] - K) : std::max(0.0, K - X[j]);
                
                // Early exercise if immediate > continuation
                if (immediate > continuation) {
                    cashFlows[i] = immediate;
                } else {
                    cashFlows[i] *= discount; // Keep future cash flow (discounted)
                }
            }
        }
        
        // Average cash flows to get option price
        double sum = 0.0;
        for (double cf : cashFlows) {
            sum += cf;
        }
        
        return sum / nPaths;
    }
    
private:
    /**
     * Generate stock price paths using geometric Brownian motion
     * 
     * @param S0 Initial stock price
     * @param r Risk-free interest rate
     * @param q Dividend yield
     * @param sigma Volatility
     * @param T Time to maturity
     * @param nSteps Number of time steps
     * @param nPaths Number of Monte Carlo paths
     * @return Vector of stock price paths
     */
    static std::vector<std::vector<double>> generatePaths(double S0, double r, double q, double sigma, 
                                                        double T, int nSteps, int nPaths) {
        double dt = T / nSteps;
        double drift = (r - q - 0.5 * sigma * sigma) * dt;
        double vol = sigma * std::sqrt(dt);
        
        // Initialize random number generator
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<double> dist(0.0, 1.0);
        
        // Generate paths
        std::vector<std::vector<double>> paths(nPaths, std::vector<double>(nSteps + 1));
        
        for (int i = 0; i < nPaths; ++i) {
            paths[i][0] = S0;
            
            for (int t = 0; t < nSteps; ++t) {
                double Z = dist(gen);
                paths[i][t+1] = paths[i][t] * std::exp(drift + vol * Z);
            }
        }
        
        return paths;
    }
    
    /**
     * Perform least squares regression to fit basis functions
     * 
     * @param X Independent variable (stock prices)
     * @param Y Dependent variable (discounted cash flows)
     * @param nBasis Number of basis functions (polynomial degree + 1)
     * @return Vector of regression coefficients
     */
    static std::vector<double> leastSquaresRegression(const std::vector<double>& X, 
                                                     const std::vector<double>& Y, int nBasis) {
        int n = X.size();
        
        // Build design matrix
        std::vector<std::vector<double>> A(n, std::vector<double>(nBasis));
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < nBasis; ++j) {
                A[i][j] = std::pow(X[i], j);
            }
        }
        
        // Compute A^T * A
        std::vector<std::vector<double>> ATA(nBasis, std::vector<double>(nBasis, 0.0));
        for (int i = 0; i < nBasis; ++i) {
            for (int j = 0; j < nBasis; ++j) {
                for (int k = 0; k < n; ++k) {
                    ATA[i][j] += A[k][i] * A[k][j];
                }
            }
        }
        
        // Compute A^T * Y
        std::vector<double> ATY(nBasis, 0.0);
        for (int i = 0; i < nBasis; ++i) {
            for (int k = 0; k < n; ++k) {
                ATY[i] += A[k][i] * Y[k];
            }
        }
        
        // Solve ATA * beta = ATY using Gaussian elimination
        // (For simplicity, we're using a basic approach without pivoting)
        for (int i = 0; i < nBasis; ++i) {
            // Normalize row i
            double pivot = ATA[i][i];
            for (int j = i; j < nBasis; ++j) {
                ATA[i][j] /= pivot;
            }
            ATY[i] /= pivot;
            
            // Eliminate column i from rows below
            for (int j = i + 1; j < nBasis; ++j) {
                double factor = ATA[j][i];
                for (int k = i; k < nBasis; ++k) {
                    ATA[j][k] -= factor * ATA[i][k];
                }
                ATY[j] -= factor * ATY[i];
            }
        }
        
        // Back-substitution
        std::vector<double> beta(nBasis);
        for (int i = nBasis - 1; i >= 0; --i) {
            beta[i] = ATY[i];
            for (int j = i + 1; j < nBasis; ++j) {
                beta[i] -= ATA[i][j] * beta[j];
            }
        }
        
        return beta;
    }
    
    /**
     * Evaluate polynomial with given coefficients at point x
     * 
     * @param coef Polynomial coefficients (ascending power)
     * @param x Evaluation point
     * @return Polynomial value at x
     */
    static double evaluatePolynomial(const std::vector<double>& coef, double x) {
        double result = 0.0;
        for (size_t i = 0; i < coef.size(); ++i) {
            result += coef[i] * std::pow(x, i);
        }
        return result;
    }
};

/**
 * Utility functions for convergence analysis and result output
 */
class OptionUtils {
public:
    /**
     * Calculate error in basis points (1 bp = 0.01%)
     * 
     * @param value Computed value
     * @param reference Reference value
     * @return Error in basis points
     */
    static double errorInBps(double value, double reference) {
        return std::abs(value - reference) / reference * 10000.0;
    }
    
    /**
     * Calculate convergence rate between three successive refinements
     * 
     * @param error1 Error at coarsest level
     * @param error2 Error at medium level
     * @param error3 Error at finest level
     * @param ratio Grid size ratio between successive levels
     * @return Estimated convergence rate
     */
    static double convergenceRate(double error1, double error2, double error3, double ratio = 2.0) {
        double r1 = std::log(error1 / error2) / std::log(ratio);
        double r2 = std::log(error2 / error3) / std::log(ratio);
        return (r1 + r2) / 2.0;
    }
    
    /**
     * Save convergence table to a CSV file
     * 
     * @param filename Output filename
     * @param gridSizes Vector of grid sizes
     * @param values Vector of computed values
     * @param reference Reference value for error calculation
     * @param method Method name for the header
     */
    static void saveConvergenceTable(const std::string& filename,
                                    const std::vector<int>& gridSizes,
                                    const std::vector<double>& values,
                                    double reference,
                                    const std::string& method) {
        if (gridSizes.size() != values.size()) {
            throw std::invalid_argument("Grid sizes and values must have the same length");
        }
        
        std::ofstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Could not open file: " + filename);
        }
        
        // Write header
        file << "Method,Grid Size,Value,Error (bps),Convergence Rate\n";
        
        // Write data
        for (size_t i = 0; i < gridSizes.size(); ++i) {
            double error = errorInBps(values[i], reference);
            double rate = 0.0;
            
            if (i >= 2) {
                double error1 = errorInBps(values[i-2], reference);
                double error2 = errorInBps(values[i-1], reference);
                double error3 = error;
                rate = convergenceRate(error1, error2, error3);
            }
            
            file << method << "," << gridSizes[i] << "," << values[i] << "," << error;
            if (i >= 2) {
                file << "," << rate;
            } else {
                file << ",";
            }
            file << "\n";
        }
    }
    
    /**
     * Save the early exercise boundary to a CSV file
     * 
     * @param filename Output filename
     * @param timePoints Vector of time points
     * @param boundary Vector of boundary values
     */
    static void saveExerciseBoundary(const std::string& filename,
                                    const std::vector<double>& timePoints,
                                    const std::vector<double>& boundary) {
        if (timePoints.size() != boundary.size()) {
            throw std::invalid_argument("Time points and boundary must have the same length");
        }
        
        std::ofstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Could not open file: " + filename);
        }
        
        // Write header
        file << "Time,ExerciseBoundary\n";
        
        // Write data
        for (size_t i = 0; i < timePoints.size(); ++i) {
            file << timePoints[i] << "," << boundary[i] << "\n";
        }
    }
    
    /**
     * Format option parameters as a string
     * 
     * @param S0 Initial stock price
     * @param K Strike price
     * @param r Risk-free interest rate
     * @param q Dividend yield
     * @param sigma Volatility
     * @param T Time to maturity
     * @param isCall Whether the option is a call (true) or put (false)
     * @return Formatted string
     */
    static std::string formatOptionParams(double S0, double K, double r, double q, double sigma, double T, bool isCall) {
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(4);
        oss << (isCall ? "Call" : "Put") << ", ";
        oss << "S0=" << S0 << ", ";
        oss << "K=" << K << ", ";
        oss << "r=" << r << ", ";
        oss << "q=" << q << ", ";
        oss << "sigma=" << sigma << ", ";
        oss << "T=" << T;
        return oss.str();
    }
    
    /**
     * Time a function execution
     * 
     * @param func Function to time
     * @return Pair of (result, execution time in seconds)
     */
    template<typename Func>
    static auto timeExecution(Func&& func) {
        auto start = std::chrono::high_resolution_clock::now();
        auto result = func();
        auto end = std::chrono::high_resolution_clock::now();
        
        std::chrono::duration<double> elapsed = end - start;
        return std::make_pair(result, elapsed.count());
    }
};

} // namespace fin

#endif // UTILS_HPP
