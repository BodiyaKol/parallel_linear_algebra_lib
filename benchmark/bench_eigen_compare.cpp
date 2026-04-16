#include <Eigen/Eigenvalues>

#include <algorithm>
#include <chrono>
#include <complex>
#include <fstream>
#include <iostream>
#include <random>
#include <vector>

#include "pla.h"

using Clock = std::chrono::high_resolution_clock;

template<typename F>
double measure_ms(F&& func, int repeats) {
    double total = 0.0;

    for (int i = 0; i < repeats; ++i) {
        auto start = Clock::now();
        func();
        auto end = Clock::now();
        total += std::chrono::duration<double, std::milli>(end - start).count();
    }

    return total / repeats;
}

pla::Matrix<double> make_random_pla_matrix(std::size_t n) {
    pla::Matrix<double> A(n, n);
    std::mt19937 gen(42 + static_cast<unsigned>(n));
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            A(i, j) = dist(gen);

    return A;
}

Eigen::MatrixXd to_eigen(const pla::Matrix<double>& A) {
    Eigen::MatrixXd E(A.rows(), A.cols());

    for (std::size_t i = 0; i < A.rows(); ++i)
        for (std::size_t j = 0; j < A.cols(); ++j)
            E(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j)) = A(i, j);

    return E;
}

double max_matching_error(
    std::vector<std::complex<double>> pla_vals,
    std::vector<std::complex<double>> eigen_vals
) {
    if (pla_vals.size() != eigen_vals.size()) {
        return std::numeric_limits<double>::infinity();
    }

    double max_err = 0.0;

    for (const auto& p : pla_vals) {
        auto best_it = std::min_element(
            eigen_vals.begin(),
            eigen_vals.end(),
            [&](const auto& a, const auto& b) {
                return std::abs(p - a) < std::abs(p - b);
            }
        );

        const double err = std::abs(p - *best_it);
        max_err = std::max(max_err, err);
        eigen_vals.erase(best_it);
    }

    return max_err;
}

std::vector<std::complex<double>> eigen_values_to_vector(
    const Eigen::VectorXcd& values
) {
    std::vector<std::complex<double>> result;
    result.reserve(static_cast<std::size_t>(values.size()));

    for (Eigen::Index i = 0; i < values.size(); ++i) {
        result.push_back(values(i));
    }

    return result;
}

int main() {
    std::vector<std::size_t> sizes = {
        2, 3, 4, 5, 8, 10, 16, 32, 64, 128, 256, 512, 1024
    };

    std::ofstream out("eigen_benchmark.csv");
    out << "size,total_pla_ms,eigen_ms,threshold_ms,ratio,passed,max_abs_error,pla_converged\n";

    for (std::size_t n : sizes) {
        const int repeats = (n <= 16) ? 10 : (n <= 128 ? 5 : 3);

        auto A = make_random_pla_matrix(n);
        auto E = to_eigen(A);

        pla::EigenOptions<double> popt;
        popt.tolerance = 1e-10;
        popt.max_iterations = 3000;

        auto pla_once = pla::eigenvalues_general(A, popt);

        Eigen::EigenSolver<Eigen::MatrixXd> eigen_once(E, false);

        auto pla_vals = pla_once.values;
        auto eigen_vals = eigen_values_to_vector(eigen_once.eigenvalues());

        const double max_error = max_matching_error(pla_vals, eigen_vals);

        double total_pla_ms = measure_ms([&]() {
            auto result = pla::eigenvalues_general(A, popt);
            volatile std::size_t sink = result.values.size();
            (void)sink;
        }, repeats);

        double eigen_ms = measure_ms([&]() {
            Eigen::EigenSolver<Eigen::MatrixXd> solver(E, false);
            volatile Eigen::Index sink = solver.eigenvalues().size();
            (void)sink;
        }, repeats);

        double threshold = eigen_ms * 3.0;
        double ratio = total_pla_ms / eigen_ms;
        bool passed = total_pla_ms <= threshold;

        out << n << ","
            << total_pla_ms << ","
            << eigen_ms << ","
            << threshold << ","
            << ratio << ","
            << (passed ? 1 : 0) << ","
            << max_error << ","
            << (pla_once.converged ? 1 : 0)
            << "\n";

        std::cout << "n=" << n
                  << " PLA=" << total_pla_ms << " ms"
                  << " Eigen=" << eigen_ms << " ms"
                  << " ratio=" << ratio
                  << " max_error=" << max_error
                  << " converged=" << pla_once.converged
                  << " passed=" << passed
                  << "\n";
    }

    std::cout << "Saved: eigen_benchmark.csv\n";
    return 0;
}