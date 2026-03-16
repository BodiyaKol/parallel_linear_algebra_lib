#include <gtest/gtest.h>
#include <Eigen/Dense>

#include <random>
#include <chrono>
#include <iostream>
#include <vector>
#include <array>
#include <atomic>

#include "pla.h"

using namespace std::chrono;

struct MatrixShape {
    int M;
    int K;
    int N;
};

static inline high_resolution_clock::time_point get_current_time_fenced() {
    std::atomic_thread_fence(std::memory_order_seq_cst);
    auto t = high_resolution_clock::now();
    std::atomic_thread_fence(std::memory_order_seq_cst);
    return t;
}

static void fill_random_pla(
    pla::Matrix<>& M,
    std::mt19937& rng,
    std::uniform_real_distribution<double>& dist)
{
    for (int i = 0; i < M.rows(); ++i)
        for (int j = 0; j < M.cols(); ++j)
            M(i, j) = dist(rng);
}

static void sync_to_eigen(const pla::Matrix<>& src, Eigen::MatrixXd& dst) {
    for (int i = 0; i < src.rows(); ++i)
        for (int j = 0; j < src.cols(); ++j)
            dst(i, j) = src(i, j);
}

template<class F>
double benchmark(F&& func, int repeats) {
    double total = 0.0;

    for (int i = 0; i < repeats; ++i) {
        auto t0 = get_current_time_fenced();
        func();
        auto t1 = get_current_time_fenced();

        total += duration<double>(t1 - t0).count();
    }

    return total / repeats;
}

static std::vector<MatrixShape> generate_shapes() {

    constexpr std::array dims{
        4, 8, 16, 32, 64, 128, 256, 511
    };

    std::vector<MatrixShape> shapes;

    for (auto M : dims)
        for (auto K : dims)
            for (auto N : dims)
                shapes.push_back({M, K, N});

    return shapes;
}

TEST(MatrixMultiplicationPerformance, RectangularSweepCompareWithEigen)
{
    constexpr int repeats = 3;

    std::mt19937 rng(123);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    for (auto [M, K, N] : generate_shapes()) {

        pla::Matrix A(M, K);
        pla::Matrix B(K, N);

        Eigen::MatrixXd Ae(M, K);
        Eigen::MatrixXd Be(K, N);

        fill_random_pla(A, rng, dist);
        fill_random_pla(B, rng, dist);

        sync_to_eigen(A, Ae);
        sync_to_eigen(B, Be);

        // warm-up
        auto Cw = A * B;
        auto Cew = Ae * Be;

        double pla_time = benchmark([&] {
            auto C = A * B;
            volatile double sink = C(0,0);
        }, repeats);

        double eigen_time = benchmark([&] {
            auto Cе = Ae * Be;
            volatile double sink = Cе(0,0);
        }, repeats);

        std::cout
            << "M=" << M
            << " K=" << K
            << " N=" << N
            << " | PLA=" << pla_time
            << "s Eigen=" << eigen_time
            << "s ratio=" << (pla_time / eigen_time)
            << std::endl;

        auto C = A * B;
        auto Ce = Ae * Be;

        EXPECT_NEAR(C(0,0), Ce(0,0), 1e-8);
    }
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}