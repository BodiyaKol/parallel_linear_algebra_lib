#include <gtest/gtest.h>
#include <Eigen/Dense>
#include <random>
#include <chrono>
#include <iostream>
#include <vector>

#include "../../../include/main_api.h"

using namespace std::chrono;

static inline std::chrono::high_resolution_clock::time_point get_current_time_fenced() {
    std::atomic_thread_fence(std::memory_order_seq_cst);
    auto res_time = std::chrono::high_resolution_clock::now();
    std::atomic_thread_fence(std::memory_order_seq_cst);
    return res_time;
}

static void fill_random_pla(pla::Matrix& M, std::mt19937& rng, std::uniform_real_distribution<double>& dist) {
    for (int i = 0; i < M.rows(); ++i)
        for (int j = 0; j < M.cols(); ++j)
            M(i, j) = dist(rng);
}

static void sync_to_eigen(const pla::Matrix& src, Eigen::MatrixXd& dst) {
    for (int i = 0; i < src.rows(); ++i)
        for (int j = 0; j < src.cols(); ++j)
            dst(i, j) = src(i, j);
}

TEST(MatrixMultiplication, CorrectnessMatchesEigenSmall) {
    std::mt19937 rng(12345);
    std::uniform_real_distribution<double> dist(-10.0, 10.0);

    const int M = 5, K = 7, N = 4;

    pla::Matrix A(M, K);
    pla::Matrix B(K, N);
    Eigen::MatrixXd Ae(M, K);
    Eigen::MatrixXd Be(K, N);

    fill_random_pla(A, rng, dist);
    fill_random_pla(B, rng, dist);
    
    sync_to_eigen(A, Ae);
    sync_to_eigen(B, Be);

    pla::Matrix C = A * B;
    Eigen::MatrixXd Ce = Ae * Be;

    ASSERT_EQ(C.rows(), Ce.rows());
    ASSERT_EQ(C.cols(), Ce.cols());

    for (int i = 0; i < C.rows(); ++i) {
        for (int j = 0; j < C.cols(); ++j) {
            EXPECT_NEAR(C(i, j), Ce(i, j), 1e-9);
        }
    }
}

TEST(MatrixMultiplication, PerformanceCompareEigen) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    std::vector<int> sizes = {4, 8, 16, 32, 64, 96, 128, 256, 512, 1024};
    const int repeats = 4;

    for (int n : sizes) {
        int M = n, K = n, N = n;

        pla::Matrix A(M, K);
        pla::Matrix B(K, N);
        Eigen::MatrixXd Ae(M, K);
        Eigen::MatrixXd Be(K, N);

        fill_random_pla(A, rng, dist);
        fill_random_pla(B, rng, dist);
        
        sync_to_eigen(A, Ae);
        sync_to_eigen(B, Be);

        pla::Matrix Cw = A * B;
        Eigen::MatrixXd Cew = Ae * Be;

        double total_pla = 0.0;
        for (int r = 0; r < repeats; ++r) {
            auto t0 = get_current_time_fenced();
            pla::Matrix C = A * B;
            auto t1 = get_current_time_fenced();
            total_pla += duration_cast<duration<double>>(t1 - t0).count();
        }
        double avg_pla = total_pla / repeats;

        // Time Eigen
        double total_eigen = 0.0;
        for (int r = 0; r < repeats; ++r) {
            auto t0 = high_resolution_clock::now();
            Eigen::MatrixXd C = Ae * Be;
            auto t1 = high_resolution_clock::now();
            total_eigen += duration_cast<duration<double>>(t1 - t0).count();
        }
        double avg_eigen = total_eigen / repeats;

        std::cout << "n=" << n << ": PLA avg=" << avg_pla << "s, Eigen avg=" << avg_eigen << "s" << std::endl;

        // Verify correctness
        pla::Matrix Cres = A * B;
        Eigen::MatrixXd Ceres = Ae * Be;
        for (int i = 0; i < Cres.rows(); ++i)
            for (int j = 0; j < Cres.cols(); ++j)
                EXPECT_NEAR(Cres(i, j), Ceres(i, j), 1e-8);
    }

    SUCCEED();
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}