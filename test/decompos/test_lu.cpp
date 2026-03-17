#include <gtest/gtest.h>
#include <Eigen/Dense>
#include <random>
#include <chrono>
#include <iostream>
#include <vector>

#include "pla.h"

using namespace std::chrono;


static inline std::chrono::high_resolution_clock::time_point get_current_time_fenced() {
    std::atomic_thread_fence(std::memory_order_seq_cst);
    auto res_time = std::chrono::high_resolution_clock::now();
    std::atomic_thread_fence(std::memory_order_seq_cst);
    return res_time;
}

static void fill_random_pla(pla::Matrix<>& M, std::mt19937& rng, std::uniform_real_distribution<double>& dist) {
    for (int i = 0; i < M.rows(); ++i)
        for (int j = 0; j < M.cols(); ++j)
            M(i, j) = dist(rng);
}

static void sync_to_eigen(const pla::Matrix<>& src, Eigen::MatrixXd& dst) {
    for (int i = 0; i < src.rows(); ++i)
        for (int j = 0; j < src.cols(); ++j)
            dst(i, j) = src(i, j);
}


// ─── helpers ──────────────────────────────────────────────────────────────────

static void check_lu_reconstruction(const pla::Matrix<>& A,
                                     const pla::LUResult<double>& res,
                                     double tol = 1e-6) {
    pla::Matrix<> LU = res.L * res.U;
    for (int i = 0; i < A.rows(); ++i)
        for (int j = 0; j < A.cols(); ++j)
            EXPECT_NEAR(LU(i, j), A(i, j), tol)
                << "mismatch at (" << i << "," << j << ")";
}

static void check_L_shape(const pla::Matrix<>& L) {
    for (int i = 0; i < L.rows(); ++i) {
        EXPECT_NEAR(L(i, i), 1.0, 1e-12) << "L diagonal at " << i;
        for (int j = i + 1; j < L.cols(); ++j)
            EXPECT_NEAR(L(i, j), 0.0, 1e-12) << "L upper at (" << i << "," << j << ")";
    }
}

static void check_U_shape(const pla::Matrix<>& U) {
    for (int i = 0; i < U.rows(); ++i)
        for (int j = 0; j < i; ++j)
            EXPECT_NEAR(U(i, j), 0.0, 1e-12) << "U lower at (" << i << "," << j << ")";
}


// ─── correctness: naive ───────────────────────────────────────────────────────

TEST(LUNaive, CorrectnessSmall) {
    std::mt19937 rng(1111);
    std::uniform_real_distribution<double> dist(-5.0, 5.0);

    const int n = 6;
    pla::Matrix<> A(n, n);
    fill_random_pla(A, rng, dist);

    auto res = pla::lu_naive(A);

    check_L_shape(res.L);
    check_U_shape(res.U);
    check_lu_reconstruction(A, res);
}

TEST(LUNaive, CorrectnessIdentity) {
    const int n = 8;
    pla::Matrix<> A = pla::Matrix<>::identity(n);

    auto res = pla::lu_naive(A);

    check_L_shape(res.L);
    check_U_shape(res.U);
    check_lu_reconstruction(A, res, 1e-12);
}

TEST(LUNaive, CorrectnessMatchesEigen) {
    std::mt19937 rng(2222);
    std::uniform_real_distribution<double> dist(-10.0, 10.0);

    for (int n : {4, 8, 16, 32, 64}) {
        pla::Matrix<> A(n, n);
        fill_random_pla(A, rng, dist);

        Eigen::MatrixXd Ae(n, n);
        sync_to_eigen(A, Ae);

        auto res = pla::lu_naive(A);
        Eigen::PartialPivLU<Eigen::MatrixXd> lu_eigen(Ae);

        check_lu_reconstruction(A, res);

        double det_pla = 1.0;
        for (int i = 0; i < n; ++i) det_pla *= res.U(i, i);

        double det_eigen = lu_eigen.determinant();

        EXPECT_NEAR(std::abs(det_pla), std::abs(det_eigen), std::abs(det_eigen) * 1e-6)
            << "determinant mismatch for n=" << n;
    }
}

TEST(LUNaive, ThrowsOnSingular) {
    const int n = 4;
    pla::Matrix<> A(n, n, 0.0);
    EXPECT_THROW(pla::lu_naive(A), std::runtime_error);
}

TEST(LUNaive, ThrowsOnNonSquare) {
    pla::Matrix<> A(3, 5);
    EXPECT_THROW(pla::lu_naive(A), std::invalid_argument);
}


// ─── correctness: blocked ─────────────────────────────────────────────────────

TEST(LUBlocked, CorrectnessSmall) {
    std::mt19937 rng(3333);
    std::uniform_real_distribution<double> dist(-5.0, 5.0);

    const int n = 8;
    pla::Matrix<> A(n, n);
    fill_random_pla(A, rng, dist);

    auto res = pla::lu_blocked(A, 4);

    check_L_shape(res.L);
    check_U_shape(res.U);
    check_lu_reconstruction(A, res);
}

TEST(LUBlocked, CorrectnessVaryingBlockSizes) {
    std::mt19937 rng(4444);
    std::uniform_real_distribution<double> dist(-5.0, 5.0);

    const int n = 64;
    pla::Matrix<> A(n, n);
    fill_random_pla(A, rng, dist);

    for (int bs : {4, 8, 16, 32, 64}) {
        auto res = pla::lu_blocked(A, bs);
        check_L_shape(res.L);
        check_U_shape(res.U);
        check_lu_reconstruction(A, res, 1e-7);
    }
}

TEST(LUBlocked, CorrectnessMatchesNaive) {
    std::mt19937 rng(5555);
    std::uniform_real_distribution<double> dist(-5.0, 5.0);

    for (int n : {5, 16, 33, 64, 128}) {
        pla::Matrix<> A(n, n);
        fill_random_pla(A, rng, dist);

        auto res_naive   = pla::lu_naive(A);
        auto res_blocked = pla::lu_blocked(A);

        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j) {
                EXPECT_NEAR(res_naive.L(i, j), res_blocked.L(i, j), 1e-6)
                    << "L mismatch at (" << i << "," << j << ") n=" << n;
                EXPECT_NEAR(res_naive.U(i, j), res_blocked.U(i, j), 1e-6)
                    << "U mismatch at (" << i << "," << j << ") n=" << n;
            }
    }
}

TEST(LUBlocked, CorrectnessMatchesEigen) {
    std::mt19937 rng(6666);
    std::uniform_real_distribution<double> dist(-10.0, 10.0);

    for (int n : {4, 16, 64, 128}) {
        pla::Matrix<> A(n, n);
        fill_random_pla(A, rng, dist);

        Eigen::MatrixXd Ae(n, n);
        sync_to_eigen(A, Ae);

        auto res = pla::lu_blocked(A);
        Eigen::PartialPivLU<Eigen::MatrixXd> lu_eigen(Ae);

        check_lu_reconstruction(A, res);

        double det_pla = 1.0;
        for (int i = 0; i < n; ++i) det_pla *= res.U(i, i);

        double det_eigen = lu_eigen.determinant();

        EXPECT_NEAR(std::abs(det_pla), std::abs(det_eigen), std::abs(det_eigen) * 1e-5)
            << "determinant mismatch for n=" << n;
    }
}

TEST(LUBlocked, ThrowsOnSingular) {
    const int n = 4;
    pla::Matrix<> A(n, n, 0.0);
    EXPECT_THROW(pla::lu_blocked(A), std::runtime_error);
}

TEST(LUBlocked, ThrowsOnNonSquare) {
    pla::Matrix<> A(3, 5);
    EXPECT_THROW(pla::lu_blocked(A), std::invalid_argument);
}


// ─── performance ──────────────────────────────────────────────────────────────

TEST(LUPerformance, CompareEigen) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    std::vector<int> sizes = {32, 64, 128, 256, 512, 1024};
    const int repeats = 4;

    for (int n : sizes) {
        pla::Matrix<> A(n, n);
        fill_random_pla(A, rng, dist);

        Eigen::MatrixXd Ae(n, n);
        sync_to_eigen(A, Ae);

        // warmup
        pla::lu_blocked(A);
        { Eigen::PartialPivLU<Eigen::MatrixXd> tmp(Ae); }

        double total_naive = 0.0;
        for (int r = 0; r < repeats; ++r) {
            auto t0 = get_current_time_fenced();
            pla::lu_naive(A);
            auto t1 = get_current_time_fenced();
            total_naive += duration_cast<duration<double>>(t1 - t0).count();
        }
        double avg_naive = total_naive / repeats;

        double total_blocked = 0.0;
        for (int r = 0; r < repeats; ++r) {
            auto t0 = get_current_time_fenced();
            pla::lu_blocked(A);
            auto t1 = get_current_time_fenced();
            total_blocked += duration_cast<duration<double>>(t1 - t0).count();
        }
        double avg_blocked = total_blocked / repeats;

        double total_eigen = 0.0;
        for (int r = 0; r < repeats; ++r) {
            auto t0 = high_resolution_clock::now();
            Eigen::PartialPivLU<Eigen::MatrixXd> lu_eigen(Ae);
            auto t1 = high_resolution_clock::now();
            total_eigen += duration_cast<duration<double>>(t1 - t0).count();
        }
        double avg_eigen = total_eigen / repeats;

        std::cout << "n=" << n
                  << ": naive avg=" << avg_naive << "s"
                  << ", blocked avg=" << avg_blocked << "s"
                  << ", Eigen avg=" << avg_eigen << "s"
                  << ", ratio_naive=" << (avg_naive / avg_eigen)
                  << ", ratio_blocked=" << (avg_blocked / avg_eigen)
                  << std::endl;
    }

    SUCCEED();
}


int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}