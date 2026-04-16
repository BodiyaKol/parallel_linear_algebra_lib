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

static void check_Q_orthogonal(const pla::Matrix<>& Q, double tol = 1e-6) {
    pla::Matrix<> QtQ = Q.transpose() * Q;
    for (int i = 0; i < QtQ.rows(); ++i)
        for (int j = 0; j < QtQ.cols(); ++j) {
            const double expected = (i == j) ? 1.0 : 0.0;
            EXPECT_NEAR(QtQ(i, j), expected, tol)
                << "Q^T*Q mismatch at (" << i << "," << j << ")";
        }
}

static void check_R_upper_triangular(const pla::Matrix<>& R, double tol = 1e-10) {
    for (int i = 0; i < R.rows(); ++i)
        for (int j = 0; j < i && j < R.cols(); ++j)
            EXPECT_NEAR(R(i, j), 0.0, tol)
                << "R subdiagonal nonzero at (" << i << "," << j << ")";
}

static void check_qr_reconstruction(const pla::Matrix<>& A,
                                    const pla::QRResult<double>& res,
                                    double tol = 1e-6) {
    pla::Matrix<> QR = res.Q * res.R;
    for (int i = 0; i < A.rows(); ++i)
        for (int j = 0; j < A.cols(); ++j)
            EXPECT_NEAR(QR(i, j), A(i, j), tol)
                << "QR != A at (" << i << "," << j << ")";
}

static void check_output_dimensions(const pla::QRResult<double>& res, int m, int n) {
    EXPECT_EQ(res.Q.rows(), m);
    EXPECT_EQ(res.Q.cols(), m);
    EXPECT_EQ(res.R.rows(), m);
    EXPECT_EQ(res.R.cols(), n);
}


TEST(QRHouseholder, CorrectnessSmallSquare) {
    std::mt19937 rng(1111);
    std::uniform_real_distribution<double> dist(-5.0, 5.0);

    const int n = 6;
    pla::Matrix<> A(n, n);
    fill_random_pla(A, rng, dist);

    auto res = pla::qr_householder(A);

    check_output_dimensions(res, n, n);
    check_Q_orthogonal(res.Q);
    check_R_upper_triangular(res.R);
    check_qr_reconstruction(A, res);
}

TEST(QRHouseholder, CorrectnessTallMatrix) {
    std::mt19937 rng(2222);
    std::uniform_real_distribution<double> dist(-10.0, 10.0);

    const int m = 10;
    const int n = 4;
    pla::Matrix<> A(m, n);
    fill_random_pla(A, rng, dist);

    auto res = pla::qr_householder(A);

    check_output_dimensions(res, m, n);
    check_Q_orthogonal(res.Q);
    check_R_upper_triangular(res.R);
    check_qr_reconstruction(A, res);
}

TEST(QRHouseholder, CorrectnessWideMatrix) {
    std::mt19937 rng(3333);
    std::uniform_real_distribution<double> dist(-10.0, 10.0);

    const int m = 4;
    const int n = 9;
    pla::Matrix<> A(m, n);
    fill_random_pla(A, rng, dist);

    auto res = pla::qr_householder(A);

    check_output_dimensions(res, m, n);
    check_Q_orthogonal(res.Q);
    check_R_upper_triangular(res.R);
    check_qr_reconstruction(A, res);
}

TEST(QRHouseholder, CorrectnessIdentity) {
    const int n = 8;
    pla::Matrix<> A = pla::Matrix<>::identity(n);

    auto res = pla::qr_householder(A);

    check_output_dimensions(res, n, n);
    check_Q_orthogonal(res.Q, 1e-12);
    check_R_upper_triangular(res.R, 1e-12);
    check_qr_reconstruction(A, res, 1e-12);
}

TEST(QRHouseholder, CorrectnessAcrossSizes) {
    std::mt19937 rng(4444);
    std::uniform_real_distribution<double> dist(-5.0, 5.0);

    for (int m : {4, 8, 16, 32, 48, 64, 96, 128}) {
        for (int n : {4, 8, 16, 32}) {
            if (n > m) continue;
            pla::Matrix<> A(m, n);
            fill_random_pla(A, rng, dist);

            auto res = pla::qr_householder(A);

            check_output_dimensions(res, m, n);
            check_Q_orthogonal(res.Q, 1e-6);
            check_R_upper_triangular(res.R, 1e-10);
            check_qr_reconstruction(A, res, 1e-6);
        }
    }
}

TEST(QRHouseholder, CorrectnessAtBlockBoundaries) {
    std::mt19937 rng(5555);
    std::uniform_real_distribution<double> dist(-5.0, 5.0);

    // QR_BLOCK = 48, test sizes that cross block boundaries
    for (int m : {47, 48, 49, 95, 96, 97}) {
        for (int n : {47, 48, 49}) {
            if (n > m) continue;
            pla::Matrix<> A(m, n);
            fill_random_pla(A, rng, dist);

            auto res = pla::qr_householder(A);

            check_output_dimensions(res, m, n);
            check_Q_orthogonal(res.Q, 1e-5);
            check_R_upper_triangular(res.R, 1e-9);
            check_qr_reconstruction(A, res, 1e-5);
        }
    }
}

TEST(QRHouseholder, CorrectnessMatchesEigen) {
    std::mt19937 rng(6666);
    std::uniform_real_distribution<double> dist(-10.0, 10.0);

    for (int m : {5, 16, 33, 64, 128}) {
        for (int n : {4, 8, 16}) {
            if (n > m) continue;
            pla::Matrix<> A(m, n);
            fill_random_pla(A, rng, dist);

            Eigen::MatrixXd Ae(m, n);
            sync_to_eigen(A, Ae);

            auto res = pla::qr_householder(A);
            Eigen::HouseholderQR<Eigen::MatrixXd> qr_eigen(Ae);

            Eigen::MatrixXd Qe = qr_eigen.householderQ() * Eigen::MatrixXd::Identity(m, m);
            Eigen::MatrixXd Re = qr_eigen.matrixQR().template triangularView<Eigen::Upper>();
            Eigen::MatrixXd QR_eigen = Qe * Re;

            check_Q_orthogonal(res.Q);
            check_R_upper_triangular(res.R);
            check_qr_reconstruction(A, res);

            for (int i = 0; i < m; ++i)
                for (int j = 0; j < n; ++j)
                    EXPECT_NEAR((res.Q * res.R)(i, j), QR_eigen(i, j), 1e-6)
                        << "QR vs Eigen mismatch at (" << i << "," << j << ") m=" << m << " n=" << n;
        }
    }
}

TEST(QRHouseholder, ThrowsOnEmpty) {
    pla::Matrix<> A(0, 0);
    EXPECT_THROW(pla::qr_householder(A), std::invalid_argument);
}

TEST(QRHouseholder, ThrowsOnZeroRows) {
    pla::Matrix<> A(0, 4);
    EXPECT_THROW(pla::qr_householder(A), std::invalid_argument);
}

TEST(QRHouseholder, ThrowsOnZeroCols) {
    pla::Matrix<> A(4, 0);
    EXPECT_THROW(pla::qr_householder(A), std::invalid_argument);
}

TEST(QRFunction, AliasQR) {
    std::mt19937 rng(7777);
    std::uniform_real_distribution<double> dist(-5.0, 5.0);

    const int m = 8;
    const int n = 5;
    pla::Matrix<> A(m, n);
    fill_random_pla(A, rng, dist);

    auto res1 = pla::qr_householder(A);
    auto res2 = pla::qr(A);

    check_qr_reconstruction(A, res1);
    check_qr_reconstruction(A, res2);

    for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j)
            EXPECT_NEAR((res1.Q * res1.R)(i, j), (res2.Q * res2.R)(i, j), 1e-12)
                << "qr_householder vs qr mismatch at (" << i << "," << j << ")";
}

TEST(QRFunction, AliasGivens) {
    std::mt19937 rng(8888);
    std::uniform_real_distribution<double> dist(-5.0, 5.0);

    const int m = 8;
    const int n = 5;
    pla::Matrix<> A(m, n);
    fill_random_pla(A, rng, dist);

    auto res1 = pla::qr_householder(A);
    auto res2 = pla::qr_givens(A);

    check_qr_reconstruction(A, res1);
    check_qr_reconstruction(A, res2);
}

TEST(QRPerformance, CompareEigen) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    std::vector<std::pair<int,int>> sizes = {
        {32, 32}, {64, 64}, {128, 64}, {256, 128},
        {511, 255}, {511, 511}, {1023, 255}, {1023, 1023}
    };
    const int repeats = 4;

    for (auto [m, n] : sizes) {
        pla::Matrix<> A(m, n);
        fill_random_pla(A, rng, dist);

        Eigen::MatrixXd Ae(m, n);
        sync_to_eigen(A, Ae);

        pla::qr_householder(A);
        { Eigen::HouseholderQR<Eigen::MatrixXd> tmp(Ae); }

        double total_pla = 0.0;
        for (int r = 0; r < repeats; ++r) {
            auto t0 = get_current_time_fenced();
            pla::qr_householder(A);
            auto t1 = get_current_time_fenced();
            total_pla += duration_cast<duration<double>>(t1 - t0).count();
        }
        double avg_pla = total_pla / repeats;

        double total_eigen = 0.0;
        for (int r = 0; r < repeats; ++r) {
            auto t0 = get_current_time_fenced();
            Eigen::HouseholderQR<Eigen::MatrixXd> qr_eigen(Ae);
            auto t1 = get_current_time_fenced();
            total_eigen += duration_cast<duration<double>>(t1 - t0).count();
        }
        double avg_eigen = total_eigen / repeats;

        std::cout << "m=" << m << " n=" << n
                  << ": pla avg=" << avg_pla << "s"
                  << ", eigen avg=" << avg_eigen << "s"
                  << ", ratio=" << (avg_pla / avg_eigen)
                  << std::endl;
    }

    SUCCEED();
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
