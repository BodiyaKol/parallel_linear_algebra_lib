#include <iostream>
#include <pla/pla.h>

using pla::Vector;
using pla::Matrix;

int main() {

    { // MATRIX

        Matrix<double> A(2, 2);
        A(0, 0) = 1.0; A(0, 1) = 2.0;
        A(1, 0) = 3.0; A(1, 1) = 4.0;

        Matrix<double> B(2, 2, 1.0); // fill with 1.0

        std::cout << "\nMatrix A:\n" << A << "\n";
        std::cout << "Matrix B:\n" << B << "\n";

        auto C = A + B;
        auto D = A - B;
        auto E = A * 2.0;

        std::cout << "A + B:\n" << C << "\n";
        std::cout << "A - B:\n" << D << "\n";
        std::cout << "A * 2:\n" << E << "\n";

        // MATRIX * VECTOR

        Vector<double> v{1.0, 1.0};

        auto result = A * v;

        std::cout << "\nA * v = " << result << "\n";

        // MATRIX * MATRIX

        auto M = A * B;

        std::cout << "\nA * B:\n" << M << "\n";

        // TRANSPOSE

        auto At = A.transpose();

        std::cout << "\ntranspose(A):\n" << At << "\n";

        // ROW / COL ACCESS

        std::cout << "row 0: " << A.row(0) << "\n";
        std::cout << "col 1: " << A.col(1) << "\n";
    
    }

    { // INPUT MATRIX

        Matrix<double> A(3, 3);

        A(0,0) = 2;  A(0,1) = -1; A(0,2) = 0;
        A(1,0) = 1;  A(1,1) =  3; A(1,2) = 2;
        A(2,0) = 2;  A(2,1) =  0; A(2,2) = 1;

        std::cout << "A:\n" << A << "\n";

        // LU DECOMPOSITION

        auto lu = pla::lu_naive(A);

        std::cout << "\n=== LU decomposition ===\n";
        std::cout << "L:\n" << lu.L << "\n";
        std::cout << "U:\n" << lu.U << "\n";

        // reconstruct A (with permutation if you have it)
        Matrix<double> LU = lu.L * lu.U;

        std::cout << "L * U:\n" << LU << "\n";

        // determinant via LU
        double det = 1.0;
        for (int i = 0; i < lu.U.rows(); ++i)
            det *= lu.U(i, i);

        std::cout << "det(A) ≈ " << det << "\n";

        // BLOCKED LU

        auto lu2 = pla::lu_blocked(A, 2);

        std::cout << "\n=== Blocked LU ===\n";
        std::cout << "L:\n" << lu2.L << "\n";
        std::cout << "U:\n" << lu2.U << "\n";

        // QR DECOMPOSITION

        auto qr = pla::qr_householder(A);

        std::cout << "\n=== QR decomposition ===\n";
        std::cout << "Q:\n" << qr.Q << "\n";
        std::cout << "R:\n" << qr.R << "\n";

        // reconstruction check
        Matrix<double> QR = qr.Q * qr.R;

        std::cout << "Q * R:\n" << QR << "\n";

        // ORTHOGONALITY CHECK

        auto QtQ = qr.Q.transpose() * qr.Q;

        std::cout << "\nQ^T * Q:\n" << QtQ << "\n";

    }

    return 0;
}