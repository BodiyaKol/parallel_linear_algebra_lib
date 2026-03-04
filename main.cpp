#include "matrix.h"
#include <iostream>

int main() {
    std::cout << "PLA Matrix class basic test...\n";

    pla::Matrix A(2, 3);
    A(0, 0) = 1.0; A(0, 1) = 2.0; A(0, 2) = 3.0;
    A(1, 0) = 4.0; A(1, 1) = 5.0; A(1, 2) = 6.0;

    pla::Matrix B(3, 2);
    B(0, 0) = 7.0;  B(0, 1) = 8.0;
    B(1, 0) = 9.0;  B(1, 1) = 10.0;
    B(2, 0) = 11.0; B(2, 1) = 12.0;

    pla::Matrix C = A * B;

    std::cout << C(0, 0) << " " << C(0, 1) << "\n";
    std::cout << C(1, 0) << " " << C(1, 1) << "\n";

    return 0;
}