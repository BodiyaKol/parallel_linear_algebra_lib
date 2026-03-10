// Тести для базових типів бібліотеки (Vector, Matrix)
// Запускай ці тести після реалізації методів у src/types

#include <cassert>
#include <cmath>
#include <iostream>

#include "../../include/pla.h"

namespace {

// Допоміжна функція для порівняння чисел з плаваючою комою
template<typename Scalar>
bool approx_equal(Scalar a, Scalar b, Scalar eps = 1e-9) {
    return std::abs(a - b) < eps;
}

// ===========================================================================
// БАЗОВІ ТЕСТИ (працюють одразу - конструктори вже реалізовані в хедерах)
// ===========================================================================

void test_vector_basics() {
    std::cout << "Тест: базові операції Vector... ";
    
    pla::Vector v{1.0, 2.0, 3.0};
    assert(v.dimension() == 3);
    assert(std::abs(v[0] - 1.0) < 1e-12);

    v[1] = 5.0;
    assert(std::abs(v[1] - 5.0) < 1e-12);
    
    pla::Vector v2(4, 2.5);
    assert(v2.dimension() == 4);
    assert(v2[0] == 2.5);
    
    std::cout << "OK" << std::endl;
}

void test_matrix_basics() {
    std::cout << "Тест: базові операції Matrix... ";
    
    pla::Matrix m(2, 3, pla::StorageOrder::RowMajor);
    assert(m.rows() == 2);
    assert(m.cols() == 3);

    m(0, 0) = 10.0;
    m(1, 2) = 7.0;

    assert(std::abs(m(0, 0) - 10.0) < 1e-12);
    assert(std::abs(m(1, 2) - 7.0) < 1e-12);
    
    std::cout << "OK" << std::endl;
}

void test_shape_and_status() {
    std::cout << "Тест: Shape та Status... ";
    
    pla::VectorShape vs{4};
    pla::MatrixShape ms{4, 4};

    assert(vs.valid());
    assert(ms.valid());

    pla::Status ok = pla::Status::success();
    assert(ok.ok());
    assert(ok.code == pla::StatusCode::Ok);
    
    std::cout << "OK" << std::endl;
}

void test_execution_policy_defaults() {
    std::cout << "Тест: ExecutionPolicy... ";
    
    pla::ExecutionPolicy policy;
    assert(policy.backend == pla::Backend::Serial);
    assert(policy.thread_count == 0);
    assert(policy.block_size == 0);
    
    std::cout << "OK" << std::endl;
}

// ===========================================================================
// ТЕСТИ ДЛЯ ОПЕРАТОРІВ І МЕТОДІВ (розкоментуй після реалізації в src/types)
// ===========================================================================

/*
void test_vector_operators_add() {
    std::cout << "Тест: Vector::operator+... ";
    
    pla::Vector a = {1.0, 2.0, 3.0};
    pla::Vector b = {4.0, 5.0, 6.0};
    
    pla::Vector c = a + b;
    
    assert(c.dimension() == 3);
    assert(c[0] == 5.0);
    assert(c[1] == 7.0);
    assert(c[2] == 9.0);
    
    std::cout << "OK" << std::endl;
}

void test_vector_operators_multiply() {
    std::cout << "Тест: Vector::operator*... ";
    
    pla::Vector a = {1.0, 2.0, 3.0};
    
    // Множення на скаляр справа
    pla::Vector b = a * 2.0;
    assert(b[0] == 2.0);
    assert(b[1] == 4.0);
    assert(b[2] == 6.0);
    
    // Множення на скаляр зліва (оператор поза класом)
    pla::Vector c = 0.5 * a;
    assert(c[0] == 0.5);
    assert(c[1] == 1.0);
    assert(c[2] == 1.5);
    
    std::cout << "OK" << std::endl;
}

void test_vector_dot() {
    std::cout << "Тест: Vector::dot... ";
    
    pla::Vector a = {1.0, 2.0, 3.0};
    pla::Vector b = {4.0, 5.0, 6.0};
    
    pla::Scalar result = a.dot(b);
    assert(approx_equal(result, 32.0)); // 1*4 + 2*5 + 3*6 = 32
    
    std::cout << "OK" << std::endl;
}

void test_vector_norm() {
    std::cout << "Тест: Vector::norm... ";
    
    pla::Vector v = {3.0, 4.0};
    pla::Scalar n = v.norm();
    assert(approx_equal(n, 5.0));
    
    std::cout << "OK" << std::endl;
}

void test_matrix_identity_static() {
    std::cout << "Тест: Matrix::identity... ";
    
    pla::Matrix I = pla::Matrix::identity(3);
    
    assert(I(0, 0) == 1.0);
    assert(I(1, 1) == 1.0);
    assert(I(2, 2) == 1.0);
    assert(I(0, 1) == 0.0);
    assert(I(1, 2) == 0.0);
    
    std::cout << "OK" << std::endl;
}

void test_matrix_transpose() {
    std::cout << "Тест: Matrix::transpose... ";
    
    pla::Matrix m(2, 3);
    m(0, 0) = 1.0; m(0, 1) = 2.0; m(0, 2) = 3.0;
    m(1, 0) = 4.0; m(1, 1) = 5.0; m(1, 2) = 6.0;
    
    pla::Matrix mt = m.transpose();
    
    assert(mt.rows() == 3);
    assert(mt.cols() == 2);
    assert(mt(0, 0) == 1.0);
    assert(mt(0, 1) == 4.0);
    assert(mt(1, 0) == 2.0);
    
    std::cout << "OK" << std::endl;
}

void test_matrix_vector_multiply() {
    std::cout << "Тест: Matrix::operator*(Vector)... ";
    
    pla::Matrix A(2, 3);
    A(0, 0) = 1.0; A(0, 1) = 2.0; A(0, 2) = 3.0;
    A(1, 0) = 4.0; A(1, 1) = 5.0; A(1, 2) = 6.0;
    
    pla::Vector x = {1.0, 2.0, 3.0};
    pla::Vector y = A * x;
    
    assert(y.dimension() == 2);
    assert(approx_equal(y[0], 14.0)); // 1*1 + 2*2 + 3*3 = 14
    assert(approx_equal(y[1], 32.0)); // 4*1 + 5*2 + 6*3 = 32
    
    std::cout << "OK" << std::endl;
}

void test_matrix_multiply() {
    std::cout << "Тест: Matrix::operator*(Matrix) [2x3 * 3x2]... ";
    
    pla::Matrix A(2, 3);
    A(0, 0) = 1.0; A(0, 1) = 2.0; A(0, 2) = 3.0;
    A(1, 0) = 4.0; A(1, 1) = 5.0; A(1, 2) = 6.0;
    
    pla::Matrix B(3, 2);
    B(0, 0) = 7.0;  B(0, 1) = 8.0;
    B(1, 0) = 9.0;  B(1, 1) = 10.0;
    B(2, 0) = 11.0; B(2, 1) = 12.0;
    
    pla::Matrix C = A * B;
    
    assert(C.rows() == 2);
    assert(C.cols() == 2);
    assert(approx_equal(C(0, 0), 58.0));  // 1*7 + 2*9 + 3*11
    assert(approx_equal(C(0, 1), 64.0));  // 1*8 + 2*10 + 3*12
    assert(approx_equal(C(1, 0), 139.0)); // 4*7 + 5*9 + 6*11
    assert(approx_equal(C(1, 1), 154.0)); // 4*8 + 5*10 + 6*12
    
    std::cout << "OK" << std::endl;
}
*/

} // namespace

int main() {
    std::cout << "\n========================================" << std::endl;
    std::cout << "   ТЕСТИ ТИПІВ БІБЛІОТЕКИ PLA" << std::endl;
    std::cout << "========================================\n" << std::endl;
    
    std::cout << "--- Базові тести (працюють одразу) ---" << std::endl;
    test_vector_basics();
    test_matrix_basics();
    test_shape_and_status();
    test_execution_policy_defaults();
    
    std::cout << "\n--- Тести операторів і методів (розкоментуй після реалізації) ---" << std::endl;
    std::cout << "  Після реалізації операторів у src/types розкоментуй тести вище" << std::endl;
    // test_vector_operators_add();
    // test_vector_operators_multiply();
    // test_vector_dot();
    // test_vector_norm();
    // test_matrix_identity_static();
    // test_matrix_transpose();
    // test_matrix_vector_multiply();
    // test_matrix_multiply();

    std::cout << "\n========================================" << std::endl;
    std::cout << "   ВСІ ТЕСТИ ПРОЙДЕНО!" << std::endl;
    std::cout << "========================================\n" << std::endl;
    
    return 0;
}
