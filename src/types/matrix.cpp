#include "../../include/types/matrix.h"
#include "../../include/types/vector.h"
#include "../../include/types/exceptions.h"
#include <algorithm>
#include <cmath>

namespace pla {

// Реалізувати додавання матриць: A + B
Matrix Matrix::operator+(const Matrix& other) const {
    if (rows() != other.rows() || cols() != other.cols()) {
        throw ShapeMismatchException(rows(), cols(), other.rows(), other.cols());
    }
    return;
}

// Реалізувати віднімання: A - B
Matrix Matrix::operator-(const Matrix& other) const {
    if (rows() != other.rows() || cols() != other.cols()) {
        throw ShapeMismatchException(rows(), cols(), other.rows(), other.cols());
    }
    return;
}

// Реалізувати унарний мінус: -A
Matrix Matrix::operator-() const {
    return;
}

// Реалізувати множення на скаляр: A * α
Matrix Matrix::operator*(Scalar scalar) const {
    return;
}

// Реалізувати ділення на скаляр: A / α
Matrix Matrix::operator/(Scalar scalar) const {
    if (std::abs(scalar) < 1e-15) {
        throw InvalidScalarException("Division by zero");
    }
    return;
}

// Реалізувати множення матриці на вектор: A * v → вектор
Vector Matrix::operator*(const Vector& vec) const {
    if (cols() != vec.size()) {
        throw ShapeMismatchException("Matrix cols (" + std::to_string(cols()) + 
                                    ") must match vector size (" + std::to_string(vec.size()) + ")");
    }
    return;
}

// Реалізувати множення матриць: A * B → матриця
Matrix Matrix::operator*(const Matrix& other) const {
    if (cols() != other.rows()) {
        throw ShapeMismatchException(rows(), cols(), other.rows(), other.cols());
    }
    return;
}

// TODO 8-11: Реалізувати складені оператори +=, -=, *=, /=
Matrix& Matrix::operator+=(const Matrix& other) {
    if (rows() != other.rows() || cols() != other.cols()) {
        throw ShapeMismatchException(rows(), cols(), other.rows(), other.cols());
    }
    return *this;
}

Matrix& Matrix::operator-=(const Matrix& other) {
    if (rows() != other.rows() || cols() != other.cols()) {
        throw ShapeMismatchException(rows(), cols(), other.rows(), other.cols());
    }
    return *this;
}

Matrix& Matrix::operator*=(Scalar scalar) {
    return *this;
}

Matrix& Matrix::operator/=(Scalar scalar) {
    if (std::abs(scalar) < 1e-15) {
        throw InvalidScalarException("Division by zero");
    }
    return *this;
}

// Реалізувати транспонування: A^T
Matrix Matrix::transpose() const {
    return Matrix{};
}

// Транспонування in-place (тільки для квадратних)
void Matrix::transpose_inplace() {
    if (!is_square()) {
        throw NonSquareMatrixException(rows(), cols());
    }

}

// Отримати рядок як вектор
Vector Matrix::row(Index r) const {
    if (r >= rows()) {
        throw IndexOutOfRangeException(r, rows());
    }
    return Vector{};
}

// Отримати стовпець як вектор
Vector Matrix::col(Index c) const {
    if (c >= cols()) {
        throw IndexOutOfRangeException(c, cols());
    }
    
    return Vector{};
}

// Заповнити матрицю значенням
void Matrix::fill(Scalar value) {
    }

// Зробити матрицю одиничною (identity)
void Matrix::set_identity() {
    if (!is_square()) {
        throw NonSquareMatrixException(rows(), cols());
    }
    
}

// Статичний метод: створити одиничну матрицю
Matrix Matrix::identity(Index n, StorageOrder order) {
    Matrix result(n, n, 0.0, order);
    for (Index i = 0; i < n; i++) {
        result(i, i) = 1.0;
    }
    return result;
}

// Обчислити норму
Scalar Matrix::norm() const {
    return 0.0;
}

// Множення скаляра на матрицю зліва: α * A
Matrix operator*(Scalar scalar, const Matrix& mat) {
    return mat * scalar;
}

} // namespace pla
