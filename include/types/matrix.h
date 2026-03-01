#pragma once

#include <vector>

#include "index.h"
#include "layout.h"
#include "scalar.h"

namespace pla {

class Vector;

class Matrix {
public:
    Matrix() = default;

    Matrix(Index rows, Index cols, StorageOrder order = StorageOrder::RowMajor)
        : rows_(rows), cols_(cols), order_(order), data_(rows * cols) {}

    Matrix(Index rows, Index cols, Scalar value, StorageOrder order = StorageOrder::RowMajor)
        : rows_(rows), cols_(cols), order_(order), data_(rows * cols, value) {}

    [[nodiscard]] Index rows() const noexcept {
        return rows_;
    }

    [[nodiscard]] Index cols() const noexcept {
        return cols_;
    }

    [[nodiscard]] Index size() const noexcept {
        return data_.size();
    }

    [[nodiscard]] StorageOrder order() const noexcept {
        return order_;
    }

    [[nodiscard]] bool is_square() const noexcept {
        return rows_ == cols_;
    }

    [[nodiscard]] Scalar* data() noexcept {
        return data_.data();
    }

    [[nodiscard]] const Scalar* data() const noexcept {
        return data_.data();
    }

    // Індексування (рядок, стовпець)
    [[nodiscard]] Scalar& operator()(Index r, Index c) noexcept {
        return data_[offset(r, c)];
    }

    [[nodiscard]] const Scalar& operator()(Index r, Index c) const noexcept {
        return data_[offset(r, c)];
    }

    Matrix operator+(const Matrix& other) const;

    Matrix operator-(const Matrix& other) const;

    Matrix operator-() const;

    Matrix operator*(Scalar scalar) const;

    Matrix operator/(Scalar scalar) const;

    Vector operator*(const Vector& vec) const;

    Matrix operator*(const Matrix& other) const;

    Matrix& operator+=(const Matrix& other);
    Matrix& operator-=(const Matrix& other);
    Matrix& operator*=(Scalar scalar);
    Matrix& operator/=(Scalar scalar);

    [[nodiscard]] Matrix transpose() const;

    void transpose_inplace();

    [[nodiscard]] Vector row(Index r) const;

    [[nodiscard]] Vector col(Index c) const;

    void fill(Scalar value);

    void set_identity();

    static Matrix identity(Index n, StorageOrder order = StorageOrder::RowMajor);

    // Норма
    [[nodiscard]] Scalar norm() const;

private:
    [[nodiscard]] Index offset(Index r, Index c) const noexcept {
        if (order_ == StorageOrder::RowMajor) {
            return r * cols_ + c;
        }
        return c * rows_ + r;
    }

    Index rows_ = 0;
    Index cols_ = 0;
    StorageOrder order_ = StorageOrder::RowMajor;
    std::vector<Scalar> data_;
};

Matrix operator*(Scalar scalar, const Matrix& mat);

} // namespace pla
