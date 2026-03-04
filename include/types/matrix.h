#pragma once

#include <iosfwd>
#include <vector>
#include <cstdlib>
#include <cstring>
#include <memory>

#include "index.h"
#include "layout.h"
#include "scalar.h"

namespace pla {

class Vector;

class Matrix {
public:
    Matrix() = default;

    Matrix(Index rows, Index cols, StorageOrder order = StorageOrder::RowMajor)
        : rows_(rows), cols_(cols), order_(order) {
            Index n = rows * cols;
            double * ptr = static_cast<double*>(std::aligned_alloc(32, n * sizeof(double)));
            elements_.reset(ptr);

            std::memset(elements_.get(), 0, n * sizeof(double));
        }

    Matrix(Index rows, Index cols, Scalar value, StorageOrder order = StorageOrder::RowMajor)
        : rows_(rows), cols_(cols), order_(order) {
            Index n = rows * cols;
            double * ptr = static_cast<double*>(std::aligned_alloc(32, n * sizeof(double)));
            elements_.reset(ptr);

            std::fill(ptr, ptr + n, value);
        }

    ~Matrix() = default;
    // Matrix(const Matrix& other) = default;

    Matrix(const Matrix& other)
        : rows_(other.rows_), cols_(other.cols_), order_(other.order_) {
            Index n = rows_ * cols_;
            double* ptr = static_cast<double*>(std::aligned_alloc(32, n * sizeof(double)));
            elements_.reset(ptr);
            std::memcpy(ptr, other.elements_.get(), n * sizeof(double));
        }

    Matrix(Matrix&& other) noexcept = default;

    Matrix& operator=(const Matrix& other) {
        if(this == &other) return *this;

        Index n = other.rows_ * other.cols_;
        double* ptr = static_cast<double*>(std::aligned_alloc(32, n * sizeof(double)));
        std::memcpy(ptr, other.elements_.get(), n * sizeof(double));

        rows_ = other.rows_;
        cols_ = other.cols_;
        order_ = other.order_;
        elements_.reset(ptr);

        return *this;
    }

    Matrix& operator=(Matrix&& other) noexcept = default;

    [[nodiscard]] Index rows() const noexcept {
        return rows_;
    }

    [[nodiscard]] Index cols() const noexcept {
        return cols_;
    }

    [[nodiscard]] Index size() const noexcept {
        return rows_ * cols_;
    }

    [[nodiscard]] StorageOrder order() const noexcept {
        return order_;
    }

    [[nodiscard]] bool is_square() const noexcept {
        return rows_ == cols_;
    }

    [[nodiscard]] Scalar* data() noexcept {
        return elements_.get();
    }

    [[nodiscard]] const Scalar* data() const noexcept {
        return elements_.get();
    }

    [[nodiscard]] Scalar& operator()(Index r, Index c) noexcept {
        return elements_.get()[offset(r, c)];
    }

    [[nodiscard]] const Scalar& operator()(Index r, Index c) const noexcept {
        return elements_.get()[offset(r, c)];
    }

    [[nodiscard]] Scalar& at(Index r, Index c);
    [[nodiscard]] const Scalar& at(Index r, Index c) const;

    void clear() noexcept {
        rows_ = 0;
        cols_ = 0;
        elements_.reset(nullptr);
    }

    void swap(Matrix& other) noexcept {
        std::swap(rows_, other.rows_);
        std::swap(cols_, other.cols_);
        std::swap(order_, other.order_);
        elements_.swap(other.elements_);
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

    [[nodiscard]] bool operator==(const Matrix& other) const;
    [[nodiscard]] bool operator!=(const Matrix& other) const;

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
    // std::vector<Scalar> elements_;
    std::unique_ptr<double[], decltype(&std::free)> elements_ {nullptr, &std::free};
};

Matrix operator*(Scalar scalar, const Matrix& mat);

void swap(Matrix& a, Matrix& b) noexcept;

std::ostream& operator<<(std::ostream& os, const Matrix& m);

} // namespace pla
