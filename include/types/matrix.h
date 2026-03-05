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
            allocate();
            std::memset(elements_.get(), 0, size() * sizeof(double));
        }

    Matrix(Index rows, Index cols, Scalar value, StorageOrder order = StorageOrder::RowMajor)
        : rows_(rows), cols_(cols), order_(order) {
            allocate();
            std::fill(elements_.get(), elements_.get() + size(), value);
        }

    ~Matrix() = default;

    Matrix(const Matrix& other)
        : rows_(other.rows_), cols_(other.cols_), order_(other.order_) {
            allocate();
            std::memcpy(elements_.get(),
                other.elements_.get(),
                size() * sizeof(double));
        }

    Matrix(Matrix&& other) noexcept = default;

    Matrix& operator=(const Matrix& other) {
        if(this == &other) return *this;

        Matrix tmp{other};
        this->swap(tmp);

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
    struct AlignedDeleter {
        void operator()(double* ptr) const noexcept {
            if (ptr)
                operator delete[](ptr, std::align_val_t(64));
        }
    };

    void allocate() {
        if (size() == 0)
            return;

        double* raw = new (std::align_val_t(64)) double[size()];
        elements_.reset(raw);
    }
    
    [[nodiscard]] Index offset(Index r, Index c) const noexcept {
        if (order_ == StorageOrder::RowMajor) {
            return r * cols_ + c;
        }
        return c * rows_ + r;
    }

    Index rows_ = 0;
    Index cols_ = 0;
    StorageOrder order_ = StorageOrder::RowMajor;

    std::unique_ptr<double[], AlignedDeleter> elements_;
};

Matrix operator*(Scalar scalar, const Matrix& mat);

void swap(Matrix& a, Matrix& b) noexcept;

std::ostream& operator<<(std::ostream& os, const Matrix& m);

} // namespace pla
