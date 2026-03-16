#pragma once

#include <iosfwd>
#include <vector>
#include <cstring>
#include <memory>
#include <type_traits>

#include "pla/types/index.h"
#include "pla/types/layout.h"
#include "pla/core/vector.h"

namespace pla {

template<typename Scalar = double>
    requires Numeric<Scalar>
class Matrix {
public:
    using valueType = Scalar;
    using size_type = Index;

    Matrix() = default;

    Matrix(Index rows, Index cols, StorageOrder order = StorageOrder::RowMajor)
        : rows_(rows), cols_(cols), order_(order)
    {
        allocate();
        std::memset(elements_.get(), 0, size() * sizeof(Scalar));
    }

    Matrix(Index rows, Index cols, Scalar value, StorageOrder order = StorageOrder::RowMajor)
        : rows_(rows), cols_(cols), order_(order)
    {
        allocate();
        std::fill(elements_.get(), elements_.get() + size(), value);
    }

    ~Matrix() = default;

    Matrix(const Matrix& other)
        : rows_(other.rows_), cols_(other.cols_), order_(other.order_)
    {
        allocate();
        std::memcpy(elements_.get(),
            other.elements_.get(),
            size() * sizeof(Scalar));
    }

    Matrix(Matrix&& other) noexcept = default;

    Matrix& operator=(const Matrix& other) {
        if(this == &other) return *this;

        Matrix tmp{other};
        this->swap(tmp);

        return *this;
    }

    Matrix& operator=(Matrix&& other) noexcept = default;

    [[nodiscard]] Index rows() const noexcept { return rows_; }
    [[nodiscard]] Index cols() const noexcept { return cols_; }
    [[nodiscard]] Index size() const noexcept { return rows_ * cols_; }

    [[nodiscard]] StorageOrder order() const noexcept { return order_; }

    [[nodiscard]] bool is_square()     const noexcept { return rows_ == cols_; }

    [[nodiscard]] Scalar* data()             noexcept { return elements_.get(); }

    [[nodiscard]] const Scalar* data() const noexcept { return elements_.get(); }

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

    Matrix<Scalar> operator+(const Matrix& other) const;

    Matrix operator-(const Matrix& other) const;

    Matrix operator-() const;

    Matrix operator*(Scalar scalar) const;

    Matrix operator/(Scalar) const;

    Vector<Scalar> operator*(const Vector<Scalar>& vec) const;

    Matrix operator*(const Matrix<Scalar>& B) const;

    Matrix& operator+=(const Matrix& other);
    Matrix& operator-=(const Matrix& other);
    Matrix& operator*=(Scalar scalar);
    Matrix& operator/=(Scalar scalar);

    [[nodiscard]] Matrix transpose() const;

    void transpose_inplace();

    [[nodiscard]] Vector<Scalar> row(Index r) const;

    [[nodiscard]] Vector<Scalar> col(Index c) const;

    void fill(Scalar value);

    void set_identity();

    static Matrix identity(Index n, StorageOrder order = StorageOrder::RowMajor);

    [[nodiscard]] Scalar norm() const;

    [[nodiscard]] bool operator==(const Matrix& other) const;
    [[nodiscard]] bool operator!=(const Matrix& other) const;

private:
    struct AlignedDeleter {
        void operator()(Scalar* ptr) const noexcept {
            if (ptr)
                operator delete[](ptr, std::align_val_t{64});
        }
    };

    void allocate() {
        if (size() == 0)
            return;

        auto* raw = new (std::align_val_t{64}) Scalar[size()];
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

    std::unique_ptr<Scalar[], AlignedDeleter> elements_;
};

template<typename Scalar>
    requires Numeric<Scalar>
Matrix<Scalar> operator*(Scalar scalar, const Matrix<Scalar>& mat);

template<typename Scalar>
    requires Numeric<Scalar>
void swap(Matrix<Scalar>& a, Matrix<Scalar>& b) noexcept;

template<typename Scalar>
    requires Numeric<Scalar>
std::ostream& operator<<(std::ostream& os, const Matrix<Scalar>& m);

} // namespace pla

#include "pla/core/matrix_impl.hpp"
#include "pla/core/vector_impl.hpp"
#include "pla/opt/matrix_add.hpp"
#include "pla/opt/matrix_multiplication.hpp"
#include "pla/decompos/lu.h"
