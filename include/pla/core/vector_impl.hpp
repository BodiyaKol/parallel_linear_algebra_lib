#ifndef PARALLEL_LINEAR_ALGEBRA_LIB_VECTOR_IMPL_H
#define PARALLEL_LINEAR_ALGEBRA_LIB_VECTOR_IMPL_H

// #include "pla/core/vector.h"
#include <algorithm>
#include <cmath>
#include <ostream>

#include "pla/exceptions.h"

namespace pla {

template<typename Scalar>
    requires Numeric<Scalar>
Vector<Scalar> Vector<Scalar>::operator+(const Vector& other) const {
    if (dimension() != other.dimension()) {
        throw SizeMismatchException(dimension(), other.dimension());
    }
    return Vector();
}

template<typename Scalar>
    requires Numeric<Scalar>
Vector<Scalar> Vector<Scalar>::operator-(const Vector& other) const {
    if (dimension() != other.dimension()) {
        throw SizeMismatchException(dimension(), other.dimension());
    }
    return Vector();
}

template<typename Scalar>
    requires Numeric<Scalar>
Vector<Scalar> Vector<Scalar>::operator-() const {
    return Vector();
}

template<typename Scalar>
    requires Numeric<Scalar>
Vector<Scalar> Vector<Scalar>::operator*(Scalar scalar) const {
    return Vector();
}

template<typename Scalar>
    requires Numeric<Scalar>
Vector<Scalar> Vector<Scalar>::operator/(Scalar scalar) const {
    if (std::abs(scalar) < 1e-15) {
        throw InvalidScalarException("Division by zero");
    }
    return Vector();
}

template<typename Scalar>
    requires Numeric<Scalar>
Vector<Scalar>& Vector<Scalar>::operator+=(const Vector& other) {
    if (dimension() != other.dimension())
        throw SizeMismatchException(dimension(), other.dimension());
    for (Index i = 0; i < dimension(); ++i)
        coordinates_[i] += other[i];
    return *this;
}

template<typename Scalar>
    requires Numeric<Scalar>
Vector<Scalar>& Vector<Scalar>::operator-=(const Vector& other) {
    if (dimension() != other.dimension())
        throw SizeMismatchException(dimension(), other.dimension());
    for (Index i = 0; i < dimension(); ++i)
        coordinates_[i] -= other[i];
    return *this;
}

template<typename Scalar>
    requires Numeric<Scalar>
Vector<Scalar>& Vector<Scalar>::operator*=(Scalar scalar) {
    for (Index i = 0; i < dimension(); ++i)
        coordinates_[i] *= scalar;
    return *this;
}

template<typename Scalar>
    requires Numeric<Scalar>
Vector<Scalar>& Vector<Scalar>::operator/=(Scalar scalar) {
    if (std::abs(scalar) < 1e-15)
        throw InvalidScalarException("Division by zero");
    for (Index i = 0; i < dimension(); ++i)
        coordinates_[i] /= scalar;
    return *this;
}


template<typename Scalar>
    requires Numeric<Scalar>
Scalar Vector<Scalar>::dot(const Vector<Scalar>& other) const {
    if (dimension() != other.dimension())
        throw SizeMismatchException(dimension(), other.dimension());
    Scalar sum = 0.0;
    for (Index i = 0; i < dimension(); ++i)
        sum += coordinates_[i] * other[i];
    return sum;
}

template<typename Scalar>
    requires Numeric<Scalar>
Scalar Vector<Scalar>::norm() const {
    return std::sqrt(dot(*this));
}

template<typename Scalar>
    requires Numeric<Scalar>
Scalar Vector<Scalar>::norm_squared() const {
    return dot(*this);
}

template<typename Scalar>
    requires Numeric<Scalar>
void Vector<Scalar>::normalize() {
    Scalar n = norm();
    if (n < 1e-15)
        throw std::runtime_error("Cannot normalize zero vector");
    (*this) /= n;
}

template<typename Scalar>
    requires Numeric<Scalar>
Vector<Scalar> Vector<Scalar>::normalized() const {
    Scalar n = norm();
    if (n < 1e-15)
        throw std::runtime_error("Cannot normalize zero vector");
    return (*this) * (1.0 / n);
}

// TODO 13: Перевірити чи вектор одиничний
template<typename Scalar>
    requires Numeric<Scalar>
bool Vector<Scalar>::is_unit(Scalar tolerance) const {
    Scalar n = norm();
    return std::abs(n - 1.0) < tolerance;
}

template<typename Scalar>
    requires Numeric<Scalar>
Vector<Scalar> operator*(Scalar scalar, const Vector<Scalar>& vec) {
    return vec * scalar;
}

template<typename Scalar>
    requires Numeric<Scalar>
Scalar& Vector<Scalar>::at(Index i) {
    if (i >= dimension()) {
        throw IndexOutOfRangeException(i, dimension());
    }
    return coordinates_[i];
}

template<typename Scalar>
    requires Numeric<Scalar>
const Scalar& Vector<Scalar>::at(Index i) const {
    if (i >= dimension()) {
        throw IndexOutOfRangeException(i, dimension());
    }
    return coordinates_[i];
}

template<typename Scalar>
    requires Numeric<Scalar>
bool Vector<Scalar>::operator==(const Vector<Scalar>& other) const {
    return coordinates_ == other.coordinates_;
}

template<typename Scalar>
    requires Numeric<Scalar>
bool Vector<Scalar>::operator!=(const Vector<Scalar>& other) const {
    return !(*this == other);
}

template<typename Scalar>
void swap(Vector<Scalar>& a, Vector<Scalar>& b) noexcept {
    a.swap(b);
}

template<typename Scalar>
std::ostream& operator<<(std::ostream& os, const Vector<Scalar>& v) {
    os << "[";
    for (Index i = 0; i < v.dimension(); ++i) {
        if (i > 0) os << ", ";
        os << v[i];
    }
    os << "]";
    return os;
}

} // namespace pla

#endif //PARALLEL_LINEAR_ALGEBRA_LIB_VECTOR_IMPL_H