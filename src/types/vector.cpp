#include "../../include/types/vector.h"
#include "../../include/types/exceptions.h"
#include <algorithm>
#include <cmath>
#include <ostream>

namespace pla {
    
Vector Vector::operator+(const Vector& other) const {
    if (dimension() != other.dimension()) {
        throw SizeMismatchException(dimension(), other.dimension());
    }
    return Vector();
}

Vector Vector::operator-(const Vector& other) const {
    if (dimension() != other.dimension()) {
        throw SizeMismatchException(dimension(), other.dimension());
    }
    return Vector();
}

Vector Vector::operator-() const {
    return Vector();
}

//Реалізувати множення на скаляр: v * α
Vector Vector::operator*(Scalar scalar) const {
    return Vector();
}

// Реалізувати ділення на скаляр: v / α
Vector Vector::operator/(Scalar scalar) const {
    if (std::abs(scalar) < 1e-15) {
        throw InvalidScalarException("Division by zero");
    }
    return Vector();
}

Vector& Vector::operator+=(const Vector& other) {
    if (dimension() != other.dimension())
        throw SizeMismatchException(dimension(), other.dimension());
    for (Index i = 0; i < dimension(); ++i)
        coordinates_[i] += other[i];
    return *this;
}

Vector& Vector::operator-=(const Vector& other) {
    if (dimension() != other.dimension())
        throw SizeMismatchException(dimension(), other.dimension());
    for (Index i = 0; i < dimension(); ++i)
        coordinates_[i] -= other[i];
    return *this;
}

Vector& Vector::operator*=(Scalar scalar) {
    for (Index i = 0; i < dimension(); ++i)
        coordinates_[i] *= scalar;
    return *this;
}

Vector& Vector::operator/=(Scalar scalar) {
    if (std::abs(scalar) < 1e-15)
        throw InvalidScalarException("Division by zero");
    for (Index i = 0; i < dimension(); ++i)
        coordinates_[i] /= scalar;
    return *this;
}


Scalar Vector::dot(const Vector& other) const {
    if (dimension() != other.dimension())
        throw SizeMismatchException(dimension(), other.dimension());
    Scalar sum = 0.0;
    for (Index i = 0; i < dimension(); ++i)
        sum += coordinates_[i] * other[i];
    return sum;
}

Scalar Vector::norm() const {
    return std::sqrt(dot(*this));
}

Scalar Vector::norm_squared() const {
    return dot(*this);
}

void Vector::normalize() {
    Scalar n = norm();
    if (n < 1e-15)
        throw std::runtime_error("Cannot normalize zero vector");
    (*this) /= n;
}

Vector Vector::normalized() const {
    Scalar n = norm();
    if (n < 1e-15)
        throw std::runtime_error("Cannot normalize zero vector");
    return (*this) * (1.0 / n);
}

// TODO 13: Перевірити чи вектор одиничний
bool Vector::is_unit(Scalar tolerance) const {
    Scalar n = norm();
    return std::abs(n - 1.0) < tolerance;
}

Vector operator*(Scalar scalar, const Vector& vec) {
    return vec * scalar;
}

Scalar& Vector::at(Index i) {
    if (i >= dimension()) {
        throw IndexOutOfRangeException(i, dimension());
    }
    return coordinates_[i];
}

const Scalar& Vector::at(Index i) const {
    if (i >= dimension()) {
        throw IndexOutOfRangeException(i, dimension());
    }
    return coordinates_[i];
}

bool Vector::operator==(const Vector& other) const {
    return coordinates_ == other.coordinates_;
}

bool Vector::operator!=(const Vector& other) const {
    return !(*this == other);
}

void swap(Vector& a, Vector& b) noexcept {
    a.swap(b);
}

std::ostream& operator<<(std::ostream& os, const Vector& v) {
    os << "[";
    for (Index i = 0; i < v.dimension(); ++i) {
        if (i > 0) os << ", ";
        os << v[i];
    }
    os << "]";
    return os;
}

} // namespace pla
