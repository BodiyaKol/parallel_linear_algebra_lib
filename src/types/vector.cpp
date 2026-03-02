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

// Реалізувати складений оператор +=, -=, *=, /=
Vector& Vector::operator+=(const Vector& other) {
    return;
}

Vector& Vector::operator-=(const Vector& other) {
    return;
}

Vector& Vector::operator*=(Scalar scalar) {
    return *this;
}

Vector& Vector::operator/=(Scalar scalar) {
    if (std::abs(scalar) < 1e-15) {
        throw InvalidScalarException("Division by zero");
    }
    return *this;
}

// Реалізувати скалярний добуток: v1.dot(v2)
Scalar Vector::dot(const Vector& other) const {
    if (dimension() != other.dimension()) {
        throw SizeMismatchException(dimension(), other.dimension());
    }
    return ;
}

// Реалізувати норму вектора 
Scalar Vector::norm() const {
    return ;
}

// Реалізувати квадрат норми
Scalar Vector::norm_squared() const {
    return ;
}

// Реалізувати нормалізацію in-place: v.normalize()
void Vector::normalize() {
    
}

// Реалізувати нормалізовану копію: v_unit = v.normalized()
Vector Vector::normalized() const {
    return;
}

// TODO 13: Перевірити чи вектор одиничний
bool Vector::is_unit(Scalar tolerance) const {
    Scalar n = norm();
    return std::abs(n - 1.0) < tolerance;
}

// Реалізувати α * v (скаляр зліва)
Vector operator*(Scalar scalar, const Vector& vec) {
    return;
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
