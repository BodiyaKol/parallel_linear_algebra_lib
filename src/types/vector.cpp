#include "../../include/types/vector.h"
#include "../../include/types/exceptions.h"
#include <cmath>
#include <algorithm>

namespace pla {
    
Vector Vector::operator+(const Vector& other) const {
    if (size() != other.size()) {
        throw SizeMismatchException(size(), other.size());
    }
    return Vector();
}

Vector Vector::operator-(const Vector& other) const {
    if (size() != other.size()) {
        throw SizeMismatchException(size(), other.size());
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
    if (size() != other.size()) {
        throw SizeMismatchException(size(), other.size());
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

} // namespace pla
