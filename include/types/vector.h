#pragma once

#include <initializer_list>
#include <vector>

#include "index.h"
#include "scalar.h"

namespace pla {

class Vector {
public:
    Vector() = default;
    explicit Vector(Index n) : data_(n) {}
    Vector(Index n, Scalar value) : data_(n, value) {}
    Vector(std::initializer_list<Scalar> init) : data_(init) {}

    [[nodiscard]] Index size() const noexcept {
        return data_.size();
    }

    void resize(Index n) {
        data_.resize(n);
    }

    [[nodiscard]] bool empty() const noexcept {
        return data_.empty();
    }

    [[nodiscard]] Scalar* data() noexcept {
        return data_.data();
    }

    [[nodiscard]] const Scalar* data() const noexcept {
        return data_.data();
    }

    [[nodiscard]] Scalar& operator[](Index i) noexcept {
        return data_[i];
    }

    [[nodiscard]] const Scalar& operator[](Index i) const noexcept {
        return data_[i];
    }

    // Операції над векторами
    Vector operator+(const Vector& other) const;

    Vector operator-(const Vector& other) const;

    Vector operator-() const;

    Vector operator*(Scalar scalar) const;

    Vector operator/(Scalar scalar) const;

    Vector& operator+=(const Vector& other);
    Vector& operator-=(const Vector& other);
    Vector& operator*=(Scalar scalar);
    Vector& operator/=(Scalar scalar);

    // Скалярний добуток (dot product): v1.dot(v2)
    [[nodiscard]] Scalar dot(const Vector& other) const;

    // Норма вектора
    [[nodiscard]] Scalar norm() const;

    // Квадрат норми
    [[nodiscard]] Scalar norm_squared() const;

    // Нормалізація in-place (змінює поточний вектор)
    void normalize();

    // Повертає нормалізовану копію
    [[nodiscard]] Vector normalized() const;

    // Перевірка чи вектор одиничний (norm ≈ 1)
    [[nodiscard]] bool is_unit(Scalar tolerance = 1e-9) const;

private:
    std::vector<Scalar> data_;
};

// Множення скаляра на вектор зліва: α * v
Vector operator*(Scalar scalar, const Vector& vec);

} // namespace pla
