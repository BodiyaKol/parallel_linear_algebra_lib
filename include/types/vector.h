#pragma once

#include <initializer_list>
#include <iosfwd>
#include <vector>

#include "index.h"
#include "scalar.h"

namespace pla {

class Vector {
public:
    Vector() = default;
    explicit Vector(Index n) : coordinates_(n) {}
    Vector(Index n, Scalar value) : coordinates_(n, value) {}
    Vector(std::initializer_list<Scalar> init) : coordinates_(init) {}

    ~Vector() = default;
    Vector(const Vector& other) = default;
    Vector(Vector&& other) noexcept = default;
    Vector& operator=(const Vector& other) = default;
    Vector& operator=(Vector&& other) noexcept = default;

    [[nodiscard]] Index dimension() const noexcept {
        return coordinates_.size();
    }

    void resize(Index n) {
        coordinates_.resize(n);
    }

    [[nodiscard]] bool empty() const noexcept {
        return coordinates_.empty();
    }

    [[nodiscard]] Scalar* coordinates() noexcept {
        return coordinates_.data();
    }

    [[nodiscard]] const Scalar* coordinates() const noexcept {
        return coordinates_.data();
    }

    [[nodiscard]] Scalar& operator[](Index i) noexcept {
        return coordinates_[i];
    }

    [[nodiscard]] const Scalar& operator[](Index i) const noexcept {
        return coordinates_[i];
    }

    [[nodiscard]] Scalar& at(Index i);
    [[nodiscard]] const Scalar& at(Index i) const;

    auto begin() noexcept { return coordinates_.begin(); }
    auto end() noexcept { return coordinates_.end(); }
    auto begin() const noexcept { return coordinates_.begin(); }
    auto end() const noexcept { return coordinates_.end(); }

    void clear() noexcept { coordinates_.clear(); }

    void swap(Vector& other) noexcept { coordinates_.swap(other.coordinates_); }

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

    [[nodiscard]] bool operator==(const Vector& other) const;
    [[nodiscard]] bool operator!=(const Vector& other) const;

private:
    std::vector<Scalar> coordinates_;
};

// Множення скаляра на вектор зліва: α * v
Vector operator*(Scalar scalar, const Vector& vec);

void swap(Vector& a, Vector& b) noexcept;

std::ostream& operator<<(std::ostream& os, const Vector& v);

} // namespace pla
