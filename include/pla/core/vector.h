#pragma once

#include <initializer_list>
#include <iosfwd>
#include <vector>

#include "pla/types/index.h"

namespace pla {

template<typename Scalar = double>
    requires Numeric<Scalar>
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

    [[nodiscard]] Scalar* data() noexcept {
        return coordinates_.data();
    }

    [[nodiscard]] const Scalar* data() const noexcept {
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
    auto end()   noexcept { return coordinates_.end();   }
    auto begin() const noexcept { return coordinates_.begin(); }
    auto end()   const noexcept { return coordinates_.end();   }

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

    [[nodiscard]] Scalar dot(const Vector& other) const;

    [[nodiscard]] Scalar norm() const;

    [[nodiscard]] Scalar norm_squared() const;

    void normalize();

    [[nodiscard]] Vector normalized() const;

    [[nodiscard]] bool is_unit(Scalar tolerance = 1e-9) const;

    [[nodiscard]] bool operator==(const Vector& other) const;
    [[nodiscard]] bool operator!=(const Vector& other) const;

private:
    std::vector<Scalar> coordinates_;
};

template<typename Scalar>
Vector<Scalar> operator*(Scalar scalar, const Vector<Scalar>& vec);

template<typename Scalar>
void swap(Vector<Scalar>& a, Vector<Scalar>& b) noexcept;

template<typename Scalar>
std::ostream& operator<<(std::ostream& os, const Vector<Scalar>& v);

} // namespace pla

#include "pla/core/vector_impl.hpp"
