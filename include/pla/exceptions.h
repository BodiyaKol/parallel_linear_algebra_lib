#pragma once

#include <stdexcept>
#include <string>
#include "pla/types/index.h"

namespace pla {

class PLAException : public std::runtime_error {
public:
    explicit PLAException(const std::string& msg) : std::runtime_error(msg) {}
};

class SizeMismatchException : public PLAException, public std::length_error {
public:
    SizeMismatchException(Index size1, Index size2)
        : PLAException(make_msg(size1, size2))
        , std::length_error(make_msg(size1, size2)) {}

    const char* what() const noexcept override {
        return PLAException::what();
    }

private:
    static std::string make_msg(Index s1, Index s2) {
        return "Size mismatch: " + std::to_string(s1) + " != " + std::to_string(s2);
    }
};

class ShapeMismatchException : public PLAException, public std::invalid_argument {
public:
    ShapeMismatchException(Index r1, Index c1, Index r2, Index c2)
        : PLAException(make_msg(r1, c1, r2, c2))
        , std::invalid_argument(make_msg(r1, c1, r2, c2)) {}

    const char* what() const noexcept override {
        return PLAException::what();
    }

private:
    static std::string make_msg(Index r1, Index c1, Index r2, Index c2) {
        return "Shape mismatch: (" + std::to_string(r1) + "x" + std::to_string(c1) +
               ") vs (" + std::to_string(r2) + "x" + std::to_string(c2) + ")";
    }
};

class NonSquareMatrixException : public PLAException, public std::invalid_argument {
public:
    NonSquareMatrixException(Index rows, Index cols)
        : PLAException(make_msg(rows, cols))
        , std::invalid_argument(make_msg(rows, cols)) {}

    const char* what() const noexcept override {
        return PLAException::what();
    }

private:
    static std::string make_msg(Index r, Index c) {
        return "Matrix must be square: " + std::to_string(r) + "x" + std::to_string(c);
    }
};

class InvalidSizeException : public PLAException, public std::invalid_argument {
public:
    explicit InvalidSizeException(Index size)
        : PLAException(make_msg(size))
        , std::invalid_argument(make_msg(size)) {}

    const char* what() const noexcept override {
        return PLAException::what();
    }

private:
    static std::string make_msg(Index s) {
        return "Invalid size: " + std::to_string(s);
    }
};


class IndexOutOfRangeException : public PLAException, public std::out_of_range {
public:
    IndexOutOfRangeException(Index idx, Index max)
        : PLAException(make_msg(idx, max))
        , std::out_of_range(make_msg(idx, max)) {}

    const char* what() const noexcept override {
        return PLAException::what();
    }

private:
    static std::string make_msg(Index idx, Index max) {
        return "Index " + std::to_string(idx) + " out of range [0, " + std::to_string(max) + ")";
    }
};


class SingularMatrixException : public PLAException, public std::domain_error {
public:
    SingularMatrixException()
        : PLAException("Matrix is singular")
        , std::domain_error("Matrix is singular") {}

    const char* what() const noexcept override {
        return PLAException::what();
    }
};


class ZeroVectorException : public PLAException, public std::domain_error {
public:
    ZeroVectorException()
        : PLAException("Cannot normalize zero vector")
        , std::domain_error("Cannot normalize zero vector") {}

    const char* what() const noexcept override {
        return PLAException::what();
    }
};


class ConvergenceException : public PLAException, public std::domain_error {
public:
    explicit ConvergenceException(const std::string& msg)
        : PLAException(msg)
        , std::domain_error(msg) {}

    const char* what() const noexcept override {
        return PLAException::what();
    }
};


class InvalidScalarException : public PLAException, public std::invalid_argument {
public:
    explicit InvalidScalarException(const std::string& msg)
        : PLAException(msg)
        , std::invalid_argument(msg) {}

    const char* what() const noexcept override {
        return PLAException::what();
    }
};


class AllocationException : public PLAException, public std::bad_alloc {
public:
    AllocationException()
        : PLAException("Memory allocation failed")
        , std::bad_alloc() {}

    const char* what() const noexcept override {
        return PLAException::what();
    }
};

class NotImplementedException : public PLAException, public std::logic_error {
public:
    NotImplementedException()
        : PLAException("Not implemented")
        , std::logic_error("Not implemented") {}

    const char* what() const noexcept override {
        return PLAException::what();
    }
};

} // namespace pla