#pragma once

#include <stdexcept>
#include <string>
#include "pla/types/index.h"

namespace pla {

class PLAException : public std::runtime_error {
public:
    explicit PLAException(const std::string& msg) : std::runtime_error(msg) {}
};

class SizeMismatchException : public PLAException {
public:
    SizeMismatchException(Index size1, Index size2)
        : PLAException("Size mismatch: " + std::to_string(size1) + " != " + std::to_string(size2)) {}
};

class ShapeMismatchException : public PLAException {
public:
    ShapeMismatchException(Index r1, Index c1, Index r2, Index c2)
        : PLAException("Shape mismatch: (" + std::to_string(r1) + "x" + std::to_string(c1) + 
                       ") vs (" + std::to_string(r2) + "x" + std::to_string(c2) + ")") {}
};

class ZeroVectorException : public PLAException {
public:
    ZeroVectorException() : PLAException("Cannot normalize zero vector") {}
};

class NonSquareMatrixException : public PLAException {
public:
    NonSquareMatrixException(Index rows, Index cols)
        : PLAException("Matrix must be square: " + std::to_string(rows) + "x" + std::to_string(cols)) {}
};

class IndexOutOfRangeException : public PLAException {
public:
    IndexOutOfRangeException(Index idx, Index max)
        : PLAException("Index " + std::to_string(idx) + " out of range [0, " + std::to_string(max) + ")") {}
};

class SingularMatrixException : public PLAException {
public:
    SingularMatrixException() : PLAException("Matrix is singular") {}
};

class InvalidSizeException : public PLAException {
public:
    explicit InvalidSizeException(Index size)
        : PLAException("Invalid size: " + std::to_string(size)) {}
};

class InvalidScalarException : public PLAException {
public:
    explicit InvalidScalarException(const std::string& msg) : PLAException(msg) {}
};

class AllocationException : public PLAException {
public:
    AllocationException() : PLAException("Memory allocation failed") {}
};

class ConvergenceException : public PLAException {
public:
    explicit ConvergenceException(const std::string& msg) : PLAException(msg) {}
};

class NotImplementedException : public PLAException {
public:
    NotImplementedException() : PLAException("Not implemented") {}
};

} // namespace pla
