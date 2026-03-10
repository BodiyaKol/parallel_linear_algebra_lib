#ifndef PARALLEL_LINEAR_ALGEBRA_LIB_SHAPE_H
#define PARALLEL_LINEAR_ALGEBRA_LIB_SHAPE_H

#pragma once

#include "pla/types/index.h"

namespace pla {

    struct VectorShape {
        Index n = 0;

        [[nodiscard]] constexpr bool valid() const noexcept {
            return n > 0;
        }
    };

    struct MatrixShape {
        Index rows = 0;
        Index cols = 0;

        [[nodiscard]] constexpr bool valid() const noexcept {
            return rows > 0 && cols > 0;
        }
    };

} // namespace pla

#endif //PARALLEL_LINEAR_ALGEBRA_LIB_SHAPE_H