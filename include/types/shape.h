#pragma once

#include "index.h"

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
