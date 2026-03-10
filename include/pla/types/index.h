#pragma once

#include <cstddef>
#include <type_traits>

namespace pla {

using Index = std::size_t;

template<typename Scalar>
concept Numeric = std::is_arithmetic_v<Scalar>;

} // namespace pla
