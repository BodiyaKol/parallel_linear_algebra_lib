#pragma once

#include "pla/types/index.h"

namespace pla {

enum class Backend {
	Serial,
	Simd,
	Tbb
};

struct ExecutionPolicy {
	Backend backend = Backend::Serial;
	Index thread_count = 0;
	Index block_size = 0;
};

} // namespace pla
