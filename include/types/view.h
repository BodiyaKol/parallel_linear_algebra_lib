#pragma once

#include "index.h"
#include "layout.h"
#include "scalar.h"

namespace pla {

struct VectorView {
	Scalar* ptr = nullptr;
	Index size = 0;
	Index stride = 1;
};

struct ConstVectorView {
	const Scalar* ptr = nullptr;
	Index size = 0;
	Index stride = 1;
};

struct MatrixView {
	Scalar* ptr = nullptr;
	Index rows = 0;
	Index cols = 0;
	Index ld = 0;
	StorageOrder order = StorageOrder::RowMajor;
};

struct ConstMatrixView {
	const Scalar* ptr = nullptr;
	Index rows = 0;
	Index cols = 0;
	Index ld = 0;
	StorageOrder order = StorageOrder::RowMajor;
};

} // namespace pla
