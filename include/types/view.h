#pragma once

#include "index.h"
#include "layout.h"

namespace pla {

template<typename Scalar>
struct VectorView {
	Scalar ptr = nullptr;
	Index size = 0;
	Index stride = 1;
};

template<typename Scalar>
struct ConstVectorView {
	const Scalar* ptr = nullptr;
	Index size = 0;
	Index stride = 1;
};

template<typename Scalar>
struct MatrixView {
	Scalar* ptr = nullptr;
	Index rows = 0;
	Index cols = 0;
	Index ld = 0;
	StorageOrder order = StorageOrder::RowMajor;
};

template<typename Scalar>
struct ConstMatrixView {
	const Scalar* ptr = nullptr;
	Index rows = 0;
	Index cols = 0;
	Index ld = 0;
	StorageOrder order = StorageOrder::RowMajor;
};

} // namespace pla
