#pragma once

namespace pla {

enum class StatusCode {
	Ok = 0,
	InvalidArgument,
	SizeMismatch,
	SingularMatrix,
	NonConvergent,
	NotImplemented,
	BackendError
};

struct Status {
	StatusCode code = StatusCode::Ok;
	const char* message = "ok";

	[[nodiscard]] constexpr bool ok() const noexcept {
		return code == StatusCode::Ok;
	}

	static constexpr Status success() noexcept {
		return {StatusCode::Ok, "ok"};
	}
};

} // namespace pla
