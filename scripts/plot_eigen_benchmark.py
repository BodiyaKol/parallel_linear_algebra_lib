import csv
import matplotlib.pyplot as plt

sizes = []
pla_ms = []
eigen_ms = []
threshold_ms = []
ratios = []
errors = []

with open("eigen_benchmark.csv", newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        sizes.append(int(row["size"]))
        pla_ms.append(float(row["total_pla_ms"]))
        eigen_ms.append(float(row["eigen_ms"]))
        threshold_ms.append(float(row["threshold_ms"]))
        ratios.append(float(row["ratio"]))
        errors.append(float(row["max_abs_error"]))

plt.figure(figsize=(10, 6))
plt.plot(sizes, pla_ms, marker="o", label="PLA")
plt.plot(sizes, eigen_ms, marker="o", label="Eigen")
plt.plot(sizes, threshold_ms, marker="o", color="red", label="3x Eigen")
plt.yscale("log")
plt.xlabel("Matrix size n")
plt.ylabel("Time, ms (log scale)")
plt.title("Eigenvalues benchmark: PLA vs Eigen")
plt.grid(True, which="both")
plt.legend()
plt.tight_layout()
plt.savefig("eigen_benchmark_time_log.png", dpi=200)

plt.figure(figsize=(10, 6))
plt.plot(sizes, ratios, marker="o", label="PLA / Eigen")
plt.axhline(3.0, color="red", linestyle="--", label="Allowed limit: 3x Eigen")
plt.xlabel("Matrix size n")
plt.ylabel("Time ratio")
plt.title("PLA slowdown relative to Eigen")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("eigen_benchmark_ratio.png", dpi=200)

plt.figure(figsize=(10, 6))
plt.plot(sizes, errors, marker="o", label="max |PLA λ - Eigen λ|")
plt.yscale("log")
plt.xlabel("Matrix size n")
plt.ylabel("Max absolute eigenvalue error")
plt.title("Eigenvalue accuracy: PLA vs Eigen")
plt.grid(True, which="both")
plt.legend()
plt.tight_layout()
plt.savefig("eigen_benchmark_accuracy.png", dpi=200)

plt.show()