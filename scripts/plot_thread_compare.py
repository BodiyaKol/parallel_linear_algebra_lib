import csv
import matplotlib.pyplot as plt

def read_csv(path):
    sizes = []
    total = []
    eigen = []
    ratio = []

    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sizes.append(int(row["size"]))
            total.append(float(row["total_pla_ms"]))
            eigen.append(float(row["eigen_ms"]))
            ratio.append(float(row["ratio"]))

    return sizes, total, eigen, ratio

sizes_s, pla_single, eigen_s, ratio_s = read_csv("eigen_benchmark_single.csv")
sizes_o, pla_openmp, eigen_o, ratio_o = read_csv("eigen_benchmark_openmp.csv")

plt.figure(figsize=(10, 6))
plt.plot(sizes_s, pla_single, marker="o", label="PLA single-thread")
plt.plot(sizes_o, pla_openmp, marker="o", label="PLA OpenMP")
plt.plot(sizes_s, eigen_s, marker="o", label="Eigen")
plt.plot(sizes_s, [3 * x for x in eigen_s], marker="o", color="red", label="3x Eigen")

plt.yscale("log")
plt.xlabel("Matrix size n")
plt.ylabel("Time, ms (log scale)")
plt.title("PLA single-thread vs OpenMP vs Eigen")
plt.grid(True, which="both")
plt.legend()
plt.tight_layout()
plt.savefig("eigen_benchmark_thread_compare.png", dpi=200)

plt.figure(figsize=(10, 6))
plt.plot(sizes_s, ratio_s, marker="o", label="PLA single / Eigen")
plt.plot(sizes_o, ratio_o, marker="o", label="PLA OpenMP / Eigen")
plt.axhline(3.0, color="red", linestyle="--", label="Allowed limit: 3x Eigen")

plt.xlabel("Matrix size n")
plt.ylabel("Time ratio")
plt.title("Threading impact: slowdown relative to Eigen")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("eigen_benchmark_thread_ratio.png", dpi=200)

plt.figure(figsize=(10, 6))
speedup = [s / o if o > 0 else 0 for s, o in zip(pla_single, pla_openmp)]

plt.plot(sizes_s, speedup, marker="o", label="OpenMP speedup over single-thread")
plt.axhline(1.0, color="red", linestyle="--", label="No speedup")

plt.xlabel("Matrix size n")
plt.ylabel("Speedup")
plt.title("OpenMP speedup for PLA eigen solver")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("eigen_benchmark_openmp_speedup.png", dpi=200)

plt.show()
