import re
import sys
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

filename = sys.argv[1] if len(sys.argv) > 1 else "benchmark.txt"

ns, pla_times, eigen_times = [], [], []

with open(filename) as f:
    for line in f:
        m = re.match(r"n=(\d+):\s+PLA avg=([\d.e+-]+)s,\s+Eigen avg=([\d.e+-]+)s", line.strip())
        if m:
            ns.append(int(m.group(1)))
            pla_times.append(float(m.group(2)))
            eigen_times.append(float(m.group(3)))

fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(ns, pla_times,   marker="o", linewidth=2, markersize=6, label="PLA (власна реалізація)")
ax.plot(ns, eigen_times, marker="s", linewidth=2, markersize=6, label="Eigen")

# ax.set_xscale("log", base=1.05)
# ax.set_yscale("log", )

ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"n={int(x)}"))
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{y:.2e}s"))

ax.set_xlabel("Розмір матриці n×n (логарифмічна шкала, база 2)", fontsize=12)
ax.set_ylabel("Середній час (с, лог. шкала)", fontsize=12)
ax.set_title("Порівняння продуктивності множення матриць: PLA vs Eigen", fontsize=13)

ax.grid(True, which="both", linestyle="--", alpha=0.5)
ax.legend(fontsize=11)
ax.tick_params(axis="x", rotation=45)

plt.tight_layout()
plt.savefig("benchmark_plot.png", dpi=150)
print("Збережено: benchmark_plot.png")
plt.show()
