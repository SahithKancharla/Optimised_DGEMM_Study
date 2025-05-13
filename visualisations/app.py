import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker  # <-- Add this at the top

# Serial naive times for matrix sizes 512, 1024, 2048
serial_naive_times = {
    512: 0.417953,
    1024: 3.177837,
    2048: 75.405665,
}

# Threads used for measurements
threads = np.array([2, 4, 6, 7, 8, 16, 32, 64])

# Fill in all parallel times for 512, 1024, 2048 matrix sizes
parallel_times = {
    512: {
        "Blocked":         [0.099851, 0.050187, 0.033423, 0.028685, 0.025109, 0.012609, 0.008232, 0.006330],
        "Blocked Unrolled":[0.076603, 0.038431, 0.025693, 0.022098, 0.019290, 0.009674, 0.007884, 0.004546],
        "Strassen Naive":  [1.186449, 0.637927, 0.420762, 0.332800, 0.333185, 0.192302, 0.101438, 0.100938],
        "Strassen Base":   [0.054681, 0.029248, 0.029062, 0.016993, 0.018045, 0.018415, 0.020137, 0.024958],
        "Strassen Base Unrolled": [0.024794, 0.014956, 0.013982, 0.010718, 0.010872, 0.011811, 0.013206, 0.017468],
    },
    1024: {
        "Blocked":         [0.798076, 0.401601, 0.268159, 0.228958, 0.202970, 0.101205, 0.059141, 0.050701],
        "Blocked Unrolled":[0.615000, 0.307909, 0.204979, 0.175858, 0.154077, 0.077406, 0.039281, 0.037105],
        "Strassen Naive":  [8.020625, 4.236705, 2.717431, 2.326509, 2.170999, 1.122448, 0.613087, 0.575866],
        "Strassen Base":   [0.386530, 0.201313, 0.200315, 0.110655, 0.113219, 0.113072, 0.117994, 0.133473],
        "Strassen Base Unrolled": [0.183716, 0.100139, 0.098943, 0.060607, 0.060976, 0.062539, 0.061904, 0.079070],
    },
    2048: {
        "Blocked":         [9.791930, 4.931426, 3.297403, 2.841755, 2.465088, 1.233346, 0.633683, 0.453593],
        "Blocked Unrolled":[5.313415, 2.709996, 1.805998, 1.552101, 1.344505, 0.671075, 0.447845, 0.312003],
        "Strassen Naive":  [57.599675, 31.076542, 19.521216, 17.020780, 15.903518, 7.877060, 3.975442, 3.646965],
        "Strassen Base":   [2.734615, 1.419409, 1.413392, 0.765089, 0.760444, 0.776766, 0.773871, 0.783679],
        "Strassen Base Unrolled": [1.304559, 0.705134, 0.694611, 0.412015, 0.413620, 0.414734, 0.415209, 0.430842],
    },
}



# Create combined plot
matrix_sizes = [512, 1024, 2048]
fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

for idx, (ax, size) in enumerate(zip(axes, matrix_sizes)):
    for algo, times in parallel_times[size].items():
        speedup = serial_naive_times[size] / np.array(times)
        ax.plot(threads, speedup, marker='o', label=algo)

    ax.set_title(f"{size}x{size}")
    ax.set_xlabel("Threads")

    # Set log scale but use actual thread count as labels
    ax.set_xscale('log', base=2)
    ax.set_xticks(threads)
    ax.set_xticklabels([str(t) for t in threads])  # Use plain numbers
    ax.grid(True, which="both", linestyle='--', linewidth=0.5)

    if idx == 2:
        ax.legend(loc='lower right')

axes[0].set_ylabel("Speedup")
axes[0].set_yscale('log', base=2)
axes[0].set_yticks([1, 2, 4, 8, 16, 32, 64, 128, 256])
axes[0].yaxis.set_major_formatter(ticker.ScalarFormatter())  # <-- Use scalar formatter for plain numbers
axes[0].yaxis.set_minor_formatter(ticker.NullFormatter())    # <-- Optional: hide minor ticks

fig.suptitle("Speedup vs Threads for Different Matrix Sizes", fontsize=14)
fig.tight_layout(rect=[0, 0, 1, 0.99])
fig.savefig("combined_speedup_last_legend.png", dpi=300)
plt.close(fig)