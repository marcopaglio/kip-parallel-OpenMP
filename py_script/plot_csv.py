import pandas as pd
import matplotlib.pyplot as plt
import sys


def plotResults(csv_filename = "../data/results.csv", min_relative_time = 0.05, min_marginal_speedup = 0.2, min_efficiency = 0.5):
    df = pd.read_csv(csv_filename)
    
    # Add new column: efficiency = speedup / num_threads
    df["Efficiency"] = None
    for keys, group in df.groupby(["ImageName", "KernelName", "ImageDimension", "KernelDimension"]):
        df.loc[group.index, "Efficiency"] = group["SpeedUp"] / group["NumThreads"]
    
    # Group same-size-images-and-kernels and calculate means
    grouped = (
        df.groupby(["ImageDimension", "KernelDimension", "NumThreads"])
          .agg({"TimePerRep_s": "mean", "SpeedUp": "mean", "Efficiency": "mean"})
          .reset_index()
    )
    
    
    for (image_dim, kernel_dim), subgroup in grouped.groupby(["ImageDimension", "KernelDimension"]):
        subgroup = subgroup.sort_values("NumThreads")
    
        fig, ax1 = plt.subplots(figsize=(8,6))
    
        # First axis for average time
        ax1.plot(
            subgroup["NumThreads"],
            subgroup["TimePerRep_s"],
            marker="o",
            linestyle="-",
            label="time",
            markersize=6
        )
        for i, x, y in zip(range(len(subgroup.index)), subgroup["NumThreads"], subgroup["TimePerRep_s"]):
            if i > 0 and (subgroup["TimePerRep_s"][i-1] - y) < min_relative_time * subgroup["TimePerRep_s"][i-1]:
                color = "red"
            else:
                color = "black"
            ax1.annotate(f"{y:.2f}", (x, y), xytext=(-10, -5), textcoords="offset points",
                ha="center", va="top", fontsize=8, color=color)
            
        ax1.set_xlabel("Threads number")
        ax1.set_ylabel("Time (s)")
        h1, l1 = ax1.get_legend_handles_labels()
    
    
        # Second axis for speedup
        ax2 = ax1.twinx()
        ax2.plot(
            subgroup["NumThreads"],
            subgroup["SpeedUp"],
            marker="s",
            linestyle="-",
            color="green",
            label="speedup"
        )
        for i, x, y in zip(range(len(subgroup.index)), subgroup["NumThreads"], subgroup["SpeedUp"]):
            if i > 0 and (y - subgroup["SpeedUp"][i-1]) < min_marginal_speedup:
                color = "red"
            else:
                color = "black"
            ax2.annotate(f"{y:.2f}", (x, y), xytext=(-10, 5), textcoords="offset points",
                ha="center", va="bottom", fontsize=8, color=color)
        
        ax2.set_ylabel("SpeedUp")
        
        
        # Third axis for efficiency
        ax2.scatter(
            subgroup["NumThreads"],
            subgroup["SpeedUp"],
            c="palegoldenrod",
            s= 150 + 500 * subgroup["Efficiency"].apply(lambda s: s),
            label="efficiency"
        )
        
        ax2_opposite = ax2.twiny()
        xticks = list([round(item,2) for item in subgroup["Efficiency"]])    
        ax2_opposite.set_xlim(ax2.get_xlim())
        ax2_opposite.set_xticks(list(subgroup["NumThreads"]))
        ax2_opposite.set_xticklabels(xticks)
        for label in ax2_opposite.get_xticklabels():
            if float(label.get_text()) < min_efficiency:
                label.set_color("red");
        ax2_opposite.set_xlabel("Efficiency")
        ax2_opposite.grid(True, axis="x", linestyle="--", alpha=0.6)
        h2, l2 = ax2.get_legend_handles_labels()
    
    
        plt.title(f"Strong scaling: {image_dim} images | {kernel_dim}x{kernel_dim} kernels")
        plt.legend(h1+h2, l1+l2, loc="center right")
        plt.tight_layout()
        
        filename = f"Strong_scaling_{image_dim}_{kernel_dim}.png".replace("x", "x")
        plt.savefig(filename, dpi=150)
        plt.show()
        plt.close()


if __name__ == "__main__":
    params = {}
    if len(sys.argv) > 1:
        params["csv_filename"] = sys.argv[1]
    if len(sys.argv) > 2:
        params["min_relative_time"] = float(sys.argv[2])
    if len(sys.argv) > 3:
        params["min_marginal_speedup"] = float(sys.argv[3])
    if len(sys.argv) > 4:
        params["min_efficiency"] = float(sys.argv[4])

    plotResults(**(params))
