import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import sys


def plotResults(csv_filename = "../data/results.csv"):
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

    # Normalize efficiency for colormap (between 0 and 1)
    norm = mcolors.Normalize(vmin=0, vmax=1)
    cmap = plt.get_cmap("RdYlGn")  # green = good efficiency, red = bad efficiency
    
    
    for (image_dim, kernel_dim), subgroup in grouped.groupby(["ImageDimension", "KernelDimension"]):
        subgroup = subgroup.sort_values("NumThreads")
    
        fig, ax1 = plt.subplots(figsize=(8,6))
    
        # First axis for average time
        ax1.plot(
            subgroup["NumThreads"],
            subgroup["TimePerRep_s"],
            marker="o",
            linestyle="-",
            label="time"
        )
                
        # Scatter with efficiency-based color
        ax1.scatter(
            subgroup["NumThreads"],
            subgroup["TimePerRep_s"],
            c=subgroup["Efficiency"].apply(lambda s: cmap(norm(min(s,1)))),
            s=100
        )
        for x, y, sp in zip(subgroup["NumThreads"], subgroup["TimePerRep_s"], subgroup["Efficiency"]):
            ax1.text(x, y, f"{sp:.2f}", fontsize=8, ha="center", va="bottom")
            
        ax1.set_xlabel("Threads number")
        ax1.set_ylabel("Average time (s)")
        h1, l1 = ax1.get_legend_handles_labels()
    
    
        # Second axis for speedup
        ax2 = ax1.twinx()
        ax2.plot(
            subgroup["NumThreads"],
            subgroup["SpeedUp"],
            marker="s",
            linestyle="--",
            color="green",
            label="speedup"
        )
        for x, y in zip(subgroup["NumThreads"], subgroup["SpeedUp"]):
            ax2.text(x, y, f"{y:.2f}", fontsize=8, ha="center", va="bottom")
        
        ax2.set_ylabel("Average speedup")
        h2, l2 = ax2.get_legend_handles_labels()
    
    
        plt.title(f"Strong scaling: {image_dim} images | {kernel_dim}x{kernel_dim} kernels")
        plt.legend(h1+h2, l1+l2, loc="center right")
        plt.grid(True, linestyle="-", alpha=0.6)
        plt.tight_layout()
        
        
        # Color bar for efficiency reference 
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=[ax1, ax2], fraction=0.04, pad=0.04)
        cbar.set_label("Efficiency")
        
        
        filename = f"Strong_scaling_{image_dim}_{kernel_dim}.png".replace("x", "x")
        plt.savefig(filename, dpi=150)
        plt.show()
        plt.close()


if __name__ == "__main__":
    params = {}
    if len(sys.argv) > 1:
        params["csv_filename"] = sys.argv[1]

    plotResults(**(params))