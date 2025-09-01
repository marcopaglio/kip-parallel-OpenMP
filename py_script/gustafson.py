import pandas as pd
import matplotlib.pyplot as plt
import sys


# csv_filename              relative path to the .cvs file to analyze.
# min_efficiency            minimum weak efficiency value in order to consider it 
#                           as a good value (bad values are printed in red).
# max_relative_time         maximum distance (in fraction) from the sequential time
#                           in order to consider it as a good time (bad values
#                           are printed in red).
#
def plotWeakScaling(csv_filename = "../data/kip_openMP_weakScaling.csv", 
                    min_efficiency = 0.7, max_relative_time = 1.3):
    min_relative_throughput = min_efficiency
    
    df = pd.read_csv(csv_filename)
    
    # Group same-size-images-and-kernels and calculate means
    grouped = (
        df.groupby(["UnitOfWork", "KernelDimension", "NumThreads"])
          .agg({
              "TimePerRep_s": "mean", 
              "WeakEfficiency": "mean", 
              "ScaledSpeedUp": "mean", 
              "Throughput_Mpix_s": "mean"
          })
          .reset_index()
    )
    
    # Main Loop (group by same data size)
    for (unit_of_work, kernel_dim), subgroup in grouped.groupby(["UnitOfWork", "KernelDimension"]):
        subgroup = subgroup.sort_values("NumThreads")
        
        ### FIRST PART: Gustafson - Scaled Speedup
        plt.figure(figsize=(7,5))
        plt_color = "green"
        plt.plot(
            subgroup["NumThreads"], 
            subgroup["ScaledSpeedUp"], 
            marker="o",
            linestyle="-",
            color=plt_color, 
            label="real speedup",
            markersize=6
        )
        for x, y in zip(subgroup["NumThreads"], subgroup["ScaledSpeedUp"]):
            plt.annotate(f"{y:.2f}", (x, y), xytext=(-10, -5), textcoords="offset points",
                ha="center", va="top", fontsize=8)
        
        plt.plot(
            subgroup["NumThreads"],
            subgroup["NumThreads"],
            linestyle="--",
            color=plt_color,
            alpha=0.4,
            label="ideal speedup (y=p)"
        )
        
        plt.xlabel("Threads number (p)")
        plt.ylabel("Scaled Speedup")
        plt.title(f"Gustafson's law evaluation: W₀ = {unit_of_work} images | {kernel_dim}x{kernel_dim} kernels")
        plt.legend(loc="best")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        
        filename = f"gustafson_evaluation_{unit_of_work}_{kernel_dim}.png".replace("x", "x")
        plt.savefig(filename, dpi=150)
        plt.show()
        plt.close()
        
        ### Interpretazione del grafico
        # Conferma se il programma scala “come dovrebbe” aumentando il problema
        # insieme ai thread.
        # - Se il grafico mostra un distacco crescente dalla linea ideale 
        #   → l’algoritmo o l’implementazione va rivista (e.g. ridurre comunicazioni o sincronizzazioni).
        
        
        
        ### SECOND PART: Show weak scaling, i.e. Efficiency vs Throughput vs Time
        fig, ax1 = plt.subplots(figsize=(9,6))

        # First axis for Weak Efficiency (misura la vicinanza al comportamento ideale, i.e efficiency=1)
        ax1_color="#1f77b4"
        ax1.set_xlabel("Threads number (p)")
        ax1.set_ylabel("Weak Efficiency", color=ax1_color)
        
        ax1.plot(
            subgroup["NumThreads"], 
            subgroup["WeakEfficiency"], 
            marker="s",
            linestyle="-",
            color=ax1_color,
            label="real efficiency",
            markersize=6
        )
        for x, y in zip(subgroup["NumThreads"], subgroup["WeakEfficiency"]):
            if y < min_efficiency:
                color = "red"
            else:
                color = "black"
            ax1.annotate(f"{y:.2f}", (x, y), xytext=(-10, -5), textcoords="offset points",
                ha="center", va="top", fontsize=8, color=color)
        
        ax1.axhline(
            y=1,
            color=ax1_color,
            alpha=0.4,
            linestyle="--",
            label="ideal efficiency (y=1)"
        )
        ax1.tick_params(axis="y", labelcolor=ax1_color)
        h1, l1 = ax1.get_legend_handles_labels()      
        
        ### Intrepretazione del grafico
        # - Se rimane stabile (vicino a 1) con l’aumentare dei thread 
        #   → buon scaling.
        # - Se scende bruscamente 
        #   → aggiungere thread non porta benefici reali
        #   → ci sono colli di bottiglia seri (memoria, NUMA, lock).

    
        # Second axis for Throughput (elaborazione per unità di tempo.)
        ax2 = ax1.twinx()
        ax2_color = "purple"
        ax2.set_ylabel("Throughput (Mpix/s)", color=ax2_color)
        
        ax2.plot(
            subgroup["NumThreads"],
            subgroup["Throughput_Mpix_s"],
            marker="^",
            linestyle="-",
            color=ax2_color, 
            label="real throughput",
            markersize=6
        )
        for x, y in zip( subgroup["NumThreads"], subgroup["Throughput_Mpix_s"]):
            if y / (x * subgroup[subgroup["NumThreads"] == 1]["Throughput_Mpix_s"].item()) < min_relative_throughput:
                color = "red"
            else:
                color = "black"
            ax2.annotate(f"{y:.2f}", (x, y), xytext=(-10, -5), textcoords="offset points",
                ha="center", va="top", fontsize=8, color=color)
            
        ax2.plot(
            subgroup["NumThreads"],
            subgroup["NumThreads"] * subgroup[subgroup["NumThreads"] == 1]["Throughput_Mpix_s"].item(),
            linestyle="--",
            color=ax2_color,
            alpha=0.4,
            label="ideal throughput (y=p*throughput₁)"
        )
        ax2.tick_params(axis="y", labelcolor=ax2_color)
        
        ### Interpretazione del grafico
        # - Se cresce quasi linearmente con i thread (vicino alla linea ideale) 
        #   → la parallelizzazione è efficace.
        # - Se si appiattisce o cala 
        #   → overhead, limiti di banda di memoria, o saturazione delle risorse.
        
        
        # Third axis for Time
        ax2.scatter(
            subgroup["NumThreads"],
            subgroup["Throughput_Mpix_s"],
            c="palegoldenrod",
            s= 100 + 60 * subgroup["TimePerRep_s"].apply(lambda s: s),
            label="time"
        )
        
        ax2_opposite = ax2.twiny()
        xticks = list([round(item,2) for item in subgroup["TimePerRep_s"]])    
        ax2_opposite.set_xlim(ax2.get_xlim())
        ax2_opposite.set_xticks(list(subgroup["NumThreads"]))
        ax2_opposite.set_xticklabels(xticks)
        for label in ax2_opposite.get_xticklabels():
            if float(label.get_text()) > max_relative_time * subgroup[subgroup["NumThreads"] == 1]["TimePerRep_s"].item():
                label.set_color("red");
        ax2_opposite.set_xlabel("Time (s)")
        ax2_opposite.grid(True, axis="x", linestyle="--", alpha=0.6)
        h2, l2 = ax2.get_legend_handles_labels()
        
        ### Interpretazione del grafico
        # Idealmente il tempo dovrebbe restare costante al crescere dei thread 
        # perché ogni thread lavora sulla stessa unità di lavoro W₀.
        # - Se cresce:
        #   → overhead di comunicazione/sincronizzazione o saturazione di banda memoria.
        # - Se decresce:
        #   → qualche effetto collaterale (cache locality, schedulazione più efficiente, ecc.),
        #   ma è raro e spesso sospetto.    
    
        plt.title(f"Weak Scaling: W₀ = {unit_of_work} images | {kernel_dim}x{kernel_dim} kernels")
        plt.legend(h1+h2, l1+l2, loc="best")
        plt.tight_layout()
        
        filename = f"weak_scaling_{unit_of_work}_{kernel_dim}.png".replace("x", "x")
        plt.savefig(filename, dpi=150)
        plt.show()
        plt.close()
        
        ### Interpretazione del grafico
        # Mostra quanto lavoro in più puoi trattare aumentando i thread.
        # Se rimane vicino alla retta, significa che il programma scala bene.
        
        


if __name__ == "__main__":
    params = {}
    if len(sys.argv) > 1:
        params["csv_filename"] = sys.argv[1]

    plotWeakScaling(**(params))