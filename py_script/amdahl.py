import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
import sys


def plotStrongScaling(csv_filename = "../data/results.csv", min_relative_time = 0.05, 
                min_marginal_speedup = 0.2, min_efficiency = 0.5):
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
    
    # Main Loop (group by same data size)
    for (image_dim, kernel_dim), subgroup in grouped.groupby(["ImageDimension", "KernelDimension"]):
        subgroup = subgroup.sort_values("NumThreads")
        
        ### FIRST PART: Show strong scaling, i.e. Time vs SpeedUp vs Efficiency
        fig, ax1 = plt.subplots(figsize=(8,6))
    
        # First axis for time
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
        
        filename = f"strong_scaling_{image_dim}_{kernel_dim}.png".replace("x", "x")
        plt.savefig(filename, dpi=150)
        plt.show()
        plt.close()
        
        
        
        
        ### SECOND PART: Amdahl valuation
        multithread_values = subgroup[subgroup["NumThreads"] > 1]
        
        ### Spiegazione dati ###
        # Col fit lineare (f) vengono usati tutti i punti e riduce il rumore della singola stima.
        # Se f decresce dopo ottimizzazioni (meno sync, migliore bilanciamento, migliore locality),
        # stai recuperando margine reale.
        # Una volta calcolato f, confronta lo speedup osservato con lo speedup massimo 
        # previsto da Amdahl usando il f stimato: S_max = 1 / (f + ((1-f) / p))
        # Se lo speedup misurato ≈ S_max, sei vicino al limite per quell’hardware e quel codice.
        
        x = 1 / multithread_values["NumThreads"].values.reshape(-1, 1)
        y = 1 / multithread_values["SpeedUp"].values.reshape(-1, 1)
    
        model = LinearRegression(fit_intercept=True)
        model.fit(x, y)
        f_est = model.intercept_[0]  # intersection axis y = estimation of f
        slope = model.coef_[0][0]
        
        print(f"\nCoppia ({image_dim}, kernel={kernel_dim}):")
        print(f"  f stimato = {f_est:.4f} (intersezione), coeff angolare={slope:.4f}")

        ### Spiegazione dati ###
        # La serial fraction di Karp–Flatt (f_p) include anche l’overhead parallelo percepito
        # (sincronizzazioni, imbalance, ecc.).
        # Se f_p è circa costante per vari p, il modello spiega bene i dati; 
        # se cresce con p, hai overhead/collo di bottiglia che aumentano con i thread 
        # (sync, memoria, false sharing, ecc.).
        
        f_p_table = []
        for p, s in zip(multithread_values["NumThreads"], multithread_values["SpeedUp"]):
            f_p = (1/s - 1/p) / (1 - 1/p)
            f_p_table.append((p, s, f_p))
    
        f_p_df = pd.DataFrame(f_p_table, columns=["NumThreads", "SpeedUp_avg", "f_p"])
        print("Tabella f_p (per singolo p):")
        print(f_p_df.to_string(index=False, float_format="%.4f"))
        
        
        # Diagram: linear fit vs f_p vs speedup data
        plt.figure(figsize=(8,6))
        # Amdahl curve evaluated by Linear Fit
        p_range = np.linspace(1, multithread_values["NumThreads"].max(), 200)
        lin_fit_speedup = 1 / (f_est + (1-f_est)/p_range)
        plt.plot(p_range, lin_fit_speedup, "-", color="red", 
                 label=f"linear fit (f ≈ {f_est:.3f})")
        # Amdahl curves evaluated by f_p
        for (p, _, f_p) in f_p_table:
            p = int(p)
            f_p_speedup = 1 / (f_p + (1-f_p)/p_range)
            plt.plot(p_range, f_p_speedup, "--", label=f"p={p} (f ≈ {f_p:.3f})")
        # speedup data
        plt.scatter(multithread_values["NumThreads"], multithread_values["SpeedUp"], 
                    label="experimental speedup", color="blue", s=60)
                    
        ### Interpretare il grafico ###
        # - Se f_p oscillano attorno a un valore stabile, vuol dire che la stima 
        #   di f è robusta e coerente.
        # - Se invece cambiano molto con p:
        #   - ai valori piccoli di p: variazioni possono essere dovute al rumore
        #     di misura (pochi thread → tempi vicini, difficile distinguere).
        #   - ai valori grandi di p: variazioni indicano overhead crescenti non
        #     catturati dal modello di Amdahl (es. costi di sincronizzazione 
        #     che crescono con i thread).
        #
        # - Punti blu sulla curva rossa → il modello spiega bene i dati
        #   Significa che il programma segue piuttosto fedelmente la legge di Amdahl
        #   con quella frazione sequenziale stimata. 
        #   In pratica: la scalabilità osservata è ben descritta dalla parte sequenziale fissa.
        #
        # - Punti blu sopra la curva rossa → performance migliore del previsto
        #   Può succedere se con più thread ci sono effetti collaterali positivi 
        #   (es. caching, NUMA locality, riduzione di overhead non lineari).
        #   Se succede, è un segnale che Amdahl è un modello troppo pessimista per quel codice.
        #
        # - Punti blu sotto la curva rossa → performance peggiore del previsto
        #   Significa che oltre al limite teorico di Amdahl ci sono altri overhead:
        #   sincronizzazioni, cache misses, false sharing, scheduling, I/O, ecc.
        #   La curva rossa diventa così un “limite superiore ideale”.
        
        plt.xlabel("NumThreads (p)")
        plt.ylabel("SpeedUp")
        plt.title(f"Amdahl curve evaluation: {image_dim} images | {kernel_dim}x{kernel_dim} kernels")
        plt.legend(loc="best")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        
        filename = f"amdahl_evaluation_{image_dim}_{kernel_dim}.png".replace("x", "x")
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

    plotStrongScaling(**(params))
