import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys

from simu import norm1, Ueq, T_linspace

def plot_folder(folder_path="Simulations"):
    
    traces = []
    fig, ax = plt.subplots(figsize=(5, 8))
    persistence_norm = norm1(Ueq)

    print("Persistence norm: ", persistence_norm)
    print("Ueq: ", Ueq)

    ax.set_ylim(0, persistence_norm*1.2)
    ax.axhline(y=float(persistence_norm), color='green', linestyle='--')  # ligne "persistence"
    # ax.invert_yaxis()
    ax.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    ax.set_xlabel("time")
    ax.set_title('Distance from extinction')
    print("Tlinspace is ", T_linspace)
    for sim in os.listdir(folder_path + "/Data"):
        
        # print("UPDATE PLOT: with ", str(sim))
        df = pd.read_csv(folder_path + "/Data/" + sim, names=["t", "x", "y", "u", "w"],
                         header=0, sep=" ", index_col=False)
        trace = []
        
        # print(np.linalg.norm(Ueq - df[df['t'] == 0].iloc[:, -2:]))
        # pour chaque temps, calculer les normes
        
        for i in T_linspace:
            Ut = df[df['t'] == i].iloc[:, 3:5]
            Ut = Ut.to_numpy()
            # print("t", i)
            # print("Ut", Ut)
            trace.append(norm1(Ut))
        
        ax.plot(T_linspace, trace)
    
    plt.show()


if __name__ == '__main__':
    plot_folder(input("Dossier à figurer : "))
