import os
import random
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.integrate import odeint
from scipy.linalg import block_diag
from tqdm import tqdm

with open("config_simu.yml", "r") as f:
    config_simu = yaml.safe_load(f)

a = config_simu["a"]
b = config_simu["b"]
c = config_simu["c"]

alpha = config_simu["alpha"]
beta = config_simu["beta"]
delta = config_simu["delta"]

L = config_simu["L"]
J = config_simu["J"]
T = config_simu["T"]
N = config_simu["N"]

freq = config_simu["freq"]
minree = config_simu["minree"]
maxree = config_simu["maxree"]
# p = config_simu["p"]
intensity = config_simu["intensity"]


# Calcul des paramètre sdérivés
dx = L / J
dy = L / J
dt = T / N
dtau = dt / 2
kx = delta * dtau / dx**2
ky = delta * dtau / dy**2


T_linspace = np.linspace(start=0, stop=T - 1, num=T, endpoint=True)

uplus = b + np.sqrt((alpha * delta - c) / a)
umoins = b - np.sqrt((alpha * delta - c) / a)
wplus = (alpha / beta) * uplus
wmoins = (alpha / beta) * umoins

persistence_vector = np.ones((J * J, 2)) * (uplus, wplus)
extinction_vector = np.zeros_like(persistence_vector)

# Utils #######


def n1(matrice):
    """norme 1 d'une matrice ie somme des abs"""
    return np.sum([np.abs(v) for v in matrice.reshape(-1)])


def gen_init(center):

    radius = 750
    low = -1

    if n1(center) == 0:
        # travail depuis extinction -> demi-boule positive
        low = 0

    noise = np.random.uniform(low, 1, center.shape)
    u0 = center + noise / n1(noise) * radius * random.random()
    # print(u0)
    # print(n1(u0 - center))

    return u0


# Simulations #######


def run_simulation(nb_simulations, sim_id="default", center=persistence_vector, p=0.0):

    def q(u):
        return u * (a * (u - b) ** 2 + c)

    def reaction(X, dt_):  # forcé de plaçer dt --> cf. odeint
        u, w = X

        du = beta * delta * w - q(u)
        dw = alpha * u - beta * w

        return [du, dw]

    save_folder = os.path.join("Simulations", sim_id)
    if os.path.exists(save_folder):
        shutil.rmtree(save_folder)
    os.makedirs(save_folder, exist_ok=True)

    # Matrices du schéma : boreale
    diag = np.ones(J)
    diagsup = np.ones(J - 1)
    D = (
        np.diag(diag * (1 + 2 * (kx + ky)), 0)
        + np.diag(diagsup * (-ky), 1)
        + np.diag(diagsup * (-ky), -1)
    )
    A = block_diag(D)
    for i in range(J - 1):
        A = block_diag(A, D)

    # Conditions au bord de Neumann
    for k in [0, J - 1, J * J - J, J * J - 1]:
        A[k][k] = 1 + kx + ky
    for n in range(1, J - 1):
        n1 = J * n
        n2 = J * n + J - 1
        A[n1][n1] = 1 + 2 * kx + ky
        A[n2][n2] = 1 + 2 * kx + ky
    for k in range(1, J - 1):
        A[k][k] = 1 + kx + 2 * ky
    for k in range(J * J - J + 1, J * J - 1):
        A[k][k] = 1 + kx + 2 * ky

    grandediag = np.ones(J * (J - 1))
    A = A + np.diag(grandediag * (-ky), J) + np.diag(grandediag * (-ky), -J)

    # Calculs
    # print("Inversion de matrices...")
    invA = np.linalg.inv(A)

    for isim in tqdm(range(nb_simulations)):

        dat_file = os.path.join(save_folder, "dat_" + str(isim) + ".csv")
        init = gen_init(center)

        u = init[:, 0]
        w = init[:, 1]

        # print('Calculs en cours...')
        file = open(dat_file, "w")
        file.write("t i j u w \n")

        # écriture fichier
        for t in range(T):
            for i in range(J):
                for j in range(J):
                    file.write(
                        str(t)
                        + " "
                        + str(i)
                        + " "
                        + str(j)
                        + " "
                        + str(u[i * J + j])
                        + " "
                        + str(w[i * J + j])
                        + "\n"
                    )

            # Méthode de Strang
            # diffusion 1/2 pas
            u = np.dot(invA, u)
            w = np.dot(invA, w)

            # réaction 1 pas
            for i in range(J):
                for j in range(J):
                    idx = i * J + j
                    X0 = [u[idx], w[idx]]
                    orbit = odeint(
                        reaction,
                        X0,
                        [0, dt],
                    )
                    u[idx], w[idx] = orbit[-1]

            # diffusion 1/2 pas
            u = np.dot(invA, u)
            w = np.dot(invA, w)

            # Partie résolution de feux

            if t % freq == 0:

                if np.random.binomial(1, p) == 1:

                    xee = random.random() * J
                    yee = random.random() * J

                    ree = minree + random.random() * (maxree - minree)

                    var_ee = lambda i, j, var: (
                        var[i * J + j] * (1 - intensity)
                        if ((i - xee) ** 2 + (j - yee) ** 2) < ree**2
                        else var[i * J + j]
                    )

                    for i in range(J):
                        for j in range(J):
                            u[i * J + j] = var_ee(i, j, u)
                            w[i * J + j] = var_ee(i, j, w)


# Plots #######


def plot_folder_distances(sim_id="default", center=persistence_vector, hline_norm=0):

    fig, ax = plt.subplots()
    fig.suptitle("Distance from extinction state over time (1-norm)")

    sim_folder = os.path.join("Simulations", sim_id)
    for sim_path in tqdm(os.listdir(sim_folder)):

        sim_path = os.path.join(sim_folder, sim_path)

        df = pd.read_csv(
            sim_path,
            header=0,
            names=["t", "i", "j", "u", "v"],
            sep=" ",
            index_col=False,
        )

        trace = []
        try:
            if n1(center - df[df["t"] == 0].iloc[:, 3:].to_numpy()) < hline_norm:

                for t in T_linspace:
                    Ut = df[df["t"] == t].iloc[:, 3:]
                    Ut = Ut.to_numpy()

                    distance_center = n1(center - Ut)
                    trace.append(distance_center)

                ax.plot(T_linspace, trace)

        except ValueError:
            # simus avortées ou fichiers non simu
            # print(sim_path)
            pass

    ax.axhline(hline_norm, linestyle="--")
    plt.savefig(os.path.join("plots", sim_id))


# Calculs


def compute_probs(sim_id="default", center=persistence_vector, b_radius=500):

    P_out = 0
    P_stay = 0
    nb_of_sims = 0

    sim_folder = os.path.join("Simulations", sim_id)
    for sim_path in tqdm(os.listdir(sim_folder)):

        sim_path = os.path.join(sim_folder, sim_path)

        df = pd.read_csv(
            sim_path,
            header=0,
            names=["t", "i", "j", "u", "v"],
            sep=" ",
            index_col=False,
        )

        # Ne considérer que les simus qui partent de la boule
        if n1(center - df[df["t"] == 0].iloc[:, 3:].to_numpy()) < b_radius:

            # compter les simus concernées
            nb_of_sims += 1

            # considérer l'état final après une période (tau-1)
            Unext = df[df["t"] == T - 1].iloc[:, 3:].to_numpy()
            distance_center = n1(Unext - center)

            if distance_center < b_radius:
                P_stay += 1
            else:
                P_out += 1

    P_out /= nb_of_sims
    P_stay /= nb_of_sims

    # print("P_out", P_out)
    # print("P_stay", P_stay)

    return P_stay, P_out


# Quelques exemples d'usage :

# run_simulation(nb_simulations=740, sim_id="def", center=persistence_vector)
# compute_probs("e740_nf", center=extinction_vector, b_radius=500)
# plot_folder_distances("e740_nf", center=extinction_vector, hline_norm=500)
