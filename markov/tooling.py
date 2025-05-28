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


# Calcul des paramètres dérivés
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


radius_balls = 0.6
# Utils #######

def ninf(matrice):
    """ norme inf d'une matrice : max des abs
    """
    return np.max(np.abs(matrice))

def gen_init_persistence():
    global radius_balls
    u_0 = np.random.uniform(uplus-radius_balls, uplus, J*J)
    w_0 = np.random.uniform(uplus-radius_balls, wplus, J*J)
    u_init = np.stack((u0, w0), axis=1)
    return u0
    
def gen_init_extinction():
    global radius_balls
    u_0 = np.random.uniform(0, radius_balls, J*J)
    w_0 = np.random.uniform(0, radius_balls, J*J)
    u_init = np.stack((u0, w0), axis=1)
    return u_init

def gen_init_intermed():
    global radius_balls
    u0 = np.random.uniform(radius_balls, uplus-radius_balls, J*J)
    w0 = np.random.uniform(radius_balls, wplus-radius_balls, J*J)
    u_init = np.stack((u0, w0), axis=1)
    return u_init        

# Simulations #######


def run_simulation(nb_simulations, sim_id="default", init_state="p", p=0.0):

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
    invA = np.linalg.inv(A)

    for isim in tqdm(range(nb_simulations)):

        dat_file = os.path.join(save_folder, "dat_" + str(isim) + ".csv")

        # construction de situation initiale en fonction de l'état
        if init_state == "p":
            u_init = gen_init_persistence()
        elif init_state == "e":
            u_init = gen_init_extinction()
        elif init_state == "i":
            u_init = gen_init_intermed()
        else:
            raise ValueError(f"Selected initial state {init_state} is not valid.")

        u = u_init[:, 0]
        w = u_init[:, 1]

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


# Plots

def plot_distances(sim_id="default", center="persistence"): 
    global radius_balls

    fig, ax = plt.subplots()
    fig.suptitle(f"Distance from {center} state over time (inf-norm)")

    if center=="persistence":
        center_vector = persistence_vector
    elif center=="extinction":
        center_vector = extinction_vector
    else:
        raise ValueError(f"Selected center {center} is not valid.")


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

            for t in T_linspace:
                Ut = df[df["t"] == t].iloc[:, 3:]
                Ut = Ut.to_numpy()

                distance_center = ninf(center_vector - Ut)
                trace.append(distance_center)

            ax.plot(T_linspace, trace)

        except ValueError:
            # simus avortées ou fichiers non simu
            # print(sim_path)
            print("error on file", sim_path)
            pass

    print(trace)
    ax.axhline(radius_balls, linestyle="--")
    return fig


def plot_intermeds(sim_id="default_intermed"):
    """Dessiner en deux plots la distance"""

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    fig.suptitle("Rendering of distance to stable states over time.")
    ax.set_xlabel("time")
    ax.set_ylabel("D(persistence)")
    ax.set_zlabel("D(extinction)")

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

        trace_from_e = []
        trace_from_p = []

        for t in T_linspace:
            Ut = df[df["t"] == t].iloc[:, 3:]
            Ut = Ut.to_numpy()

            trace_from_e.append(n1(Ut - extinction_vector))
            trace_from_p.append(n1(Ut - persistence_vector))

        ax.plot(T_linspace, trace_from_p, trace_from_e)
    plt.show()


# Calculs


def compute_probs(sim_id="default", center=persistence_vector, b_radius=500):

    P_out = 0  # sortir de la boule vers le centre
    P_stay = 0  # rester dans la boule de départ
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

    return P_stay, P_out


def compute_probs_intermed(b_radius=500, sim_id="default_intermed"):
    """Ici, b_radius désigne le rayon des boules Persistence et Extinction.
    On calcule la proba de sortir de l'état intermédiaire en 'tau' temps.
    """

    P_to_persistence = 0  # sortir de la boule vers la Persistence
    P_to_extinction = 0  # sortir de la boule vers l'extinction
    P_stay = 0  # rester dans zone intermédiaire
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

        u0 = df[df["t"] == 0].iloc[:, 3:].to_numpy()
        uTau = df[df["t"] == T - 1].iloc[:, 3:].to_numpy()
        start_distances = n1(persistence_vector - u0), n1(extinction_vector - u0)

        if start_distances[0] > b_radius and start_distances[1] > b_radius:
            # u0 bien généré : pas dans une des deux boules
            nb_of_sims += 1

            # considérer l'état final après une période (tau-1)
            end_p_distance = n1(uTau - persistence_vector)
            end_e_distance = n1(uTau - extinction_vector)

            if end_p_distance < 500:
                P_to_persistence += 1
            elif end_e_distance < 500:
                P_to_extinction += 1
            else:
                P_stay += 1

    P_stay /= nb_of_sims
    P_to_extinction /= nb_of_sims
    P_to_persistence /= nb_of_sims

    return P_stay, P_to_extinction, P_to_persistence




# Quelques exemples d'usage :

# run_simulation(nb_simulations=740, sim_id="def", init_state=persistence_vector)
# compute_probs("e740_nf", init_state=extinction_vector, b_radius=500)
# plot_folder_distances("e740_nf", init_state=extinction_vector, hline_norm=500)
# run_simulation(740, sim_id="Inter", init_state="i")
run_simulation(200, sim_id="Inter_Feux", init_state="i", p=1.0)
fig1 = plot_distances("Inter_Feux", center="persistence")
fig2 = plot_distances("Inter_Feux", center="extinction")
plt.show()
