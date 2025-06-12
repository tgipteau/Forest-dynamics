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

# TOOLS

RADIUS_BALLS = 0.6
NB_SIMULATIONS = 1200


def ninf(matrice):
    """norme inf d'une matrice : max des abs"""
    return np.max(np.abs(matrice))


def gen_init_persistence():
    u_0 = np.random.uniform(uplus - RADIUS_BALLS, uplus, J * J)
    w_0 = np.random.uniform(uplus - RADIUS_BALLS, wplus, J * J)
    u_init = np.stack((u_0, w_0), axis=1)
    return u_init


def gen_init_extinction():
    u_0 = np.random.uniform(0, RADIUS_BALLS, J * J)
    w_0 = np.random.uniform(0, RADIUS_BALLS, J * J)
    u_init = np.stack((u_0, w_0), axis=1)
    return u_init


def gen_init_intermed():
    u_0 = np.random.uniform(RADIUS_BALLS, uplus - RADIUS_BALLS, J * J)
    w_0 = np.random.uniform(RADIUS_BALLS, wplus - RADIUS_BALLS, J * J)
    u_init = np.stack((u_0, w_0), axis=1)
    return u_init


def run_simulation_with_params(nb_simulations, sim_id="default", init_state="p", p=0.0):

    def q(u):
        return u * (a * (u - b) ** 2 + c)

    def reaction(X, dt_):  # forcé de plaçer dt --> cf. odeint
        u, w = X

        du = beta * delta * w - q(u)
        dw = alpha * u - beta * w

        return [du, dw]

    # Delete and recreate the save folder
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


# CONFIGURATIONS SIMULATIONS


def run_persistence_no_fire():
    sim_id = "persistence_no_fire"
    run_simulation_with_params(
        nb_simulations=NB_SIMULATIONS,
        sim_id=sim_id,
        init_state="p",
        p=0.0,
    )


def run_extinction_no_fire():
    sim_id = "extinction_no_fire"
    run_simulation_with_params(
        nb_simulations=NB_SIMULATIONS,
        sim_id=sim_id,
        init_state="e",
        p=0.0,
    )


def run_intermed_no_fire():
    sim_id = "intermed_no_fire"
    run_simulation_with_params(
        nb_simulations=NB_SIMULATIONS,
        sim_id=sim_id,
        init_state="i",
        p=0.0,
    )


def run_persistence_with_fire():
    sim_id = "persistence_with_fire"
    run_simulation_with_params(
        nb_simulations=NB_SIMULATIONS,
        sim_id=sim_id,
        init_state="p",
        p=1.0,
    )


def run_extinction_with_fire():
    sim_id = "extinction_with_fire"
    run_simulation_with_params(
        nb_simulations=NB_SIMULATIONS,
        sim_id=sim_id,
        init_state="e",
        p=1.0,
    )


def run_intermed_with_fire():
    sim_id = "intermed_with_fire"
    run_simulation_with_params(
        nb_simulations=NB_SIMULATIONS,
        sim_id=sim_id,
        init_state="i",
        p=1.0,
    )


# COMPUTE PROBABILITIES


def compute_probs_with_parameters(sim_id="default", starting_from=p, with_fire=False):

    P_out = 0  # sortir de la boule
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

        nb_of_sims += 1

        if not with_fire:
            # considérer l'état final après une période (tau-1)
            Unext = df[df["t"] == T - 1].iloc[:, 3:].to_numpy()
        elif with_fire:
            # considérer l'état après un seul pas de temps
            Unext = df[df["t"] == 1].iloc[:, 3:].to_numpy()

        if starting_from == "p":

            def is_in_ball(u):
                return ninf(u - persistence_vector) < RADIUS_BALLS

        elif starting_from == "e":

            def is_in_ball(u):
                return ninf(u - extinction_vector) < RADIUS_BALLS

        elif starting_from == "i":

            def is_in_ball(u):
                return (
                    ninf(u - persistence_vector) > RADIUS_BALLS
                    and ninf(u - extinction_vector) > RADIUS_BALLS
                )

        if is_in_ball(Unext):
            P_stay += 1
        else:
            P_out += 1

    # Normalisation des probabilités
    P_out /= nb_of_sims
    P_stay /= nb_of_sims

    print(
        f"Simulation {sim_id} with fire={with_fire} and starting from {starting_from}:"
    )
    print(f"  P_out: {P_out}, P_stay: {P_stay}")


def compute_persistence_with_fire():
    return compute_probs_with_parameters(
        sim_id="persistence_with_fire", starting_from="p", with_fire=True
    )


def compute_extinction_with_fire():
    return compute_probs_with_parameters(
        sim_id="extinction_with_fire", starting_from="e", with_fire=True
    )


def compute_intermed_with_fire():
    return compute_probs_with_parameters(
        sim_id="intermed_with_fire", starting_from="i", with_fire=True
    )


def compute_persistence_no_fire():
    return compute_probs_with_parameters(
        sim_id="persistence_no_fire", starting_from="p", with_fire=False
    )


def compute_extinction_no_fire():
    return compute_probs_with_parameters(
        sim_id="extinction_no_fire", starting_from="e", with_fire=False
    )


def compute_intermed_no_fire():
    return compute_probs_with_parameters(
        sim_id="intermed_no_fire", starting_from="i", with_fire=False
    )


# MAIN FUNCTIONS


def run_all_simulations():
    run_persistence_no_fire()
    run_extinction_no_fire()
    run_intermed_no_fire()
    run_persistence_with_fire()
    run_extinction_with_fire()
    run_intermed_with_fire()


def compute_all_simulations():
    compute_persistence_no_fire()
    compute_extinction_no_fire()
    compute_intermed_no_fire()
    compute_persistence_with_fire()
    compute_extinction_with_fire()
    compute_intermed_with_fire()


if __name__ == "__main__":

    run_all_simulations()
