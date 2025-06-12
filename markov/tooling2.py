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
p = config_simu["p"]
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


RADIUS_BALLS = 0.6


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
    u0 = np.random.uniform(RADIUS_BALLS, uplus - RADIUS_BALLS, J * J)
    w0 = np.random.uniform(RADIUS_BALLS, wplus - RADIUS_BALLS, J * J)
    u_init = np.stack((u_0, w_0), a_xis=1)
    return u_init
