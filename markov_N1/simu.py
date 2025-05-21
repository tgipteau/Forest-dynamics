import os
import numpy as np
import random
from scipy.linalg import block_diag
from tqdm import tqdm
from scipy.integrate import odeint
import sys
import random

args = sys.argv

a = 1
b = 1
c = 0.01

# boreale
alpha = 2
beta = 1
delta = 0.55

# discrétisation
L = 20
J = 40
T = 50
N = 1000
freq = 1
minree = 2
maxree = 10

p = 0.0 
intensity = 0.7

# Calcul des paramètre sdérivés
dx = L / J
dy = L / J
dt = T / N
dtau = dt / 2
kx = delta * dtau / dx ** 2
ky = delta * dtau / dy ** 2

T_linspace = np.linspace(start=0, stop=T-1, num=T, endpoint=True)

uplus = b + np.sqrt((alpha * delta - c) / a)
umoins = b - np.sqrt((alpha * delta - c) / a)
wplus = (alpha / beta) * uplus
wmoins = (alpha / beta) * umoins

Ueq = np.array([uplus, wplus])
Ueq = np.repeat(Ueq, J * J, axis=0).reshape(J * J, 2)

Um = np.array([umoins, wmoins])
Um = np.repeat(Um, J * J, axis=0).reshape(J * J, 2)


def norm1(matrix: np.array):
    n1 = 0
    for row in matrix:
        for val in row:
            n1 += np.abs(val)
    
    return n1


def run_simulation(nb_simulations, ballcenter, ballradius, save_folder="Simulations"):
    
    def q(u):
        return u * (a * (u - b) ** 2 + c)
    
    def reaction(X, t):
        u, w = X
         
        du = beta * delta * w - q(u)
        dw = alpha * u - beta * w
        
        return [du, dw]
    
    def starter():
        
        center = np.repeat(ballcenter, J * J).reshape(2, J*J)
       
        noise = np.random.uniform(-1, 1, (2,J*J))
        #print("noise.shape : ", noise.shape)
        noise /= norm1(noise)
        noise *= random.random() * ballradius
        
        starter_ = center + noise
        starter_ = np.maximum(np.zeros_like(starter_), starter_)
        return starter_

    os.makedirs(save_folder, exist_ok=True)
    os.makedirs(os.path.join(save_folder, "Fires"), exist_ok=True)
    os.makedirs(os.path.join(save_folder, "Data"), exist_ok=True)
    
    # Matrices du schéma : boreale
    diag = np.ones(J)
    diagsup = np.ones(J - 1)
    D = np.diag(diag * (1 + 2 * (kx + ky)), 0) + np.diag(diagsup * (-ky), 1) + np.diag(diagsup * (-ky), -1)
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
        
        dat_file = os.path.join(save_folder, "Data/data" + str(isim) + ".csv")
        fires_file = os.path.join(save_folder, "Fires/fires" + str(isim) + ".dat")
        
        # Condition initiale
        x = np.linspace(0.0, L, J)
        y = np.linspace(0.0, L, J)
        X = np.linspace(0.0, L * L, J * J)
        
        [u_0, w_0] = starter()
        
        u = u_0
        w = w_0
        
        # print('Calculs en cours...')
        file = open(dat_file, 'w')
        file_fire = open(fires_file, 'w')
        file.write('t i j u w \n')
        
        for t in range(T):
            for i in range(J):
                for j in range(J):
                    file.write(
                        str(t) + ' ' + str(i) + ' ' + str(j) + ' ' + str(u[i * J + j]) + ' ' + str(w[i * J + j]) + '\n'
                    )
            
            """Méthode de Strang"""
            # diffusion 1/2 pas
            u = np.dot(invA, u)
            w = np.dot(invA, w)
            
            # réaction 1 pas
            for i in range(J):
                for j in range(J):
                    idx = i * J + j
                    X0 = [u[idx], w[idx]]
                    orbit = odeint(reaction, X0, [0, dt], )
                    u[idx], w[idx] = orbit[-1]
            
            # diffusion 1/2 pas
            u = np.dot(invA, u)
            w = np.dot(invA, w)
            
            if t % freq == 0 and t != 0 :
                if np.random.binomial(1, p) == 1:
                    
                    xee = random.random() * J
                    yee = random.random() * J
                    
                    ree = minree + random.random() * (maxree - minree)
                    
                    file_fire.write(str(t) + ' ' + str(xee) + ' ' + str(yee) + ' '
                                    + str(ree) + ' ' + str(intensity) + '\n')
                    
                    var_ee = lambda i, j, var: var[i * J + j] * (1 - intensity) if ((i - xee) ** 2 + (
                            j - yee) ** 2) < ree ** 2 else var[i * J + j]
                    
                    for i in range(J):
                        for j in range(J):
                            u[i * J + j] = var_ee(i, j, u)
                            w[i * J + j] = var_ee(i, j, w)


if __name__ == "__main__":
    nb_simulations = input("nb simu :")
    ballcenter = input("centre simu (persistence/extinction): ")
    ballradius = input("rayon au centre : ")
    save_folder = input("dossier de sauvegarde : ")

    if ballcenter == "p":
        ballcenter = uplus, wplus
    elif ballcenter == "e":
        ballcenter = 0, 0
    else:
        ballcenter = float(ballcenter), float(ballcenter)

    print("Ball center: ", ballcenter)
    print("Ball radius: ", ballradius)
    print("Number of simulations: ", nb_simulations)
    print("Save folder: ", save_folder)
    print("Running simulation...")
    run_simulation(int(nb_simulations), ballcenter, float(ballradius), save_folder)
 
