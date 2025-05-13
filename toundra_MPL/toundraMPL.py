from matplotlib.patches import Polygon
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import yaml
import numpy as np
from scipy.linalg import block_diag
from scipy.integrate import odeint
import os
import random
import math
from tqdm import tqdm

with open("config_toundraMPL.yaml", "r") as f:
    config = yaml.safe_load(f)


########################################################################################
# ----------------------- Définition des constantes / semi-variables
# NOTA : NE PAS MODIFIER ICI. CHANGER LE FICHIER CONFIG.YAML
########################################################################################

os.makedirs("Simulations", exist_ok=True)
MAKE_NEW_SIMULATION = config["MAKE_NEW_SIMULATION"]
SIMULATION_SAVE_FOLDER = os.path.join("Simulations", config["SIMULATION_SAVE_FOLDER"])

ISO_SCALE = config["ISO_SCALE"]  # échelle du rendu
MAX_ELMTS_PER_TILE = config["MAX_ELMTS_PER_TILE"]  # ratio par rapport à iso_scale
ELMT_SIZE = int(config['ELMT_SIZE'] * ISO_SCALE)  # taille des arbres en ratio (peut être > 1) par rapport à iso_scale
TPS = config["TPS"]  # ticks par seconde (=vitesse de lecture)
FIRE_DURATION = config["FIRE_DURATION"]  # durée en pas de temps (integer)
DEAD_TREE_DURATION = config["DEAD_TREE_DURATION"]  # idem

# Généralités fenêtre
SCREEN_WIDTH = 2200
SCREEN_HEIGHT = 1200


# Paramètres rendu isométrique
ANGLE = 23
COS = math.cos(math.radians(ANGLE))
SIN = math.sin(math.radians(ANGLE))

rot = -160
cos_rot = math.cos(math.radians(rot))
sin_rot = math.sin(math.radians(rot))

ORIGIN_X = SCREEN_WIDTH // 3.2
ORIGIN_Y = SCREEN_HEIGHT * 0.1

AREA_SCALE = 1  # hectares par pixel

# Max d'arbres (réel)
TREES_T_MIN = 0
TREES_T_MAX = config['MAX_TRUE_NUMBER_OF_TREES']

# MAX T est valué par data_util lors du chargement des données (voir fonction "build_from_solution")
MAX_T = -1
TILE_SIZE = 1

# constantes couleurs
BROWN = (143/255, 50/255, 43/255)
BLACK = (0/255, 0/255, 0/255)
WHITE = (1.0, 1.0, 1.0)
DARK_GREEN = (35/255, 94/255, 35/255)
LIGHT_GREEN = (86/255, 179/255, 86/255)
RED = (1.0, 0.0, 0.0)
SNOW = (197/255, 205/255, 217/255)
ICE = (160/255, 190/255, 235/255)
BLUE = (0.0, 0.0, 1.0)
ORANGE = (1.0, 128/255, 0.0)

########################################################################################
# ----------------------- Initialisation de Pygame et du videoWriter
########################################################################################


print("Program started.")

# Paramètres de la simulation
FIG_WIDTH, FIG_HEIGHT = SCREEN_WIDTH / 100, SCREEN_HEIGHT / 100  # Conversion pixels → pouces
DPI = 100  # Densité standard

# Dossier de sauvegarde
os.makedirs(SIMULATION_SAVE_FOLDER, exist_ok=True)

def create_figure():
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=DPI)
    ax.set_xlim(0, SCREEN_WIDTH)
    ax.set_ylim(0, SCREEN_HEIGHT)
    ax.set_aspect('equal')
    ax.axis('off')  # Pas de bordures ni d’axes
    return fig, ax

########################################################################################
# ----------------------- Chargement des assets et sources externes
########################################################################################

def read_params_from(path):
    formatted_lines = []
    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                # c'est un param
                param, val = parts
                formatted_lines.append((param + ': ' + str(val)))
            elif len(parts) == 1:
                # c'est un break
                formatted_lines.append("----------")
    
    return formatted_lines


def load_and_scale_image(image_path, size):
    img = Image.open(image_path).convert("RGBA")
    img = img.resize((size, size), Image.LANCZOS)
    return img

images = {
    "mixte_tree": load_and_scale_image("assets/mixte.png", ELMT_SIZE),
    "boreal_tree": load_and_scale_image("assets/boreal.png", ELMT_SIZE),
    "dead_tree": load_and_scale_image("assets/deadtree.png", ELMT_SIZE),
    "burning_tree": load_and_scale_image("assets/burning_tree.png", ELMT_SIZE),
}


########################################################################################
# ----------------------- Partie simulation - Différences finies
########################################################################################

## Chargement des paramètres depuis le fichier YAML (config)
# competitions
a = config["params_simulation"]["a"]
b = config["params_simulation"]["b"]
c = config["params_simulation"]["c"]
gamma_south = config["params_simulation"]["gamma_south"]
gamma_north = config["params_simulation"]["gamma_north"]

# boreale
alpha_b = config["params_simulation"]["alpha_b"]
beta_b = config["params_simulation"]["beta_b"]
delta_b = config["params_simulation"]["delta_b"]

# mixte
alpha_m = config["params_simulation"]["alpha_m"]
beta_m = config["params_simulation"]["beta_m"]
delta_m = config["params_simulation"]["delta_m"]

# discrétisation
L = config["params_simulation"]["L"]
J = config["params_simulation"]["J"]
T = config["params_simulation"]["T"]
N = config["params_simulation"]["N"]
p = config["params_simulation"]["p"]
freq = config["params_simulation"]["freq"]
intensity = config["params_simulation"]["intensity"]
minree = config["params_simulation"]["minree"]
maxree = config["params_simulation"]["maxree"]

# Calcul des paramètres dérivés
dx = L / J
dy = L / J
dt = T / N
dtau = dt / 2
kx_b = delta_b * dtau / dx ** 2
ky_b = delta_b * dtau / dy ** 2
kx_m = delta_m * dtau / dx ** 2
ky_m = delta_m * dtau / dy ** 2

init_method = "naive"


def randomize_starter(uplus, wplus, umoins, wmoins):
    if init_method == "naive":
        # méthode naive : état instable +- 0.5 partout, sur chaque point
        u_0 = np.zeros(J * J)
        w_0 = np.zeros(J * J)
        for i in range(J):
            for j in range(J):
                u_0[i * J + j] = umoins + (-1 + 2 * random.random()) * 0.5
                w_0[i * J + j] = wmoins + (-1 + 2 * random.random()) * 0.5
        
        return u_0, w_0
    
    elif init_method == "bumps":
        # méthode "à bosses"
        max_bosses = 50
        max_height = uplus
        max_d = 20
        
        nb_bosses = random.randint(1, max_bosses)
        u_0 = np.zeros(J * J)
        w_0 = np.zeros(J * J)
        
        for _ in range(nb_bosses):
            
            h = random.random() * max_height
            d = random.randint(1, max_d)
            cx = random.randint(0, J)
            cy = random.randint(0, J)
            
            print(f"h = {h}, d = {d}, cx = {cx}, cy = {cy}")
            
            bosse_func = lambda i, j: h * d / ((cx - i) ** 2 + (cy - j) ** 2 + d)
            
            for i in range(J):
                for j in range(J):
                    u_0[i * J + j] += bosse_func(i, j)
                    w_0[i * J + j] += bosse_func(i, j)
        
        return u_0, w_0

# Termes de réaction du modèle

gamma = np.linspace(start=gamma_south, stop=gamma_north, num=J)

def q_b(ub, um, gamma_i):
    return ub * (a * (ub + um - b) ** 2 + c) + gamma_i * ub * um

def q_m(ub, um, gamma_i):
    return um * (a * (um + ub - b) ** 2 + c) - gamma_i * ub * um


def reaction(X, t, i):
    ub, um, wb, wm = X
    gamma_i = gamma[i]

    dub = alpha_b * wb - q_b(ub, um, gamma_i)
    dwb = -beta_b * wb + alpha_b * ub

    dum = alpha_m * wm - q_m(ub, um, gamma_i)
    dwm = -beta_m * wm + alpha_m * um

    return [dub, dum, dwb, dwm]



def run_simulation(save_folder="Simulations/default"):
    
    os.makedirs(save_folder, exist_ok=True)
    
    dat_file = os.path.join(save_folder, "data.csv")
    param_file = os.path.join(save_folder, "params.txt")
    fires_file = os.path.join(save_folder, "fires.dat")
    
    ### Écriture des paramètres dans un fichier texte
    with open(param_file, "w") as f_:
        f_.write("Simulation " + SIMULATION_SAVE_FOLDER + "\n")
        f_.write("----------\n")
        for key, value in config["params_simulation"].items():
            f_.write(f"{key} {value}\n")
        # Saut de ligne avant les paramètres dérivés
        f_.write("----------\n")
        # Écriture des paramètres dérivés
        derived_params = {
            "dx": dx,
            "dy": dy,
            "dt": dt,
            "dtau": dtau,
        }
        for key, value in derived_params.items():
            f_.write(f"{key} {value}\n")
    
    print(f"Paramètres enregistrés dans {param_file}")
    
    # Équilibres du modèle
    uplus = b + np.sqrt(abs((alpha_b ** 2 - beta_b * c) / (a * beta_b)))
    wplus = alpha_b * uplus / beta_b
    
    umoins = b - np.sqrt(abs((alpha_b ** 2 - beta_b * c) / (a * beta_b)))
    wmoins = alpha_b * umoins / beta_b
    
    print("Uplus = (", uplus, ", ", wplus, ")")
    print("Umoins = (", umoins, ", ", wmoins, ")")
    
    # Condition initiale
    x = np.linspace(0.0, L, J)
    y = np.linspace(0.0, L, J)
    X = np.linspace(0.0, L * L, J * J)
    
    u_b0, w_b0 = randomize_starter(uplus, wplus, umoins, wmoins)
    
    u_m0, w_m0 = randomize_starter(uplus, wplus, umoins, wmoins)
 
    # Matrices du schéma : boreale
    diag = np.ones(J)
    diagsup = np.ones(J - 1)
    D = np.diag(diag * (1 + 2 * (kx_b + ky_b)), 0) + np.diag(diagsup * (-ky_b), 1) + np.diag(diagsup * (-ky_b), -1)
    A_b = block_diag(D)
    for i in range(J - 1):
        A_b = block_diag(A_b, D)
    
    # Conditions au bord de Neumann
    for k in [0, J - 1, J * J - J, J * J - 1]:
        A_b[k][k] = 1 + kx_b + ky_b
    for n in range(1, J - 1):
        n1 = J * n
        n2 = J * n + J - 1
        A_b[n1][n1] = 1 + 2 * kx_b + ky_b
        A_b[n2][n2] = 1 + 2 * kx_b + ky_b
    for k in range(1, J - 1):
        A_b[k][k] = 1 + kx_b + 2 * ky_b
    for k in range(J * J - J + 1, J * J - 1):
        A_b[k][k] = 1 + kx_b + 2 * ky_b
    
    grandediag = np.ones(J * (J - 1))
    A_b = A_b + np.diag(grandediag * (-ky_b), J) + np.diag(grandediag * (-ky_b), -J)
    
    # Matrices du schéma : mixte
    diag = np.ones(J)
    diagsup = np.ones(J - 1)
    D = np.diag(diag * (1 + 2 * (kx_m + ky_m)), 0) + np.diag(diagsup * (-ky_m), 1) + np.diag(diagsup * (-ky_m), -1)
    A_m = block_diag(D)
    for i in range(J - 1):
        A_m = block_diag(A_m, D)
    
    # Conditions au bord de Neumann
    for k in [0, J - 1, J * J - J, J * J - 1]:
        A_m[k][k] = 1 + kx_m + ky_m
    for n in range(1, J - 1):
        n1 = J * n
        n2 = J * n + J - 1
        A_m[n1][n1] = 1 + 2 * kx_m + ky_m
        A_m[n2][n2] = 1 + 2 * kx_m + ky_m
    for k in range(1, J - 1):
        A_m[k][k] = 1 + kx_m + 2 * ky_m
    for k in range(J * J - J + 1, J * J - 1):
        A_m[k][k] = 1 + kx_m + 2 * ky_m
    
    grandediag = np.ones(J * (J - 1))
    A_m = A_m + np.diag(grandediag * (-ky_m), J) + np.diag(grandediag * (-ky_m), -J)
    
    # Calculs
    print("Inversion de matrices...")
    invA_b = np.linalg.inv(A_b)
    invA_m = np.linalg.inv(A_m)
    
    u_b = u_b0
    w_b = w_b0
    u_m = u_m0
    w_m = w_m0
    
    print('Calculs en cours...')
    file = open(dat_file, 'w')
    file_fire = open(fires_file, 'w')
    file.write('t i j ub wb um wm \n')
    
    for t in tqdm(range(T)):
        for i in range(J):
            for j in range(J):
                file.write(
                    str(t) + ' ' + str(i) + ' ' + str(j) + ' ' + str(u_b[i * J + j]) + ' ' + str(w_b[i * J + j]) +
                    ' ' + str(u_m[i * J + j]) + ' ' + str(w_m[i * J + j]) + '\n'
                )
        
        """Méthode de Strang"""
        # diffusion 1/2 pas
        u_b = np.dot(invA_b, u_b)
        w_b = np.dot(invA_b, w_b)
        u_m = np.dot(invA_m, u_m)
        w_m = np.dot(invA_m, w_m)
        
        # réaction 1 pas
        for i in range(J):
            for j in range(J):
                idx = i * J + j
                X0 = [u_b[idx], u_m[idx], w_b[idx], w_m[idx]]
                orbit = odeint(reaction, X0, [0, dt], args=(j,))
                u_b[idx], u_m[idx], w_b[idx], w_m[idx] = orbit[-1]
        
        # diffusion 1/2 pas
        u_b = np.dot(invA_b, u_b)
        w_b = np.dot(invA_b, w_b)
        u_m = np.dot(invA_m, u_m)
        w_m = np.dot(invA_m, w_m)
        
        if t % freq == 0:
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
                        u_b[i * J + j] = var_ee(i, j, u_b)
                        w_b[i * J + j] = var_ee(i, j, w_b)
                        u_m[i * J + j] = var_ee(i, j, u_m)
                        w_m[i * J + j] = var_ee(i, j, w_m)
    
    print('Fin du programme de simulation.')


def get_dfs(from_folder="Simulations/default"):
    df = pd.read_csv(os.path.join(from_folder, "data.csv"),
                     names=["t", "x", "y", "ub", "wb", "um", "wm"], header=0, sep=" ", index_col=False)
    df = df.drop_duplicates()
    
    df_feux = pd.read_csv(os.path.join(from_folder, "fires.dat"),
                          names=['t', 'x', 'y', 'r', 'I'], header=None, sep=" ", index_col=False)
    
    return df, df_feux


########################################################################################
# ----------------------- Gestion rendu isométrique
########################################################################################

# utilisé dans Tile.draw_tile. Les coordonnées de Nodes et Fires sont cartésiennes.
def projeter_en_isometrique(x, y):
    cx = x
    cy = y
    
    xr = (cx * cos_rot) + (cy * sin_rot)
    yr = (cx * sin_rot) - (cy * cos_rot)
    
    x_proj = (-xr - yr) * COS * ISO_SCALE + ORIGIN_X
    y_proj = (-xr + yr) * SIN * ISO_SCALE + ORIGIN_Y
    
    return x_proj, y_proj


def grid_to_iso(i, j):
    """ Convertit les coordonnées de la grille (i, j) en coordonnées isométriques. """
    
    # Coordonnées centrées sur l'origine
    cx = i * TILE_SIZE / 2
    cy = j * TILE_SIZE / 2
    
    # Appliquer la rotation et la mise à l'échelle isométrique
    xr = (cx * cos_rot) - (cy * sin_rot)
    yr = (cx * sin_rot) + (cy * cos_rot)
    
    x_iso = (-xr - yr) * COS * ISO_SCALE + ORIGIN_X
    y_iso = (-xr + yr) * SIN * ISO_SCALE + ORIGIN_Y
    
    return x_iso, y_iso


########################################################################################
# ----------- Importation des résultats FreeFem++ et construction des nodes
########################################################################################


# Node : sommet de maille, avec vecteurs des valeurs en fonction du temps
class Node:
    """ Une node est d'abord un point réel / non-interpolé, donné par la sortie freefem"""
    
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.ub = None
        self.um = None
        
        self.fires = None


# build_from_solution : construction des nodes à partir du dossier de sortie FreeFem++
def build_from_solution(saved_simulation_folder="Simulations/default"):
    global MAX_T
    
    print(f"Building from solution folder \"{saved_simulation_folder}\"...")
    nodes__ = []
    
    df, df_feux = get_dfs(saved_simulation_folder)
    MAX_T = df["t"].max()
    # nota : le df_feux est passé en retour, il n'est pas utilisé avant que les mailles soient définies
    
    print(f"\tMax time is {MAX_T}")
    
    ub_min, ub_max = df["ub"].min(), df["ub"].max()
    um_min, um_max = df["um"].min(), df["um"].max()
    
    unique_xy_list = list(df[['x', 'y']].drop_duplicates().itertuples(index=False, name=None))
    
    print("TILE_SIZE = ", TILE_SIZE)
    
    for xy in unique_xy_list:
        x = xy[0]
        y = xy[1]
        node = Node(x, y)
        
        filtered_df = df[(df['x'] == x) & (df['y'] == y)]
        
        node.ub = filtered_df['ub'].to_numpy()
        node.um = filtered_df['um'].to_numpy()
        
        nodes__.append(node)
    
    print(f"\tBuilt {len(nodes__)} nodes.")
    print(f"\tub_min = {ub_min}")
    print(f"\tub_max = {ub_max}")
    print(f"\tum_min = {um_min}")
    print(f"\tum_max = {um_max}")
    print("\n")
    
    return nodes__, df_feux


########################################################################################
# ----------- Gestion des mailles à partir des nodes récupérées ci-dessus
########################################################################################

# classe simple pour stocker les feux proprement
class Fire:
    
    def __init__(self, t_start, x, y, r, I):
        self.t_start = t_start
        # coordonnées cartésiennes
        self.x = x
        self.y = y
        self.r = r
        self.I = I


# class::Tree : arbre avec sa position (isométrique) son type et affiché ou non. Méthode d'affichage incluse.
class Tree:
    
    def __init__(self, state, *pos):
        # coordonnées isométriques :
        x, y = (pos[0]-ELMT_SIZE//2, pos[1])
        # états : 'hide', 'mixte', 'boreal', 'burning', 'dead'
        self.states = ['hide'] * MAX_T
        self.extent = (x, x + ELMT_SIZE, y, y + ELMT_SIZE)
    
    # affichage de l'arbre en fonction du state et du temps
    def blit(self, ax, t):
        if self.states[t] == 'boreal':
            ax.imshow(images["boreal_tree"],
                      extent=self.extent, origin='upper')
        elif self.states[t] == 'mixte':
            ax.imshow(images["mixte_tree"],
                      extent=self.extent)
        elif self.states[t] == 'burning':
            ax.imshow(images["burning_tree"],
                      extent=self.extent)
        elif self.states[t] == 'dead':
            ax.imshow(images["dead_tree"],
                      extent=self.extent)
        elif self.states[t] == 'hide':
            pass


# class::Tile : Une maille. Vecteurs moyens des nodes qui la compose. Contient la liste de ses instances d'arbres,
# gère ses feux, l'affichage de ses arbres et son affichage (fond, bordure).
class Tile:
    """ Représente une zone polygonale (triangle) dans l'écran avec ses nœuds et ses propriétés."""
    
    def __init__(self, topleft_node):
        
        self.topleft_node = topleft_node
        self.iso_x, self.iso_y = projeter_en_isometrique(self.topleft_node.x, self.topleft_node.y)
        
        self.corners = []
        self.corners.append((self.topleft_node.x, self.topleft_node.y))
        self.corners.append((self.topleft_node.x + TILE_SIZE, self.topleft_node.y))
        self.corners.append((self.topleft_node.x + TILE_SIZE, self.topleft_node.y + TILE_SIZE))
        self.corners.append((self.topleft_node.x, self.topleft_node.y + TILE_SIZE))
        
        self.center = (self.topleft_node.x + TILE_SIZE // 2, self.topleft_node.y + TILE_SIZE // 2)
        
        self.area = TILE_SIZE * TILE_SIZE
        self.ub = topleft_node.ub
        self.um = topleft_node.um
        
        # nombre véritable d'arbres
        self.true_nb_boreals = [u_t * self.area * AREA_SCALE for u_t in self.ub]
        self.true_nb_mixtes = [u_t * self.area * AREA_SCALE for u_t in self.um]
        
        # nombre d'arbres à blitter (fonction du nombre max d'arbres à montrer)
        self.screen_nb_boreals = [int((true_nb_trees_t - TREES_T_MIN) * MAX_ELMTS_PER_TILE // 2
                                      / (TREES_T_MAX - TREES_T_MIN))
                                  for true_nb_trees_t in self.true_nb_boreals]
        
        self.screen_nb_mixtes = [int((true_nb_trees_t - TREES_T_MIN) * MAX_ELMTS_PER_TILE // 2
                                     / (TREES_T_MAX - TREES_T_MIN))
                                 for true_nb_trees_t in self.true_nb_mixtes]
        
        self.trees = []  # liste d'instance de trees
        self.generate_trees()
        random.shuffle(self.trees)
    
    # generate_trees : construction des positions d'abres (avec random_point_in_triangle) et affectation des positions
    # à des instances d'arbres. Par défaut, tous les arbres sont .show=False
    def generate_trees(self):
        
        ## 1 - génération des positions
        generated_points = []
        
        while len(generated_points) < MAX_ELMTS_PER_TILE:
            x, y = (random.uniform(self.topleft_node.x, self.topleft_node.x + TILE_SIZE),
                    random.uniform(self.topleft_node.y, self.topleft_node.y + TILE_SIZE))
            
            iso_x, iso_y = projeter_en_isometrique(x, y)
            iso_point = (int(iso_x), int(iso_y))
            
            if iso_point not in generated_points:
                generated_points.append(iso_point)
        
        ## 2- construction des arbres
        for point in generated_points:
            tree = Tree('hide', *point)
            self.trees.append(tree)
        
        # construction des showtimes
        for t in range(MAX_T):
            
            nb_boreals_t = self.screen_nb_boreals[t]
            nb_mixtes_t = self.screen_nb_mixtes[t]
            for tree in self.trees[:nb_boreals_t]:
                tree.states[t] = 'boreal'
            # contournement d'un défaut de slice index natif de python : [-0:] == [0:] !! >:(
            if -nb_mixtes_t != 0:
                for tree in self.trees[-nb_mixtes_t:]:
                    tree.states[t] = 'mixte'

    # dessin de la maille, et de son contour si décommenté ci-dessous
    def draw_tile(self, ax):
        points_iso = [projeter_en_isometrique(corner[0], corner[1]) for corner in self.corners]
        poly = Polygon(points_iso, closed=True, facecolor=SNOW, edgecolor=None, zorder=0)
        ax.add_patch(poly)
    
    def draw_tile_borders(self, ax):
        points_iso = [projeter_en_isometrique(corner[0], corner[1]) for corner in self.corners]
        poly = Polygon(points_iso, closed=True, fill=False, edgecolor='black', linewidth=1, zorder=2)
        ax.add_patch(poly)
        
    # affichage des arbres ayant .show=True (appelé après draw_tiles pour bliter par dessus la maille)
    def draw_trees(self, ax, t):
        for tree in self.trees:
            tree.blit(ax, t)


# create_tiles : Création des mailles à partir des nodes (issues de build_from_solution a priori)
def create_tiles(nodes):
    print("Generating tiles...")
    
    tiles_ = []
    for node in nodes:
        tile = Tile(node)
        tiles_.append(tile)
    
    print(f"\tGenerated {len(tiles_)} tiles.")
    
    mean_area = np.mean([tile.area for tile in tiles_])
    max_true_boreals = np.max([tile.true_nb_boreals for tile in tiles_])
    max_true_mixtes = np.max([tile.true_nb_mixtes for tile in tiles_])
    max_screen_boreals = np.max([tile.screen_nb_boreals for tile in tiles_])
    max_screen_mixtes = np.max([tile.screen_nb_mixtes for tile in tiles_])
    
    print("\nTILES PROPERTIES")
    print(f"\tmean_area: {mean_area}")
    print(f"\tmax_true_boreals: {max_true_boreals}")
    print(f"\tmax_true_mixtes: {max_true_mixtes}")
    print(f"\tmax_screen_boreals: {max_screen_boreals}")
    print(f"\tmax_screen_mixtes: {max_screen_mixtes}")
    
    return tiles_


# resolve_burning_trees : fonction qui relie les feux et les tiles. Détermine les arbres qui brûlent.
def resolve_burning_trees(tiles, df_feux):
    ## 1 - construction des feux
    fires = []
    for _, fire in df_feux.iterrows():
        fire = Fire(fire["t"], fire["x"], fire["y"], fire["r"], fire["I"])
        fires.append(fire)
    
    ## 2 - affectation des feux aux arbres
    for fire in fires:
        
        # on pose les temps de transition pour aérer le code plus bas
        # pourquoi le +1 ? parce que lorsque t_start est à x, les arbres disparaissent à x+1 dans la simu.
        # on utilise toujours t_start pour regarder "eligible_trees" (cf plus bas) ; mais les images de feu doivent
        # commencer lorsque les arbres ont disparus, pour les remplacer, càd à x+1.
        
        fire_start = int(fire.t_start) + 1
        fire_end = min(fire_start + FIRE_DURATION, MAX_T)
        dead_start = fire_end
        dead_end = min(dead_start + DEAD_TREE_DURATION, MAX_T)
        
        for tile in tiles:
            
            dif_x = abs(tile.center[0] - fire.x)
            dif_y = abs(tile.center[1] - fire.y)
            
            # vérifier que le centre de la tuile est dans le feu = tuile en feu
            if dif_x ** 2 + dif_y ** 2 < fire.r ** 2:
                
                # arbres eligibles à brûler si pas 'hide' ou 'dead' au temps du feu
                eligible_trees = [tree for tree in tile.trees
                                  if tree.states[int(fire.t_start) - 1] not in ['hide', 'dead']]
                
                nb_to_burn = int(len(eligible_trees) * fire.I)  # nb d'arbres qui brûlent parmi les eligibles
                selected_trees = random.sample(eligible_trees, nb_to_burn)
                
                for tree in selected_trees:
                    for fire_time in range(fire_start, fire_end):
                        tree.states[fire_time] = 'burning'
                    for dead_time in range(dead_start, dead_end):
                        tree.states[dead_time] = 'dead'


########################################################################################
# ------------------------ Main
########################################################################################


if __name__ == "__main__":
    
    # C'est parti
    print(" ###### STARTED.")
    print("Running main.\n")
    
    if not os.path.exists(SIMULATION_SAVE_FOLDER) and not MAKE_NEW_SIMULATION:
        print(f"ERROR : Simulation {SIMULATION_SAVE_FOLDER} not found.\n"
              f" Try setting \"MAKE_NEW_SIMULATION\" to \"True\", or select another folder.")
        quit(1)
    
    if MAKE_NEW_SIMULATION:
        # lance la simulation et récupère les paramètres
        run_simulation(SIMULATION_SAVE_FOLDER)
        
    """
    # Gestion de l'affichage des paramètres
    lines = read_params_from(os.path.join(SIMULATION_SAVE_FOLDER, "params.txt"))  # à adapter
    font = pygame.font.Font(None, 26)
    text_params = [font.render(line, True, BLACK) for line in lines]
    """
    
    # Construction des instances du rendu
    nodes, df_feux = build_from_solution(SIMULATION_SAVE_FOLDER)
    tiles = create_tiles(nodes)
    resolve_burning_trees(tiles, df_feux)
    
    print(f"MAX_ELMTS_PER_TILE is set to {MAX_ELMTS_PER_TILE}.")
    
    frames_folder = os.path.join(SIMULATION_SAVE_FOLDER, "Frames")
    os.makedirs(frames_folder, exist_ok=True)
    
    ########################################################################################
    # -------------------- Boucle principale
    ########################################################################################
    
    print("Generating frames.\n")

    
    for t in range(65, 80):
        
        print(f"Generating frame {t}...")
        print("fig ax makes")
        fig, ax = create_figure()
        
        print("tiles draws...")
        for tile in tiles:
            tile.draw_tile(ax)
        """
        for tile in tiles:
            tile.draw_tile_borders(ax)
        """
        print("tiles trees draws...")
        for tile in tiles:
            tile.draw_trees(ax, t)
        
        print("saving fig...")
        ax.set_facecolor(BLACK)
        fig.canvas.draw()  # Dessine la figure
        img = np.array(fig.canvas.renderer.buffer_rgba())  # Récupère l'image au format RGBA
        #img = np.flipud(img)  # Miroir vertical
        #file_path = os.path.join(frames_folder, f"frame_{t:03d}.png")
        file_path = "test_frame.png"
        plt.imsave(file_path, img)
        print("done.\n")


    
    

    
    
