"""
Programme pour calculer symboliquement la probabilité d'oscillation en
fonction de la probabilité de feu p. Basé sur le calcul de la chaîne de
markov obtenu par tooling2.py, que j'ai réduite à une forme 1D
(voir cahier de notes).
"""

from itertools import product

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

plt.rcParams.update({"font.size": 15})


class Polynome:
    # Exemple d'utilisation
    # pol1 = Polynome(0.3, 2)
    # pol2 = Polynome(1, 4)
    # print(pol1) print(pol2)
    # print(pol1 + pol2)
    # print(pol1 * pol2)
    def __init__(self, *coeffs):
        self.coeffs = list(coeffs)

    def __repr__(self) -> str:
        s = ""
        for d, coeff in enumerate(self.coeffs):
            if coeff != 0:
                if d == 0:
                    s += f"{coeff} +"
                elif d == 1:
                    s += f"{coeff}X +"
                else:
                    s += f"{coeff}X^{d} +"
        return s[:-1]

    def __add__(self, other: "Polynome"):
        max_len = max(len(self.coeffs), len(other.coeffs))
        new_coeffs = []
        for i in range(max_len):
            c1 = self.coeffs[i] if i < len(self.coeffs) else 0
            c2 = other.coeffs[i] if i < len(other.coeffs) else 0
            new_coeffs.append(c1 + c2)
        return Polynome(*new_coeffs)

    def __mul__(self, other: "Polynome"):
        new_coeffs = [0] * (len(self.coeffs) + len(other.coeffs) - 1)
        for i, c1 in enumerate(self.coeffs):
            for j, c2 in enumerate(other.coeffs):
                new_coeffs[i + j] += c1 * c2
        return Polynome(*new_coeffs)

    def evaluate(self, X):
        result = 0
        for d, coeff in enumerate(self.coeffs):
            result += coeff * (X**d)
        return result


transition_matrix = [
    [Polynome(0.655, 0.345 * 0.0525), Polynome(0.345, -0.345 * 0.0525), Polynome(0)],
    [
        Polynome(0, 0.545 * 0.0525),
        Polynome(0.545, +0.2921625),
        Polynome(0.455, -0.320775),
    ],
    [Polynome(0), Polynome(0, 0.705), Polynome(1, -0.705)],
]


def generate_words_from_E(M):
    prefix = "E"
    suffix = "IP"
    letters = ["E", "I"]
    all_words = []

    for body_len in tqdm(range(M - 1)):  # corps de taille 0 à M-2 inclus
        for combo in product(letters, repeat=body_len):
            word = prefix + "".join(combo) + suffix
            all_words.append(word)
    return all_words


def generate_words_from_P(M):
    prefix = "P"
    suffix = "IE"
    letters = ["P", "I"]
    all_words = []

    for body_len in tqdm(range(M - 1)):  # corps de taille 0 à M-2 inclus
        for combo in product(letters, repeat=body_len):
            word = prefix + "".join(combo) + suffix
            all_words.append(word)
    return all_words


def compute_word_probability(word: str) -> Polynome:
    """
    Calcule la probabilité d'un mot donné en utilisant la matrice de transition.
    """
    p = Polynome(1)
    for i in range(len(word) - 1):
        if word[i] == "E" and word[i + 1] == "E":
            p *= transition_matrix[0][0]
        elif word[i] == "E" and word[i + 1] == "I":
            p *= transition_matrix[0][1]
        elif word[i] == "I" and word[i + 1] == "E":
            p *= transition_matrix[1][0]
        elif word[i] == "I" and word[i + 1] == "I":
            p *= transition_matrix[1][1]
        elif word[i] == "P" and word[i + 1] == "E":
            p *= transition_matrix[2][0]
        elif word[i] == "P" and word[i + 1] == "I":
            p *= transition_matrix[2][1]
        elif word[i] == "I" and word[i + 1] == "P":
            p *= transition_matrix[1][2]
        elif word[i] == "E" and word[i + 1] == "P":
            p *= transition_matrix[0][2]
        elif word[i] == "P" and word[i + 1] == "P":
            p *= transition_matrix[2][2]
    return p


M = 20

words_from_E = generate_words_from_E(M)
words_from_P = generate_words_from_P(M)

print(f"Nombre de mots de longueur {M} commençant par E : {len(words_from_E)}")
print(words_from_E[:10])

polynomes_from_E = [compute_word_probability(word) for word in words_from_E]
polynome_from_P = [compute_word_probability(word) for word in words_from_P]

pEP = Polynome(0)
pPE = Polynome(0)

for pol in tqdm(polynomes_from_E):
    pEP += pol

for pol in tqdm(polynome_from_P):
    pPE += pol


print(f"\n pEP = {pEP}\n")
print(f"pPE = {pPE}\n")

S = pEP * pPE
print(f"Probabilité d'une oscillation : {S}\n")

# Affichage des valeurs brutes pour quelques p
S_values = [S.evaluate(p) for p in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]]
print(f"S_values: {S_values}")

min_lengths = [1, 10, 20]
p_values = np.linspace(0, 1, 100)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle(f"Effet du nombre minimal d'oscillations (M = {M})", fontsize=16)

for ax, min_oscillation_length in zip(axes, min_lengths):
    S_power = Polynome(1)
    for _ in range(min_oscillation_length):
        S_power *= S  # élève S à la puissance min_oscillation_length

    # Évaluation avec clipping pour éviter les artefacts négatifs
    S_power_values = []
    for p in p_values:
        val = S_power.evaluate(p)
        # Correction si valeur très négative
        if val < 0:
            val = 0.0
        S_power_values.append(val)

    ax.plot(p_values, S_power_values)
    ax.set_title(f"min = {min_oscillation_length}")
    ax.set_xlabel("Probabilité de feu $p$")
    ax.grid(True)

axes[0].set_ylabel("Probabilité d'oscillation")
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
