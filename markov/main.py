import matplotlib.pyplot as plt
import tooling


def fonc1():

    # je lance plein de simulations, avec des plots et des calculs de proba
    # les paramètres de la simu sont réglés dans config_simu.yml, à part p
    # p, je le précise quand je lance la simu

    persistence_vector = tooling.persistence_vector
    extinction_vector = tooling.extinction_vector

    probs = open("probs.txt", "a")

    # persistence sans feu
    # tooling.run_simulation(500, sim_id="auto_p_nf", center=persistence_vector, p=0.0)
    tooling.plot_folder_distances("auto_p_nf", persistence_vector, hline_norm=500)
    pstay, pout = tooling.compute_probs("auto_p_nf", persistence_vector, b_radius=500)
    probs.write("p sans feu\n")
    probs.write(f"pstay:{pstay}, pout:{pout}\n")

    # persistence avec feu
    # tooling.run_simulation(500, sim_id="auto_p_f", center=persistence_vector, p=1.0)
    tooling.plot_folder_distances("auto_p_f", persistence_vector, hline_norm=500)
    pstay, pout = tooling.compute_probs("auto_p_f", persistence_vector, b_radius=500)
    probs.write("p avec feu\n")
    probs.write(f"pstay:{pstay}, pout:{pout}\n")

    # extinction sans feu
    # tooling.run_simulation(500, sim_id="auto_e_nf", center=extinction_vector, p=0.0)
    tooling.plot_folder_distances("auto_e_nf", extinction_vector, hline_norm=500)
    pstay, pout = tooling.compute_probs("auto_e_nf", extinction_vector, b_radius=500)
    probs.write("e sans feu\n")
    probs.write(f"pstay:{pstay}, pout:{pout}\n")

    # extinction avec feu
    # tooling.run_simulation(500, sim_id="auto_e_f", center=extinction_vector, p=1.0)
    tooling.plot_folder_distances("auto_e_f", extinction_vector, hline_norm=500)
    pstay, pout = tooling.compute_probs("auto_e_f", extinction_vector, b_radius=500)
    probs.write("e avec feu\n")
    probs.write(f"pstay:{pstay}, pout:{pout}\n")


def fonc2():
    persistence_vector = tooling.persistence_vector
    extinction_vector = tooling.extinction_vector

    # f = open("probs.txt", "a")
    # f.write("\n\t***Running from Intermed...\n")
    #
    # tooling.run_simulation(2000, "auto_i_nf", center=None, p=0.0)
    # p_stay, p_to_ext, p_to_per = tooling.compute_probs_intermed(
    #     b_radius=500, sim_id="auto_i_nf"
    # )
    # f.write(f"{p_stay=}, {p_to_ext=}, {p_to_per=}\n")

    # tooling.plot_intermeds("auto_i_nf")
    fig_to_p = tooling.plot_folder_distances(
        "auto_i_nf", center=persistence_vector, hline_norm=500
    )
    fig_to_e = tooling.plot_folder_distances(
        "auto_i_nf", center=extinction_vector, hline_norm=500
    )
    fig_to_p.show()
    fig_to_e.show()
    plt.show()

def fonc3():
    """ Générations intermédiaires avec la nouvelle norme infinie
    """
    


