import tooling

if __name__ == "__main__":

    # je lance plein de simulations, avec des plots et des calculs de proba
    # les paramètres de la simu sont réglés dans config_simu.yml, à part p
    # p, je le précise quand je lance la simu

    persistence_vector = tooling.persistence_vector
    extinction_vector = tooling.extinction_vector

    probs = open("probs.txt", "a")

    # persistence sans feu
    tooling.run_simulation(500, sim_id="auto_p_nf", center=persistence_vector, p=0.0)
    tooling.plot_folder_distances("auto_p_nf", persistence_vector, hline_norm=500)
    pstay, pout = tooling.compute_probs("auto_p_nf", persistence_vector, b_radius=500)
    probs.write("p sans feu\n")
    probs.write(f"pstay:{pstay}, pout:{pout}\n")

    # persistence avec feu
    tooling.run_simulation(500, sim_id="auto_p_f", center=persistence_vector, p=1.0)
    tooling.plot_folder_distances("auto_p_f", persistence_vector, hline_norm=500)
    pstay, pout = tooling.compute_probs("auto_p_f", persistence_vector, b_radius=500)
    probs.write("p avec feu\n")
    probs.write(f"pstay:{pstay}, pout:{pout}\n")

    # extinction sans feu
    tooling.run_simulation(500, sim_id="auto_e_nf", center=extinction_vector, p=0.0)
    tooling.plot_folder_distances("auto_e_nf", extinction_vector, hline_norm=500)
    pstay, pout = tooling.compute_probs("auto_e_nf", extinction_vector, b_radius=500)
    probs.write("e sans feu\n")
    probs.write(f"pstay:{pstay}, pout:{pout}\n")

    # extinction avec feu
    tooling.run_simulation(500, sim_id="auto_e_f", center=extinction_vector, p=1.0)
    tooling.plot_folder_distances("auto_e_f", extinction_vector, hline_norm=500)
    pstay, pout = tooling.compute_probs("auto_e_f", extinction_vector, b_radius=500)
    probs.write("e avec feu\n")
    probs.write(f"pstay:{pstay}, pout:{pout}\n")
