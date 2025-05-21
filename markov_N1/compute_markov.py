from numpy import save
import pandas as pd 
import os
import numpy as np 

from simu import Ueq 


def norm1(matrix: np.array):
    n1 = 0
    for row in matrix:
        for val in row:
            n1 += np.abs(val)
    
    return n1


pNorm = norm1(Ueq)

def compute(save_folder, ball_size = 1000):

    def in_extinction_ball(U):
        return norm1(U)<ball_size

    def in_persistence_ball(U):
        global pNorm
        uNorm = norm1(U)
        return np.abs(uNorm-pNorm) < ball_size        

    #save_folder = input("Dossier save à traiter : ")
    save_folder = os.path.join(save_folder, "Data")


    stayers_count = 0
    leavers_count = 0

    for dat in os.listdir(save_folder):

        df = pd.read_csv(os.path.join(save_folder, dat), sep=" ", header=0,
                        index_col = False)
        
        temps_next = df[df['t']==21].iloc[:, 3:5].to_numpy()


        if in_persistence_ball(temps_next) :
            stayers_count += 1 
        else :
            leavers_count += 1 

    total = stayers_count + leavers_count 
    p_stay = stayers_count / total
    p_leave = leavers_count / total

    print("p_stay : ", p_stay)
    print("p_leave : ", p_leave)


if __name__ == "__main__":

    save_folder = "p_long_feu"
    compute(save_folder)
