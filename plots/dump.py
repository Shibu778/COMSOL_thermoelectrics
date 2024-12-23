# This file contains the unused functions

import numpy as np
import os


class Dump:
    def __init__(self):
        pass

    def read_file(self, filename):
        # Read the data from the file
        with open(filename, "r") as file:
            data = file.readlines()
        data = [line.strip().split() for line in data]
        data = data[5:]
        data = np.array(data, dtype=float)
        return data

    def read_V_vs_T_data(self, data, headers=["T", "V"]):
        # Read the V vs T data from the file
        T = data[:, 0] + self.low_T  # Temperature of the top plate
        V = data[:, 1]  # Voltage across the bar
        data_VT = {headers[0]: T, headers[1]: V}
        return data_VT

    def read_V_vs_I_data(self, data, headers=["I", "V"]):
        # Read the V vs I data from the file
        I = data[:, 0]  # Current applied at top plate
        V = data[:, 1]
        data_VI = {headers[0]: I, headers[1]: V}
        return data_VI


# V vs T data
# chf = Constant_Heat_Flux()
# data = chf.read_file("../data/HC_CHF_V_vs_T.txt")
# data_VT = chf.read_V_vs_T_data(data)
# print(data_VT)
# print(data)

# root_dir = "../data/test/"
# for filename in [
#     "S_CHF_V_vs_I0_500K.txt",
#     "HC_CHF_V_vs_I0_500K.txt",
#     "HH_CHF_V_vs_I0_500K.txt",
#     "HM_CHF_V_vs_I0_500K.txt",
# ]:
#     chf = CHF_Experiment()

#     # V vs I data
#     data = chf.read_file(root_dir + filename)
#     data_VI = chf.read_V_vs_I_data(data)
#     print(filename.split(".")[0], chf.calc_relevant_data(data_VI))
