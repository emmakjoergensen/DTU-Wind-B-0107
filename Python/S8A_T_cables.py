# -*- coding: utf-8 -*-
"""S8A_T_Cables

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1UaUuOtUKz46-4s-JvPeebNu_x4_bYnZ6
"""

## INSTALL ALL REQUIRED PACKAGES
import importlib
if not importlib.util.find_spec("py_wake"):
  !pip install git+https://gitlab.windenergy.dtu.dk/TOPFARM/PyWake.git
if not importlib.util.find_spec("topfarm"):
  !pip install git+https://gitlab.windenergy.dtu.dk/TOPFARM/TopFarm2.git
if not importlib.util.find_spec("ed_win"):
  !pip install git+https://gitlab.windenergy.dtu.dk/TOPFARM/edwin.git@main#egg=ed_win[interarray]

!pip install ssms

#%% IMPORT ALL REQUIRED PACKAGES

import numpy as np
import pandas as pd

#%% IMPORT ALL REQUIRED FUNCTIONS
from ed_win.wind_farm_network import WindFarmNetwork, GeneticAlgorithmDriver

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Læs koordinater fra CSV-filen
coordinates_df = pd.read_csv('/content/drive/My Drive/Colab Notebooks/Bachelor Project/Final cable plots/Optimized_coordinates_S8AT.csv')
optimized_x = coordinates_df['x'].to_numpy()
optimized_y = coordinates_df['y'].to_numpy()

# Initialiser turb_init med x- og y-koordinaterne
turb_init = np.asarray([optimized_x, optimized_y]).T

#Fixed position for the substation, where it is planned
substations_pos = np.asarray([[421344.95], [6246613.734]]).T

#Three types of cable, [thikness, n_wt, cost]
cables = np.array([[240, 4, 118], [500, 7, 170], [1000, 10, 270]]) # 66 kV

def cable_position(turb_init, **kwargs):
    L = 0
    G = None

    try:
        wfn = WindFarmNetwork(turbines_pos=turb_init, substations_pos=substations_pos, cables=cables)
        G = wfn.optimize()
        print('Solved with Heuristics')
    except Exception as e:
        print(f'Heuristics failed: {e}. Trying Genetic Algorithm...')
        wfn = WindFarmNetwork(turbines_pos=turb_init, substations_pos=substations_pos, cables=cables, drivers=[GeneticAlgorithmDriver()])
        G = wfn.optimize()
        print('Solved with Genetic Algorithm')

    L = G.size(weight="length")
    print('Total length is:', L, 'meters')

    return L / 1000, G  # Return both total cable length in kilometers and the optimized graph

# Calculate cable length and get the graph using turb_init
total_cable_length, G = cable_position(turb_init)
print(f'Total cable length in kilometers: {total_cable_length}')

G.plot()