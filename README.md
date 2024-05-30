This is a github repository which includes code and data files from the bachelor thesis 'Influence of Global Warming Potential taxation on Levelized Cost of Energy and Life Cycle Assessment of Off-shore Wind
Farms'. 

Authors: Emma Koch Jørgensen and Freja Rølle Jakobsen

The files in this repository is made up of CSV and Python files.

CSV files:
- 'UTM_coordinates_Horn.csv' contains latitude and longitude coordinates, elevation, UTM easting and UTM northing coordinates. Obtained from EMODnet.
- 'UTM_filter_coordinates_Thor.csv' contains latitude and longitude coordinates, elevation, UTM easting and UTM northing coordinates. Obtained from EMODnet, but filtered to the specific site.
- 'Optimized_coordinates_3a_Horn.csv' contains UTM coordinates of wind turbines positions after optimization on scenario 3a for Horns Rev 1
- 'Optimized_coordinates_3b_Horn.csv' contains UTM coordinates of wind turbines positions after optimization on scenario 3b for Horns Rev 1
- 'Optimized_coordinates_3a_Thor.csv' contains UTM coordinates of wind turbines positions after optimization on scenario 3a for Thor Wind Farm
- 'Optimized_coordinates_3b_Thor.csv' contains UTM coordinates of wind turbines positions after optimization on scenario 3b for Thor Wind Farm

Python files:
- '1_Horn.py' contains the code for scenario 1, an optimization of LCoE at Horns Rev 1, focusing solely on steel usage in the foundation.
- '2_Horn.py' contains the code for scenario 2, an optimization of LCoE at Horns Rev 1, including a CO2eq tax to the total steel mass.
- '3a_Horn.py' contains the code for scenario 3a, an optimization of LCoE at Horns Rev 1, incorporating inter-array cables with the current substation position, and including CO2eq tax in the total steel mass and cables.
- '3b_Horn.py' contains the code for scenario 3b, an optimization of LCoE at Horns Rev 1, incorporating inter-array cables with a our proposed substation position, and including CO2eq tax in the total steel mass and cables. 
- '1_Thor.py' contains the code for scenario 1, an optimization of LCoE at Thor Wind Farm, focusing solely on steel usage in the foundation.
- '2_Thor.py' contains the code for scenario 2, an optimization of LCoE at Thor Wind Farm, including a CO2eq tax to the total steel mass.
- '3a_Thor.py' contains the code for scenario 3a, an optimization of LCoE at Thor Wind Farm, incorporating inter-array cables with a planned substation position, and including CO2eq tax in the total steel mass and cables.
- '3b_Thor.py' contains the code for scenario 3b, an optimization of LCoE at Thor Wind Farm, incorporating inter-array cables with our proposed substation position, and including CO2eq tax in the total steel mass and cables.
- '3a_horn_cables.py' contains the code for plotting the cable layout in scenario 3a for Horns Rev 1. It is configured to run on Google Colab and requires access to the respective CSV file.
- '3b_horn_cables.py' contains the code for plotting the cable layout in scenario 3b for Horns Rev 1. It is configured to run on Google Colab and requires access to the respective CSV file.
- '3a_thor_cables.py' contains the code for plotting the cable layout in scenario 3a for Thor Wind Farm. It is configured to run on Google Colab and requires access to the respective CSV file.
- '3b_thor_cables.py' contains the code for plotting the cable layout in scenario 3b for Thor Wind Farm. It is configured to run on Google Colab and requires access to the respective CSV file.
