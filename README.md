This is a github repository which includes code and data files from the bachelor thesis 'Influence of Global Warming Potential taxation on Levelized Cost of Energy and Life Cycle Assessment of Off-shore Wind
Farms'. 

Authors: Emma Koch Jørgensen and Freja Rølle Jakobsen

The files in this repository is made up of CSV and Python files.

CSV files:
- 'UTM_coordinates_Horn.csv' contains latitude and longitude coordinates, elevation, UTM easting and UTM northing coordinates. Obtained from EMODnet.
- 'UTM_filter_coordinates_Thor.csv' contains latitude and longitude coordinates, elevation, UTM easting and UTM northing coordinates. Obtained from EMODnet, but filtered to the specific site.
- 'Optimized_coordinates_S3HR1.csv' contains UTM coordinates of wind turbines positions after optimization on scenario 3 Horn Base + Cables.
- 'Optimized_coordinates_S4AHR1.csv' contains UTM coordinates of wind turbines positions after optimization on scenario 4A Horn Base + CO2eq tax + Cables.
- 'Optimized_coordinates_S4BHR1.csv' contains UTM coordinates of wind turbines positions after optimization on scenario 4B Horn Base + CO2eq tax + Cables.
- 'Optimized_coordinates_S7T.csv' contains UTM coordinates of wind turbines positions after optimization on scenario 7 Thor Base + Cables.
- 'Optimized_coordinates_S8AT.csv' contains UTM coordinates of wind turbines positions after optimization on scenario 4A Thor Base + CO2eq tax + Cables.
- 'Optimized_coordinates_S4BT.csv' contains UTM coordinates of wind turbines positions after optimization on scenario 4B Thor Base + CO2eq tax + Cables.


Python files:
- 'S1_HR1.py' contains the code for scenario 1 Horn Base, standard wind farm optimization, only using cost of operation and maintenance and monopiles.
- 'S2_HR1.py' contains the code for scenario 2 Horn Base + CO2eq tax, wind farm optimization adding a CO2 equivalent tax on the steel mass of the wind farm.
- 'S3_HR1.py' contains the code for scenario 3 Horn Base + Cables, standard wind farm optimization including the cost and optimization of cables, with the current substation position
- 'S3_HR1_cables.py' contains the code for plotting the cable layout in scenario 3 Horn Base + Cables. It is configured to run on Google Colab and requires access to the respective CSV file.
- 'S4A_HR1.py' contains the code for scenario 4A Horn Base + CO2eq tax + Cables, wind farm optimization adding CO2 equivalent tax on both the steel mass and the cables, using the current substation position. 
- 'S4A_HR1_cables.py' contains the code for plotting the cable layout in scenario 4A Horn Base + CO2eq tax + Cables. It is configured to run on Google Colab and requires access to the respective CSV file.
- 'S4B_HR1.py' contains the code for scenario 4B Horn Base + CO2eq tax + Cables, wind farm optimization adding CO2 equivalent tax on both the steel mass and the cables, using our best guess for a substation position
- 'S4B_HR1_cables.py' contains the code for plotting the cable layout in scenario 4B Horn Base + CO2eq tax + Cables. It is configured to run on Google Colab and requires access to the respective CSV file.
  
- 'S5_T.py' contains the code for scenario 5 Thor Base, standard wind farm optimization, only using cost of operation and maintenance and monopiles.
- 'S6_T.py' contains the code for scenario 6 Thor Base + CO2eq tax, wind farm optimization adding a CO2 equivalent tax on the steel mass of the wind farm
- 'S7_T.py' contains the code for scenario 7 Thor Base + Cables, standard wind farm optimization including the cost and optimization of cables, with planned substation position.
- 'S7_T_cables.py' contains the code for plotting the cable layout in scenario 7 Thor Base + Cables. It is configured to run on Google Colab and requires access to the respective CSV file.
- 'S8A_T.py' contains the code for scenario 8A Thor Base + CO2eq tax + Cables, wind farm optimization adding CO2 equivalent tax on both the steel mass and the cables, using the planned substation position
- 'S8A_T_cables.py' contains the code for plotting the cable layout in scenario 8A Thor Base + CO2eq tax + Cables. It is configured to run on Google Colab and requires access to the respective CSV file.
- 'S8B_T.py' contains the code for scenario 8B Thor Base + CO2eq tax + Cables, wind farm optimization adding CO2 equivalent tax on both the steel mass and the cables, using our best guess for a substation position
- 'S8B_T_cables.py' contains the code for plotting the cable layout in scenario 8B Thor Base + CO2eq tax + Cables. It is configured to run on Google Colab and requires access to the respective CSV file.
