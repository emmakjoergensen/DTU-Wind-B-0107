#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 24 11:55:59 2024

@author: emmakochjorgensen
"""
#-------SCENARIO 3B  HORNS REV 1-------#
#%% IMPORT ALL REQUIRED PACKAGES

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime

#%% IMPORT ALL REQUIRED FUNCTIONS

from matplotlib.patches import Circle
from scipy.interpolate import Rbf
from openmdao.api import n2
from topfarm._topfarm import TopFarmProblem, TopFarmGroup
from topfarm.cost_models.cost_model_wrappers import CostModelComponent
from topfarm.plotting import XYPlotComp
from topfarm.cost_models.py_wake_wrapper import PyWakeAEPCostModelComponent
from topfarm import SpacingConstraint, XYBoundaryConstraint
from py_wake.examples.data.hornsrev1 import wt_x, wt_y, HornsrevV80, Hornsrev1Site
from py_wake import NOJ
from py_wake.utils.gradients import autograd
from ed_win.wind_farm_network import WindFarmNetwork, GeneticAlgorithmDriver
from tabulate import tabulate
from topfarm.easy_drivers import EasyScipyOptimizeDriver
from topfarm.drivers.random_search_driver import RandomizeTurbinePosition_Circle
from topfarm.easy_drivers import EasyRandomSearchDriver

#%% SETUP SITE DATA

# Load data
df_Horn = pd.read_csv('UTM_coordinates_Horn.csv', delimiter=',')

# Site boundary
x_site = np.array([df_Horn['utm_easting'].min(), df_Horn['utm_easting'].max(), df_Horn['utm_easting'].min(), df_Horn['utm_easting'].max()])
y_site = np.array([df_Horn['utm_northing'].min(), df_Horn['utm_northing'].max(), df_Horn['utm_northing'].min(), df_Horn['utm_northing'].max()])

#%% SETUP OF WIND FARM MODEL

# Set number of turbines
num_turbines = 80

# Generate random x and y coordinates within the site boundaries
x_init = np.random.uniform(df_Horn['utm_easting'].min(), df_Horn['utm_easting'].max(), size=num_turbines)
y_init = np.random.uniform(df_Horn['utm_northing'].min(), df_Horn['utm_northing'].max(), size=num_turbines)

# Print coordinates for each turbine
print("Coordinates for each turbine:")
for i, (x, y) in enumerate(zip(x_init, y_init), start=1):
    print(f"Turbine {i}: (x={x}, y={y})")

#%% FIXED VALUES

n_wt = len(x_init) #Number of turbines
wt = HornsrevV80() #WT model (the same as in Hornsrev1)
site = Hornsrev1Site() #Define the site
wf_model = NOJ(site, wt) #N.O. Jensen wake model for the WF
aep = wf_model(wt_x, wt_y).aep() #AEP model of this WF

#Fixed position for the substation
substations_pos = np.asarray([[428167], [6149224]]).T

#Three types of cable, [thikness, n_wt, cost]
cables = np.array([[95, 4, 85], [150, 8, 125], [400, 16, 240]])

rated_power = 2 # MW
omega = 0.0475 # interest rate 
LT = 20 # Life time of WF, years
cost_WT = 1.1*10**6 # €/MW for the WT
CRF = omega / (1-(1+omega)**(-LT)) # recovery factor
tax1 = 750 * 0.13 # Tax rates pr ton of CO2
cost_fixed = 40*1000*n_wt*rated_power # €/y
cost_var = 3.0 # €/MWh

#%% MASS OF NACELLE AND TOWER

power_kW = rated_power*1000 # Rated power in kW

#Calculations for the tower mass, from INNWIND model
tower_mass = (power_kW/10000)**(2.5/2)*628500*n_wt # kg

#Calculations for the nacelle mass, from INNWIND model
low_speed_shaft = (power_kW/5000)**(3/2)*27210
main_bearing = 0.0092*(8/600*wt.diameter()-0.033)*(wt.diameter()**2.5)*2
mechanical_brake_couplings = (power_kW/5000)**(3/2)*1000
bed_plate = wt.diameter()**1.953 *1.228
hydraulic_cooling_system = 0.08*power_kW
nacelle_cover = 13000*(power_kW/5000)**(2/2)
yaw_system = wt.diameter()**3.314 *0.0009*1.6
nacelle_mass = (low_speed_shaft+main_bearing+mechanical_brake_couplings+bed_plate+hydraulic_cooling_system+nacelle_cover+yaw_system)*n_wt

#%% SETUP OF WRAPPER FUNCTIONS

# Find water depth at x,y, using Thin Plate Spline interpolation
def water_depth_func(x, y, df):
    # Interpolate water depth from seabed elevation data
    points = df[['utm_easting', 'utm_northing']].values
    values = -df['elevation'].values  # Depth is negative of elevation

    # Perform Thin Plate Spline interpolation
    rbf = Rbf(points[:, 0], points[:, 1], values, function='thin_plate')
    water_depth = rbf(x, y)

    return np.maximum(0, water_depth)  # Ensure depth is nonnegative

# Find steel mass at given water depth
def calculate_steel_mass(water_depth, **kwargs):
  # Calculations for the steel in the monopile and transition piece
  a = 21.3 * 1000
  b = 137.5 * 1000
  
  # Calculate cost using the lowest elevation
  monopile_cost = (a * water_depth + b) * rated_power # €
  
  # Calculate mass of steel
  monopile_mass1 = monopile_cost / 8.1  # €/kg

  # Sum monopile mass and cost for all water depths
  monopile_mass = np.sum(monopile_mass1)
  monopile_cost = np.sum(monopile_cost)

  # Calculate total steel mass
  steel_mass = monopile_mass + tower_mass + nacelle_mass # kg

  return [steel_mass, monopile_cost]

# Find length and cost of inter array cables, at turbine locations
def cable_position(x, y, **kwargs):
    L = 0
    turb_pos = np.asarray([x, y]).T

    try:
        wfn = WindFarmNetwork(turbines_pos=turb_pos, substations_pos=substations_pos, cables=cables) # Solve with heuristic algorithm
        G = wfn.optimize()
    except:
        wfn = WindFarmNetwork(turbines_pos=turb_pos, substations_pos=substations_pos, cables=cables, drivers=[GeneticAlgorithmDriver()]) # Solve with genetic algorithm
        G = wfn.optimize()
    
    L = G.size(weight="length") # Save total cable length
    cost_cable = G.size(weight="cost") # Save cost of total cable length

    return [L, cost_cable] 

# Find CO2eq of steel and cables and tax it
def cost_emissions(steel_mass, cable_length, **kwargs):
  # Calculates mass of CO2e from emission factor on steel
  CO2e_sup =  3.62 * (steel_mass/1000) # t CO2e/t

  # Calculates mass of CO2e from emission factor on cables
  CO2e_cab = (42.6941 * cable_length)/1000 # t CO2e/km
  
  CO2e = CO2e_sup + CO2e_cab # Calculate total mass of CO2e 

  CO2e_tax = CO2e * tax1 # Tax it
  return [CO2e, CO2e_tax]

#%% SETUP OF ECONOMICAL MODEL

# Find LCOE CAPEX from values of wrapper functions
def LCOE_CAPEX(AEP, CO2e_tax, monopile_cost, cable_length, cost_cable, **kwargs):
  AEP = AEP * 1000  # MW

  # Calculate cost of all capital expenditures (incl. CO2e tax)
  cost_CAPEX = cost_WT * n_wt * rated_power + CO2e_tax + monopile_cost + cost_cable # €

  # Calculate the LCOE CAPEX from the wind farm
  LCOE_CAPEX = ((cost_CAPEX)/(AEP))*CRF  # €/MWh

  return [LCOE_CAPEX, cost_CAPEX]

# Find LCOE OPEX
def LCOE_OPEX(AEP, **kwargs):
  AEP = AEP * 1000 # MW
  LCOE_OPEX = (cost_fixed)/AEP + cost_var # €/MWh
  return LCOE_OPEX

# Find LCOE
def calculate_LCOE(LCOE_OPEX, LCOE_CAPEX, **kwargs):
  LCOE = LCOE_CAPEX + LCOE_OPEX # €/MWh
  return LCOE


#%% SETUP OF COMPONENTS

aep_component = PyWakeAEPCostModelComponent(wf_model, n_wt, grad_method=autograd, objective=False)

water_depth_component = CostModelComponent(input_keys=[('x', x_init), ('y', y_init)],
                                           n_wt=n_wt,
                                           cost_function=lambda x, y: water_depth_func(x, y, df=df_Horn),
                                           objective=False,
                                           output_keys=[('water_depth', np.zeros(n_wt))])


support_structure_component = CostModelComponent(input_keys=[('water_depth', 30*np.ones(n_wt))],
                                          n_wt=n_wt,
                                          cost_function=calculate_steel_mass,
                                          objective=False,
                                          output_keys=[('steel_mass', 0),('monopile_cost',0)])


cable_position_component = CostModelComponent(input_keys=[('x', x_init), ('y', y_init)],
                                           n_wt=n_wt,
                                           cost_function= cable_position,
                                           objective=False,
                                           output_keys=[('cable_length', 0), ('cost_cable', 0)])


cost_emissions_component = CostModelComponent(input_keys=[('steel_mass', 0), ('cable_length', 0)],
                                          n_wt=n_wt,
                                          cost_function=cost_emissions,
                                          objective=False,
                                          output_keys=[('CO2e', 0),('CO2e_tax', 0)])


LCOE_CAPEX_component = CostModelComponent(input_keys=[('AEP',0), ('CO2e_tax', 0), ('monopile_cost',0), ('cable_length', 0), ('cost_cable',0)],
                                          n_wt=n_wt,
                                          cost_function=LCOE_CAPEX,
                                          objective=False,
                                          output_keys=[('LCOE_CAPEX',0),('cost_CAPEX',0)])

LCOE_OPEX_component = CostModelComponent(input_keys=[('AEP', 0)],
                                          n_wt=n_wt,
                                          cost_function=LCOE_OPEX,
                                          objective=False,
                                          output_keys=[('LCOE_OPEX',0)])

LCOE_component = CostModelComponent(input_keys=[('LCOE_CAPEX', 0), ('LCOE_OPEX', 0)],
                              n_wt=n_wt,
                              cost_function=calculate_LCOE,
                              objective=True,
                              maximize=False,
                              output_keys=[('LCOE', 0)])

cost_comp = TopFarmGroup([aep_component, water_depth_component, cable_position_component, support_structure_component, cost_emissions_component, LCOE_CAPEX_component, LCOE_OPEX_component, LCOE_component])


#%%## SETUP OF PROBLEM WITH RANDOM SEARCH
# Random Search algorithm
problem_RS = TopFarmProblem(design_vars={'x': x_init, 'y': y_init},
                         driver=EasyRandomSearchDriver(randomize_func=RandomizeTurbinePosition_Circle(max_step=6400), max_time=10000, disp=False),
                  cost_comp=cost_comp,
                  constraints=[XYBoundaryConstraint(np.asarray([x_site, y_site]).T, boundary_type='rectangle'),
                               SpacingConstraint(4 * wt.diameter())],
                  plot_comp=XYPlotComp())

#%% SMART START

# Make grid of 100x100
xs = np.linspace(df_Horn['utm_easting'].min(), df_Horn['utm_easting'].max(), 100)
ys = np.linspace(df_Horn['utm_northing'].min(), df_Horn['utm_northing'].max(), 100)

XX, YY = np.meshgrid(xs, ys)

res = problem_RS.smart_start(XX, YY, aep_component.get_aep4smart_start(ws=9.5))

#%% EVALUATE RANDOM SEARCH 

# Evaluate problem including Smart Start positions
problem_RS.evaluate()

#%% OPTIMIZE WITH RANDOM SEARCH

costRS, stateRS, recorderRS = problem_RS.optimize()

#%% SAVE RANDOM SEARCH RECORDER

t = datetime.datetime.now()
time_stamp = t.strftime('%Y_%m_%d_%H_%M')

# Save recorder to file
recorderRS.save('Scenario3b_Horn' + time_stamp + '_RS')

# Make N2 diagram
n2(problem_RS, embeddable=True) 

#%%## SETUP OF PROBLEM WITH SLSQP

# Sequential Least SQuare Programming
problem_SLSQP = TopFarmProblem(design_vars={'x': stateRS['x'], 'y': stateRS['y']},
                         driver=EasyScipyOptimizeDriver(optimizer='SLSQP', maxiter=200, tol=1e-6, disp=True),
                  cost_comp=cost_comp,
                  constraints=[XYBoundaryConstraint(np.asarray([x_site, y_site]).T, boundary_type='rectangle'),
                              SpacingConstraint(4 * wt.diameter())],
                  plot_comp=XYPlotComp(),
                         expected_cost=1e-3)

#%% EVALUATE SLSQP

# Evaluate problem including Random Search positions
problem_SLSQP.evaluate() 

#%% OPTIMIZE WITH SLSQP

costSLSQP, stateSLSQP, recorderSLSQP = problem_SLSQP.optimize()

#%% SAVE SLSQP RECORDER

# Save recorder to file
recorderSLSQP.save('Scenario3b_Horn' + time_stamp + '_SLSQP')

# Make N2 diagram
n2(problem_SLSQP, embeddable=True)

#%% PLOTS

# Make plot of seabed including before/after positions of turbines
def plot_seabed_and_turbines(df, original_turbine_locations, optimized_turbine_locations, optimized_x, optimized_y, LCOE_optimized):
    # Plotting
    fig, ax = plt.subplots(figsize=(10, 8))

    # Surface plot
    surf = ax.tricontourf(df['utm_easting'], df['utm_northing'], df['elevation'], cmap='viridis')

    ax.set_xlabel('UTM Easting (m)', labelpad=10)
    ax.set_ylabel('UTM Northing (m)', labelpad=10)

    plt.title(f'Placement of turbines at Horns Rev 1\n Optimization of LCoE with CO2 eq.-taxation on steel and cables\n LCoE after optimization: {LCOE_optimized:.2f} €/MWh'.format(LCOE_init, LCOE_optimized), pad=10)

    # Plot original turbine locations
    original_turbine_locations = np.array(original_turbine_locations)
    ax.scatter(original_turbine_locations[:, 0], original_turbine_locations[:, 1], c='orange', marker='o', label='Random Turbine Locations', s=30)

    # Plot optimized turbine locations
    ax.scatter(optimized_x, optimized_y, c='red', marker='x', label='Optimized Turbine Locations', s=60)

    # Add connecting lines between original and optimized turbine locations
    for i in range(len(original_turbine_locations)):
        ax.plot([original_turbine_locations[i, 0], optimized_x[i]], [original_turbine_locations[i, 1], optimized_y[i]], color='red', linestyle='--')

    # Add spacing constraints as circles around optimized turbine locations
    for x, y in zip(optimized_x, optimized_y):
        constraint_circle = Circle((x, y), 2*wt.diameter(), edgecolor='black', facecolor='none')
        ax.add_patch(constraint_circle)
        constraint_circle.set_linestyle('--')

    # Set aspect ratio to equal
    ax.set_aspect('equal')

    # Add color bar
    cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label='Elevation (meters)')

    # Add legend
    plt.legend(loc='lower right')

    # Add grid lines
    ax.grid(True, linestyle='--', color='gray')

    # Format tick labels
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:.0f}'.format(x)))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0f}'.format(y)))

    plt.show()


def plot_optimized_turbine_locations(df, optimized_x, optimized_y, LCOE_optimized):
    # Plotting
    fig, ax = plt.subplots(figsize=(10, 8))

    # Surface plot
    surf = ax.tricontourf(df['utm_easting'], df['utm_northing'], df['elevation'], cmap='viridis')

    ax.set_xlabel('UTM Easting (m)', labelpad=10)
    ax.set_ylabel('UTM Northing (m)', labelpad=10)

    plt.title(f'Optimized Placement of Turbines at Horns Rev 1\n Optimization of LCoE with CO2 eq.-taxation on steel and cables\nLCoE after optimization: {LCOE_optimized:.2f} €/MWh', pad=10)

    # Plot optimized turbine locations
    ax.scatter(optimized_x, optimized_y, c='red', marker='x', label='Optimized Turbine Locations', s=100)

    # Add spacing constraints as circles around optimized turbine locations
    for x, y in zip(optimized_x, optimized_y):
        constraint_circle = Circle((x, y), 2*wt.diameter(), edgecolor='black', facecolor='none')
        ax.add_patch(constraint_circle)
        constraint_circle.set_linestyle('--')

    # Set aspect ratio to equal
    ax.set_aspect('equal')

    # Add color bar
    cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label='Elevation (meters)')

    # Add legend
    plt.legend(loc='lower right')

    # Add grid lines
    ax.grid(True, linestyle='--', color='gray')

    # Format tick labels
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:.0f}'.format(x)))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0f}'.format(y)))

    plt.show()

# Get results from saved state and recorder
optimized_x = stateSLSQP['x']
optimized_y = stateSLSQP['y']
LCOE_init = recorderRS['LCOE'][1]
LCOE_optimized = recorderRS['LCOE'][-1]

# Plot seabed and turbine locations with before/after turbine locations
plot_seabed_and_turbines(df_Horn, np.column_stack((x_init, y_init)), np.column_stack((optimized_x, optimized_y)), optimized_x, optimized_y, LCOE_optimized)

# Plot only optimized turbine locations with seabed details
plot_optimized_turbine_locations(df_Horn, optimized_x, optimized_y, LCOE_optimized)

#%% PRINT RESULTS

# Update of values
def current_state_values (x, y, recorderSLSQP, recorderRS, **kwargs):
  LCOE_part1 = (((recorderRS['CO2e_tax'][1]/(recorderRS['AEP'][1]*1000))*CRF)/recorderRS['LCOE'][1])*100 #GWh->MWh
  LCOE_part2 = (((recorderSLSQP['CO2e_tax'][-1]/(recorderSLSQP['AEP'][-1]*1000))*CRF)/recorderSLSQP['LCOE'][-1])*100 #GWh->MWh
  lev_CO2_mass1 =((recorderRS['CO2e'][1]*10**6)/(recorderRS['AEP'][1]*10**6)) #t->g, GWh->kWh
  lev_CO2_mass2 =((recorderSLSQP['CO2e'][-1]*10**6)/(recorderSLSQP['AEP'][-1]*10**6)) #t->g, GWh->kWh
  CF1 = (recorderRS['AEP'][1]*1000)/(n_wt*rated_power*8760)*100 #GWh->MWh
  CF2 = (recorderSLSQP['AEP'][-1]*1000)/(n_wt*rated_power*8760)*100 #GWh->MWh
  difference_percentage = ((recorderRS['steel_mass'][1] - recorderSLSQP['steel_mass'][-1]) / recorderRS['steel_mass'][1]) * 100

  results = [
    ["Water depth", np.sum(recorderRS['water_depth'][1])/n_wt, np.sum(recorderSLSQP['water_depth'][-1])/n_wt, "m"],
    ["AEP", recorderRS['AEP'][1] , recorderSLSQP['AEP'][-1], "Gwh"],
    ["CF", CF1, CF2, "%"],
    ["steel mass", recorderRS['steel_mass'][1], recorderSLSQP['steel_mass'][-1], "kg"],
    ["CO2e mass", recorderRS['CO2e'][1] , recorderSLSQP['CO2e'][-1] , "t"],
    ["CO2e tax", recorderRS['CO2e_tax'][1], recorderSLSQP['CO2e_tax'][-1], "€"],
    ["Cost CAPEX",  recorderRS['cost_CAPEX'][1], recorderSLSQP['cost_CAPEX'][-1], "€"],
    ["LCOE_CAPEX", recorderRS['LCOE_CAPEX'][1], recorderSLSQP['LCOE_CAPEX'][-1], "€/Mwh"],
    ["LCOE_OPEX", recorderRS['LCOE_OPEX'][1], recorderSLSQP['LCOE_OPEX'][-1], "€/Mwh"],
    ["LCOE", recorderRS['LCOE'][1], recorderSLSQP['LCOE'][-1], "€/Mwh"],
    ["Levelized CO2e mass", lev_CO2_mass1, lev_CO2_mass2, "g/kwh"],
    ["CO2 tax contribution", LCOE_part1, LCOE_part2, "%"],
    ["Steel mass decrease", 0, difference_percentage, "%"],
    ["Cable length", recorderRS['cable_length'][1], recorderSLSQP['cable_length'][-1], "m"],
    ["Cable cost", recorderRS['cost_cable'][1], recorderSLSQP['cost_cable'][-1], "€"]


  ]

  print()
  print(tabulate(results, headers=["Parameter", "Value before", "Value after" ,"Units"], tablefmt="pretty"))
  print()

print("Results")
current_state_values (stateSLSQP['x'],stateSLSQP['y'], recorderSLSQP, recorderRS)

#%% SAVE OPTIMIZED LOCATIONS TO CSV FILE

# Create dataframe
df = pd.DataFrame({
    'x': stateSLSQP['x'],
    'y': stateSLSQP['y']
})

# Save dataframe to file
df.to_csv('Optimized_coordinates_3b_Horn.csv', index=False) # Coordinates will be used to plot inter array cable layout
