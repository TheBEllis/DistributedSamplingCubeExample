#!/usr/env/python3

import openmc
import openmc.lib
import re
import os
import sys
import csv
import h5py
import numpy as np
import matplotlib.pyplot as plt

# --- number of elems in mesh tallys ---
mesh_elems = 384

#
compiled_data_path = "./photon_model_compiled/comparison_results/"
meshsource_data_path = "./photon_model/comparison_results/"
particle_counts = []

# --- number of files we are analysing ---
n_files = len([f for f in os.listdir(compiled_data_path) if os.path.isfile(
    os.path.join(compiled_data_path, f))])


# --- np ndarrays to store tally results ---
compiled_results = np.ndarray(shape=(mesh_elems, n_files))
meshsource_results = np.ndarray(shape=(mesh_elems, n_files))


# --- loop over all results files and store data from flux tally ---
for dirpath, _, filenames in os.walk(compiled_data_path):
    for i, f in enumerate(filenames):
        particle_counts.append(int((re.findall(r'\d+', f))[0]))
        sp = openmc.StatePoint(os.path.abspath(os.path.join(dirpath, f)))
        flux_tally = sp.get_tally(name='photon_flux')
        flux = flux_tally.get_values(scores=['flux'], filters=[
            openmc.ParticleFilter], filter_bins=[('photon',)])
        compiled_results[:, i] = np.squeeze(flux[:, :, :])

# --- loop over all results files and store data from flux tally, mesh source results ---
for dirpath, _, filenames in os.walk(meshsource_data_path):
    for i, f in enumerate(filenames):
        sp = openmc.StatePoint(os.path.abspath(os.path.join(dirpath, f)))
        flux_tally = sp.get_tally(name='photon_flux')
        flux = flux_tally.get_values(scores=['flux'], filters=[
            openmc.ParticleFilter], filter_bins=[('photon',)])
        meshsource_results[:, i] = np.squeeze(flux[:, :, :])

# --- difference array ---
differences = compiled_results - meshsource_results

# ---calculate percentage differences---
perc_diff = np.divide(np.abs(differences), np.abs(compiled_results), out=np.full_like(
    differences, np.nan, dtype=float), where=compiled_results != 0)

# --- get the mean percentage difference over all element bins ---
perc_diff_means = np.nanmean(perc_diff, axis=0, keepdims=True)
perc_diff_means *= 100

# --- reorder percentage_differences_means by ascending particle counts ---
indices = np.argsort(particle_counts)
reordered_percentage_diff = np.squeeze(perc_diff_means)[indices]
particle_counts.sort()

# Create the plot
plt.figure(figsize=(8, 5))

# Plot the line connecting points
plt.plot(particle_counts, reordered_percentage_diff,
         linestyle='-', color='blue', label='Line')

# Plot the scatter points on top
plt.scatter(particle_counts, reordered_percentage_diff,
            color='red', s=50, label='Points')

# Add labels and title
plt.xlabel('Particle Histories')
plt.ylabel('% diff')
plt.xscale('log')
plt.yscale('log')
plt.title('Percentage Difference Between Mesh Source and Compiled Source')
# plt.legend()
plt.grid(True)

# Show the plot
plt.savefig(fname="PercDiffFlux.png", dpi=300)
