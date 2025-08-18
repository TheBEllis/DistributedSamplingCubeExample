#!/usr/env/python3

import openmc
import openmc.lib
import os
import sys
import csv
import h5py


def create_photon_source(bins_file, mesh):
    photon_bins = []
    photon_sources = []

    centroid_x = []

    with open("./centroids.csv", 'r', newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            for i, bin in enumerate(row):
                centroid_x.append(float(bin))

    print(centroid_x)
    with open(bins_file, 'r', newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            for i, bin in enumerate(row):
                photon_bins.append(float(bin))

    for i in range(0, 127):

        photon_spectra = []
        for j in range(0, 24):
            # photon_spectra.append(centroid_x[i])
            photon_spectra.append(1)

        photon_dist = openmc.stats.Tabular(
            photon_bins, photon_spectra, 'histogram')
        source = openmc.IndependentSource()
        source.energy = photon_dist
        source.particle = 'photon'
        # source.strength = 1
        source.strength = 24 * centroid_x[i]
        photon_sources.append(source)

    mesh_source = openmc.MeshSource(mesh, photon_sources)
    print("STRENGTH")
    print(mesh_source.strength)
    mesh_source.set_total_strength(1)
    return mesh_source


def make_model(temperature=296):

    model = openmc.Model()
    model.settings.dagmc = True
    # lead at room temperature # molten lead should be 10.66
    # steel
    steel = openmc.Material(name="steel", temperature=temperature)
    steel.add_nuclide("C13", 1.51325480e-05, "ao")
    steel.add_nuclide("C12", 1.35086745e-03, "ao")
    steel.add_nuclide("Mn55", 1.99100000e-02, "ao")
    steel.add_nuclide("P31", 7.95000000e-04, "ao")
    steel.add_nuclide("S34", 2.14834688e-05, "ao")
    steel.add_nuclide("S32", 4.86608589e-04, "ao")
    steel.add_nuclide("S33", 3.83329280e-06, "ao")
    steel.add_nuclide("S36", 7.46496000e-08, "ao")
    steel.add_nuclide("Si28", 1.79598856e-02, "ao")
    steel.add_nuclide("Si29", 9.11951747e-04, "ao")
    steel.add_nuclide("Si30", 6.01162667e-04, "ao")
    steel.add_nuclide("Cr53", 1.89874635e-02, "ao")
    steel.add_nuclide("Cr50", 8.68335215e-03, "ao")
    steel.add_nuclide("Cr52", 1.67449803e-01, "ao")
    steel.add_nuclide("Cr54", 4.72638155e-03, "ao")
    steel.add_nuclide("Ni62", 3.38662710e-03, "ao")
    steel.add_nuclide("Ni60", 2.44346846e-02, "ao")
    steel.add_nuclide("Ni64", 8.62474080e-04, "ao")
    steel.add_nuclide("Ni58", 6.34340554e-02, "ao")
    steel.add_nuclide("Ni61", 1.06215882e-03, "ao")
    steel.add_nuclide("Fe56", 6.10087944e-01, "ao")
    steel.add_nuclide("Fe58", 1.87506594e-03, "ao")
    steel.add_nuclide("Fe57", 1.40895912e-02, "ao")
    steel.add_nuclide("Fe54", 3.88643986e-02, "ao")
    steel.set_density("g/cm3", 8.0)
    # fe56 doesnt go high enough
    # steel.add_s_alpha_beta('c_Fe56',0.61)

    materials = openmc.Materials()

    materials.append(steel)

    model.materials = materials

    surf = openmc.Sphere(r=250, surface_id=99999, boundary_type="vacuum")
    dag = openmc.DAGMCUniverse(
        "/home/bill/Projects/PyFIS/cube_example/mesh/cube_dag.h5m", name='daguni', universe_id=42)
    cell = openmc.Cell(fill=dag, region=-surf, cell_id=99999)

    model.geometry = openmc.Geometry([cell])

    # tallies
    # Create a mesh that will be used for tallying
    mesh = openmc.UnstructuredMesh(
        filename='/home/bill/Projects/PyFIS/cube_example/mesh/cube.e', library='libmesh', mesh_id=1)

    # Create a mesh filter that can be used in a tally
    mesh_filter = openmc.MeshFilter(mesh)
    mesh_born_filter = openmc.MeshBornFilter(mesh)

    # score in the breeding regions the neutron spectrum
    photon_filter = openmc.ParticleFilter("photon")
    photon_flux_tally = openmc.Tally(name="photon_flux")
    photon_flux_tally.filters = [mesh_filter, photon_filter]
    photon_flux_tally.scores = ['flux']
    photon_flux_tally.estimator = 'collision'

    photon_flux_born_tally = openmc.Tally(name="photon_flux_born")
    photon_flux_born_tally.filters = [mesh_born_filter, photon_filter]
    photon_flux_born_tally.scores = ['flux']
    photon_flux_born_tally.estimator = 'collision'

    photon_absorption_tally = openmc.Tally(name="photon_absorption")
    photon_absorption_tally.filters = [mesh_filter, photon_filter]
    photon_absorption_tally.scores = ['absorption']
    photon_absorption_tally.estimator = 'collision'

    model.tallies = [photon_flux_tally,
                     photon_absorption_tally, photon_flux_born_tally]

    # settings
    # Indicate how many particles to run

    model.settings.batches = 20
    model.settings.inactive = 0
    # 1e10 histories per batch
    model.settings.particles = 50000
    model.settings.photon_transport = True
    model.settings.electron_treatment = 'ttb'
    model.settings.run_mode = 'fixed source'
    model.settings.energy_mode = 'continuous-energy'

    # Create mesh source
    model.settings.source = create_photon_source(
        './photon_bins.csv', mesh)

    model.settings.particle = ['photon']

    return model


model = make_model()
model.export_to_xml("./photon_model.xml")
num_mpi_tasks = sys.argv[1]
sp_path = model.run(mpi_args=['mpiexec', '-n', str(num_mpi_tasks)])

sp = openmc.StatePoint(
    sp_path)

flux_tally = sp.get_tally(name='photon_flux')
flux_born_tally = sp.get_tally(name='photon_flux_born')
absorption_tally = sp.get_tally(name='photon_absorption')

flux = flux_tally.get_values(scores=['flux'], filters=[
    openmc.ParticleFilter], filter_bins=[('photon',)])

flux_born = flux_born_tally.get_values(scores=['flux'], filters=[
    openmc.ParticleFilter], filter_bins=[('photon',)])

absorption = absorption_tally.get_values(scores=['absorption'], filters=[
    openmc.ParticleFilter], filter_bins=[('photon',)])

mesh = sp.meshes[1]
n_vols = len(mesh.volumes)
flux.resize(n_vols)
absorption.resize(n_vols)
flux_born.resize(n_vols)
data_dict = {'photon_flux': flux,
             'photon_absorption': absorption,
             'photon_flux_born': flux_born}

out_name = "./photons_" + str(num_mpi_tasks) + ".vtk"

mesh.write_data_to_vtk(out_name, data_dict)
