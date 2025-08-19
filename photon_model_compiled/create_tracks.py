import vtk
import random
import openmc
import numpy as np
import sys
import os


def get_points(tracks_folder, n_particles):

    # Starting positions of all particles
    positions = np.ndarray(shape=(n_particles, 3))

    # Dictionary where key = particle_id, value = mpi_rank
    particle_dict = {}

    # Get all track files from a folder and get the absolute paths to those files
    track_files = []

    for dirpath, _, filenames in os.walk(tracks_folder):
        for f in filenames:
            track_files.append(os.path.abspath(os.path.join(dirpath, f)))
    # ---n_mpi_ranks must be the same as the number of files in the dir
    n_mpi_ranks = len(track_files)

    # --- Make it so we take an equal (or almost equal) number of particles from each track files
    particles_per_rank = [n_particles//n_mpi_ranks] * n_mpi_ranks
    remainder = n_particles % n_mpi_ranks
    for i in range(remainder):
        particles_per_rank[i] += 1

    particle_id = 0
    for rank, file in enumerate(track_files):
        tracks = openmc.Tracks(file)
        for i in range(particles_per_rank[rank]):
            track = tracks[i]
            # Get the starting position of the first particle in the particle track,
            # probably the only particle in our case
            starting_pos = track.particle_tracks[0].states['r'][0]
            positions[particle_id, 0] = starting_pos[0]
            positions[particle_id, 1] = starting_pos[1]
            positions[particle_id, 2] = starting_pos[2]
            particle_dict[particle_id] = rank
            # print(particle_id, " ", rank)
            particle_id += 1
    return particle_dict, positions, n_mpi_ranks


tracks_folder = sys.argv[1]

particle_dict, positions, num_mpi_tasks = get_points(tracks_folder, 8000)

# --- Parameters ---
num_points = len(positions)
output_filename = "particles_compiled_" + str(num_mpi_tasks) + ".vtp"
# --- Create Points ---
points = vtk.vtkPoints()
for i in range(num_points):
    # x, y, z = random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)
    x, y, z = float(positions[i][0]), float(
        positions[i][1]), float(positions[i][2])
    points.InsertNextPoint(x, y, z)

# --- Create Vertices ---
vertices = vtk.vtkCellArray()
for i in range(num_points):
    vertices.InsertNextCell(1)
    vertices.InsertCellPoint(i)

# --- Create Scalar Array ---
scalars = vtk.vtkFloatArray()
scalars.SetName("rank")
scalars.SetNumberOfComponents(1)

# Populate scalars: real values for first 5, -1.0 for rest
for i in range(num_points):
    scalars.InsertNextValue(particle_dict[i])

# --- Create PolyData ---
polydata = vtk.vtkPolyData()
polydata.SetPoints(points)
polydata.SetVerts(vertices)

# CRITICAL: Add scalar data explicitly to point data
polydata.GetPointData().AddArray(scalars)
polydata.GetPointData().SetActiveScalars(
    "rank")  # Make sure it's active

# --- Write to .vtp (binary format) ---
writer = vtk.vtkXMLPolyDataWriter()
writer.SetFileName(output_filename)
writer.SetInputData(polydata)
# writer.SetDataModeToAppended()     # binary mode
# writer.EncodeAppendedDataOn()
writer.SetDataModeToAscii()
# writer.SetCompressorTypeToNone()
writer.Write()

print(f"Successfully wrote: {output_filename}")
