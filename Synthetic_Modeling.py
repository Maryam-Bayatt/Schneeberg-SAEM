#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 16:51:16 2026

@author: Maryam Bayat
"""

import pygimli as pg
from custEM.meshgen import meshgen_utils as mu
from custEM.meshgen.meshgen_tools import BlankWorld
from custEM.inv.inv_utils import MultiFWD
import numpy as np
from custEM.meshgen.dem_interpolator import DEM
import os
from saem import CSEMSurvey, CSEMData, tools
import zipfile
import matplotlib.pyplot as plt

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # #  create synthetic data for inversion  # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def angle_between_segments(p1, p2, p3, p4):
    """
    Computes the angle between segment p1->p2 and segment p3->p4.
    Each point is a tuple (x, y) or (x, y, z).
    Returns angle in degrees.
    """

    # Vector u = p2 - p1
    u = [p2[i] - p1[i] for i in range(2)]

    # Vector v = p4 - p3
    v = [p4[i] - p3[i] for i in range(2)]

    # Dot product
    dot = sum(u[i] * v[i] for i in range(len(u)))

    # Magnitudes
    mag_u = np.sqrt(sum(u[i]**2 for i in range(len(u))))
    mag_v = np.sqrt(sum(v[i]**2 for i in range(len(v))))

    # Avoid division-by-zero
    if mag_u == 0 or mag_v == 0:
        raise ValueError("One of the segments has zero length.")

    # Compute angle in radians
    cos_theta = dot / (mag_u * mag_v)

    # Numerical stability: clamp value to [-1, 1]
    cos_theta = max(-1, min(1, cos_theta))

    theta_rad = np.arccos(cos_theta)
    #theta_deg = np.degrees(theta_rad)

    return theta_rad

def get_prism_poly_coords(points, r=5.):

    polys = []
    extra = []
    phi0 = 0.
    for j, point in enumerate(points[:-1]):
        phi = np.arctan2(points[j+1, 1] - points[j, 1], 
                         points[j+1, 0] - points[j, 0]) - np.pi/2.
        # ang2 = angle_between_segments(points[j], points[j+1],
        #                               points[j+1], points[j+2]) / 2.
        # ang2o = ang2 + np.pi/2.
        theta = (phi + phi0) / 2.
        
        print(np.rad2deg(phi), np.rad2deg(phi0), np.rad2deg(theta))
                
        if j == 0:
            polys.append([[point[0] + np.cos(phi) * r, 
                           point[1] + np.sin(phi) * r, 
                           point[2]],
                          [point[0] - np.cos(phi) * r, 
                           point[1] - np.sin(phi) * r, 
                           point[2]],
                          [point[0], 
                           point[1], 
                           point[2] + (r * np.sqrt(3))]])
            
            # shift first lower coords "inside"
            polys[0][0][0] -= np.sin(phi) * r * np.sqrt(2)
            polys[0][0][1] += np.cos(phi) * r * np.sqrt(2)
            polys[0][1][0] -= np.sin(phi) * r * np.sqrt(2)
            polys[0][1][1] += np.cos(phi) * r * np.sqrt(2)

        else:
            polys.append([[point[0] + np.cos(theta) * r, 
                           point[1] + np.sin(theta) * r,
                           point[2]],
                          [point[0] - np.cos(theta) * r,
                           point[1] - np.sin(theta) * r,
                           point[2]],
                          [point[0],
                           point[1],
                           point[2] + (r * np.sqrt(3))]])    
            
            extra.append([[point[0] + np.cos(theta) * r, 
                           point[1] + np.sin(theta) * r,
                           point[2]],
                          [point[0] - np.cos(theta) * r,
                           point[1] - np.sin(theta) * r,
                           point[2]],
                          [point[0] + np.sin(theta) * r * np.sqrt(2),
                           point[1] - np.cos(theta) * r * np.sqrt(2),
                           point[2]]])
        
        phi0 = phi
         
    # plt.plot(polys[0][0][0], polys[0][0][1], 'x')
    # plt.plot(polys[0][1][0], polys[0][1][1], 'x')
    # plt.plot(polys[0][2][0], polys[0][2][1], 'x')      
    # plt.axis('equal')
    # print(polys[0])
    # print(np.rad2deg(phi))
    point = points[-1]
    polys.append([[point[0] + np.cos(phi) * r, 
                   point[1] + np.sin(phi) * r,
                   point[2]],
                  [point[0] - np.cos(phi) * r,
                   point[1] - np.sin(phi) * r,
                   point[2]],
                  [point[0], 
                   point[1], +
                   point[2] + (r * np.sqrt(3))]])

    # shift last lower coord "inside" 
    polys[-1][0][0] += np.sin(phi) * r * np.sqrt(2)
    polys[-1][0][1] -= np.cos(phi) * r * np.sqrt(2)
    polys[-1][1][0] += np.sin(phi) * r * np.sqrt(2)
    polys[-1][1][1] -= np.cos(phi) * r * np.sqrt(2)

    return np.array(polys), np.array(extra)


def add_infra_prisms_air(M, polys, extra, cz=100.):

    nn1 = [M.Omega.createNodeWithCheck(p) for p in polys[0]]
    for j, point in enumerate(polys[1:]):
        nn2 = [M.Omega.createNodeWithCheck(p) for p in polys[j+1]]
        # close sides of prisms
        for ll in range(len(nn1) - 1):
            if ll == 0 and j < len(extra):
                n0 = M.Omega.createNodeWithCheck(extra[j][2])
                M.Omega.createTriangleFace(nn1[0], nn2[0], n0)
                M.Omega.createTriangleFace(nn1[1], nn2[1], n0)
                M.Omega.createTriangleFace(n0, nn1[1], nn1[0])
            else:
                M.Omega.createQuadrangleFace(nn1[ll], nn1[ll+1],
                                                nn2[ll+1], nn2[ll])
        M.Omega.createQuadrangleFace(nn1[-1], nn1[0], nn2[0], nn2[-1])
        nn1 = nn2
    
    M.add_marker('Powerline', [np.mean([polys[0][2][0], polys[1][2][0]]),
                           np.mean([polys[0][2][1], polys[1][2][1]]),
                           np.mean([polys[0][2][2], polys[1][2][2]]) - 1.],
                 cell_size=cz)
      
    return(M)

# fix random seeed for reproducibility
np.random.seed(99999)

# %% mesh and survey parameter definitions

xshift = - 331125.
yshift = - 5611615.

# rotation for synthetic Rx grid to match approximately with Tx direction
rot = -37.

# %% create world

dem=DEM('schneeberg.asc', easting_shift=xshift, northing_shift=yshift) # DEM file

mesh = "Fault_Infra"

dim = 1e4
M = BlankWorld(name=mesh,
                x_dim=[-dim, dim],
                y_dim=[-dim, dim],
                z_dim=[-dim, dim],
                t_dir='.',
                topo = 'schneeberg.asc', 
                easting_shift = xshift,
                northing_shift= yshift,
                centering = False, 
                inner_area='box',    
                inner_area_size=[5e3, 35e2], 
                outer_area_cell_size=1e5,
                inner_area_cell_size=1e4,
                z_approx=900.,
                preserve_edges=True,
                )

#%% txs 

tx1_coords = tools.readCoordsFromKML(xmlfile="TX1.kml", proj='utm')
tx1 = np.column_stack((tx1_coords[0], tx1_coords[1], np.zeros(len(tx1_coords[0]))))

tx1[:, 0] += xshift
tx1[:, 1] += yshift
tx1[:, 2] = M.get_topo_vals(tx1, z=0.)
tx1 = mu.refine_path(tx1, length=20.)
txs= [tx1]

#%% Reading data & Rx

dirc1= "/home/bayat/schneeberg/Tx1/"

data32 = CSEMData(dirc1 + "Tx1_data/tfN32/*.mat", txPos= dirc1 + "TX1.kml", zone= 33)
data4 = CSEMData(dirc1 + "Tx1_data/tfN4/*.mat", txPos= dirc1 + "TX1.kml", zone= 33)

data32.filter(minTxDist=1000., maxTxDist=5000)

data4.filter(every=8)
data4.filter(minTxDist=0., maxTxDist=1000.)

data4.radius=data32.radius
data4.addData(data32)
data = data4

# ---- Data filtering for inversion ---- #
# data.filter(line=31)
# data.filter(minTxDist=400)
# data.oringin = np.array([331125., 5611615.])
# data.filter(polygon="Powerlines.kml", minTxDist=180)
# data.filter(polygon="GasPipe.kml", minTxDist=180)

rx_air = np.zeros((len(data.rx), 3))
rx_air[:, 0] = data.rx
rx_air[:, 1] = data.ry

rx_air[:,0] += xshift
rx_air[:,1] += yshift

rx_line=data.line

data.saveData("TX1_Data.npz")

#data.showPos()
# for ln in np.unique(data.line):
#     fig, ax = plt.subplots(figsize=(8, 6))

#     for all_ln in np.unique(data.line):
#         idx = np.where(data.line == all_ln)[0]
#         ax.scatter(data.rx[idx], data.ry[idx],
#                    color="lightgray", s=10, zorder=1) # all lines in gray

#     idx_hl = np.where(data.line == ln)[0]
#     ax.scatter(data.rx[idx_hl], data.ry[idx_hl],
#                color="red", s=20, zorder=2, label=f"Line {ln}") # current line in red

#     ax.set_aspect("equal")
#     ax.set_xlabel("X[m]")
#     ax.set_ylabel("Y[m]")
#     ax.set_title(f"Survey lines (highlighting line {ln})")
#     ax.legend()
#     plt.show()

# %% Infrastructure
# The geometry of the infrastructure should be followed

powerline= [
           [328670., 5614065., 0.],
           [330423., 5613759., 0.],
           [330640., 5612867., 0.],
           [330925., 5612637., 0.],
           [331808., 5610970., 0.],
           [332545., 5609960., 0.],
           [332764., 5609488., 0.]]

gaspipe1= [
    [327422, 5611363, 0.],
    [329174, 5612359, 0.],
    [330582, 5613036, 0.]]

gaspipe2=[
    [330610, 5613022, 0.],
    [330671, 5612376, 0.],
    [330709, 5612282, 0.],
    [331014, 5611968, 0.],
    [331263, 5611765, 0.],
    [331718, 5611209, 0.],
    [331793, 5611168, 0.],
    [332099, 5610901, 0.],
    [332280, 5610425, 0.]]

gaspipe21=[
    [330610, 5612814, 0.],
    [330671, 5612376, 0.],
    [330709, 5612282, 0.],
    [331014, 5611968, 0.],
    [331263, 5611765, 0.],
    [331474, 5611502, 0.]]
    
gaspipe22=[
    [331646, 5611299, 0.],
    [331793, 5611168, 0.],
    [332099, 5610901, 0.],
    [332280, 5610425, 0.]]

gaspipe3= [
   [331022, 5611976, 0.],
   [331198, 5612094, 0.],
   [331308, 5612034, 0.],
   [331561, 5612130, 0.],
   [332022, 5612158, 0.],
   [332037, 5612179, 0.],
   [332147, 5612850, 0.]]

gaspipe31= [
   [331022, 5611976, 0.],
   [331179, 5612082, 0.]]
   
gaspipe32=[
   [331263, 5612062, 0.],
   [331368, 5612061, 0.],
   [331561, 5612130, 0.],
   [332022, 5612158, 0.],
   [332037, 5612179, 0.],
   [332147, 5612850, 0.]]

gaspipe4=[
    [330621, 5613107, 0.],
    [330630, 5613238, 0.],
    [331146, 5613535, 0.],
    [331329, 5614255, 0.],
    [333592, 5614943, 0.]]

# Shifting and segmenting the gas pipeleines to follow the topography.
gasPipes = [gaspipe1, gaspipe2, gaspipe3, gaspipe4]
pipes = []
for pipe in gasPipes:
    pipe = np.array(pipe)
    pipe[:, 0] += xshift
    pipe[:, 1] += yshift
    pipe = mu.refine_path(pipe, length=35.)
    pipes.append(pipe)

# If topography is not applied, neither segmentation nor surface line are required.
gasPipes_surf = [gaspipe1, gaspipe21, gaspipe22, gaspipe31, gaspipe32, gaspipe4]

for p in gasPipes_surf:
    p = np.array(p)
    p[:, 0] += xshift
    p[:, 1] += yshift
    surface_pipe = mu.refine_path(p, length=10.)
    surface_pipes.append(surface_pipe)
    
# %% powerline surface lines
# If topography is not applied, surface line is not required.
pl= [[328666., 5614061., 0.],
     [330411., 5613747., 0.],
     [330633., 5612865., 0.],
     [330920., 5612632., 0.],
     [331803., 5610965., 0.],
     [332540., 5609955., 0.],
     [332759., 5609483., 0.]]

pl = np.array(pl)
pl[:, 0] += xshift
pl[:, 1] += yshift
pl[:, 0] -= 10. 
pl[:, 1] -= 10.
surface_power=mu.refine_path(pl, length=11.)

# powerline segmentation  for grounding points
powerline = np.array(powerline)
powerline[:, 0] += xshift
powerline[:, 1] += yshift
powerline= mu.refine_path(powerline, length=300.)

#%% grounding points
polys, extra = M.get_prism_poly_coords(powerline, r=7.)
groundings = [np.copy(polys[0]), np.copy(polys[-1])]
inter_groundings = []
for j in range(len(extra)):
    inter_groundings.append(np.copy(extra[j]))

# %% surface mesh & halfspace

insert_lines= [surface_power] +  surface_pipes 

# define surface mesh
M.build_surface(insert_line_tx=txs,
                insert_paths=groundings+inter_groundings,
                insert_lines= insert_lines
                )

M.build_halfspace_mesh()

#%% anomaly; Dipping plane
x_plate = -60.0
y_plate = 0.0
z_topo = dem(x_plate, y_plate)
z_fault = z_topo - 650

M.add_plate_tilted(dx=40., dy=7000., dz=1300., cell_size=1e4,
    origin=[-60.0, 0.0, z_fault], dip=-40., dip_azimuth=34., tilt_x=-1.)

#%% Infrastructure

##---- Adding Power line into the mesh when topography is included ----##

for j in range(len(polys)):
    TOPO = M.get_topo_vals(polys[j], z=20.)
    dz = TOPO[0] - polys[j][0, 2]   # shift based on first base node
    polys[j][:, 2] += dz

for j in range(len(extra)):
    TOPO = M.get_topo_vals(extra[j], z=20.)
    dz = TOPO[0] - extra[j][0, 2]
    extra[j][:, 2] += dz

M = add_infra_prisms_air(M, polys, extra, cz=50.)

groundings[0][:, 2] = M.get_topo_vals(groundings[0], z=0.)
groundings[1][:, 2] = M.get_topo_vals(groundings[1], z=0.)

for k in range(len(inter_groundings)):
    inter_groundings[k][:, 2] = M.get_topo_vals(inter_groundings[k], z=0.)

# close surface to air prism manually
n1surf = [M.Omega.createNodeWithCheck(p) for p in groundings[0]]
n2surf = [M.Omega.createNodeWithCheck(p) for p in groundings[-1]]
n1air = [M.Omega.createNodeWithCheck(p) for p in polys[0]]
n2air = [M.Omega.createNodeWithCheck(p) for p in polys[-1]]

# close sides of prisms
for ll in range(len(n1surf) - 1):
    M.Omega.createQuadrangleFace(n1air[ll], n1air[ll+1],
                                 n1surf[ll+1], n1surf[ll])
    M.Omega.createQuadrangleFace(n2air[ll], n2air[ll+1],
                                 n2surf[ll+1], n2surf[ll])

M.Omega.createQuadrangleFace(n1air[-1], n1air[0], n1surf[0], n1surf[-1])
M.Omega.createQuadrangleFace(n2air[-1], n2air[0], n2surf[0], n2surf[-1])

# close connection to surface for inter groundings
for k in range(len(inter_groundings)):
    n1surf = [M.Omega.createNodeWithCheck(p) for p in inter_groundings[k]]
    n1air = [M.Omega.createNodeWithCheck(p) for p in extra[k]]

    # close sides of prisms
    for ll in range(len(n1surf) - 1):
        M.Omega.createQuadrangleFace(n1air[ll], n1air[ll+1],
                                     n1surf[ll+1], n1surf[ll])

    M.Omega.createQuadrangleFace(n1air[-1], n1air[0], n1surf[0], n1surf[-1])
    
 ##---- Adding gas pipes into the mesh ----##
for pipe in pipes:
    pipe[:, 2] = M.get_topo_vals(pipe, z=-10)
    M.add_infra_prisms(pipe, r=5., cell_size=50.,  label='Pipe', marker=5)
      
#%% RX refinement

rx_air[:, 2] = M.get_topo_vals(rx_air, z=60.)
rx_common, rx_single, rx_tri = mu.refine_adaptive(rx_air, txs, r1=10., r2=5.,
                                                  r3=1, d3=200., d2=500,
                                                  #min_tx_dist=100.
                                                  )

M.add_rx(rx_common)

M.add_paths(rx_tri)

# %% Tetgen mesh

# add boundary mesh and mesh generation
M.extend_world(10., 10., 10.)

M.call_tetgen(tet_param='-pq1.6/12aA', suppress='') # Define the mesh quallity here

raise SystemExit()

# %% save survey parameters to import for inversion mesh generation

if os.path.isdir('data'):
    pass
else:
    os.makedirs('data')
np.savez('data/parameters_Fault_Infra.npz',
         xshift=xshift,
         yshift=yshift,
         rot=rot,
         txs=np.array(txs, dtype=object),
         rx_air=rx_air,
         infra=powerline,
         pipes=np.array(pipes, dtype=object),
         rx_line=rx_line
         )

# %% set synthetic data generation parameters

err = 0.05        #                           mu_0         nT
noise_B = 1e-3    # related to H (A/m) by 4 * pi * 1e-7 * 1e9 ~= 1256

p_fwd = 2         # polynomial order for forward modeling/ beeter to use P=2 for forward calculations
mod = 'synth_Fault_Infra'

n_cores=72        # total number of cores used
min_freqs = 3     # set to 2, 3 or 4 to trade time vs. RAM requirements, Or None if there is no memory issue
#                 # e.g., 3 means each thread calculates 3 freqs,
#                 # so time is tripled but RAM requirements are only 1/3

freqs = [23.15, 32.0, 45.25, 64.0, 90.51, 128.0, 181.02, 256.0,
         362.04, 512.0, 724.08, 1024.0]

# Define the CMP and number of Tx
cmps = [['Bx', 'By', 'Bz']]
tx_ids = [[0]]

# for anomaly markers 2, 3, 4, 5, 6, respectively
sig_m = [1/50, 2000., 200.]
# for marker 1, surrounding halfspace
sig_bg = 1e-3

# %% create synthetic data

pfname = 'Bxyz_Fault_Infra'
fop = MultiFWD(mod, mesh, list(freqs), cmps, tx_ids, sig_bg=sig_bg,
               p_fwd=p_fwd, min_freqs=min_freqs, n_cores=n_cores)
data = fop.response(sig_m)
np.save('data/' + pfname + '_no_noise.npy', data)

data = np.load('data/' + pfname + '_no_noise.npy')
# define abs_error for all recordoings (B fields)
abs_error = np.abs(data) * err + noise_B

# add noise
data += np.random.randn(len(data)) * abs_error

# remove very weak amplitudes
data[np.abs(data) < noise_B*0.5] = np.nan

# convert column vector to data structure required by inversion module
fop.export_npz('data/' + pfname, data, abs_error)

