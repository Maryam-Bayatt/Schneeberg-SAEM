# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 08:19:43 2026

@author: Maryam Bayat
"""

import numpy as np
import pygimli as pg
from custEM.meshgen import meshgen_utils as mu
from custEM.meshgen.meshgen_tools import BlankWorld
from custEM.inv.inv_utils import MultiFWD
from saem import tools
from saem import CSEMData
import time
start_time = time.time() 

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # #  run synthetic data inversion # # # # # # # # # # # #
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

    #poly1, poly2 = [], []
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
    
    M.add_marker('infra', [np.mean([polys[0][2][0], polys[1][2][0]]),
                           np.mean([polys[0][2][1], polys[1][2][1]]),
                           np.mean([polys[0][2][2], polys[1][2][2]]) - 1.],
                 cell_size=cz)
      
    return(M)

# %% specify polnomial order
#suited refinement parameters are chosen below accordingly
p = 1
invmod = 'p' + str(p)
invmesh = "inv_Fault_Infra_invDom"

# %% create mesh

param = np.load('data/parameters_Fault_Infra.npz', allow_pickle=True)

# define inversion area roughly covering the Rx grid
invpoly = np.array([[-4300., -2800., 0.],
                    [ 3900., -2800., 0.],
                    [ 3900.,  3400., 0.],
                    [-4300.,  3400., 0.]])

# Convert degrees to radians
theta = np.deg2rad(30)

# Rotation matrix around Z-axis
R = np.array([
    [np.cos(theta), -np.sin(theta), 0],
    [np.sin(theta),  np.cos(theta), 0],
    [0,              0,             1]])

invpoly = invpoly @ R.T

# create world (PLC)
M = BlankWorld(name=invmesh,
               x_dim=[-1e4, 1e4],
               y_dim=[-1e4, 1e4],
               z_dim=[-1e4, 1e4],
               topo="schneeberg.asc",
               easting_shift=float(param['xshift']),
               northing_shift=float(param['yshift']),
               inner_area_cell_size=4e3,
               preserve_edges=True)

#%% Tx

tx1 = np.asarray(param['txs'][0], dtype=float)

# %% infrastructure
# To incorporate the infrastructure in the inversion domain, the infra geometry is needed
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

xshift = param['xshift']
yshift = param['yshift']

gasPipes = [gaspipe1, gaspipe2, gaspipe3, gaspipe4]
pipes = []
for pipe in gasPipes:
    pipe = np.array(pipe)
    pipe[:, 0] += xshift
    pipe[:, 1] += yshift
    pipe = mu.refine_path(pipe, length=35)
    pipes.append(pipe)

gasPipes_surf = [gaspipe1, gaspipe21, gaspipe22, gaspipe31, gaspipe32, gaspipe4]
surface_pipes = []
for pi in gasPipes_surf:
    pi = np.array(pi)
    pi[:, 0] += xshift
    pi[:, 1] += yshift
    surface_pipe = mu.refine_path(pi, length=10.)
    surface_pipes.append(surface_pipe)
    
#powerline surface lines

pl= [
          [328665., 5614061., 0.], 
          [330411., 5613747., 0.],
          [330634., 5612865., 0.],
          [330920., 5612632., 0.],
          [331803., 5610965., 0.],
          [332540., 5609955., 0.],
          [332759., 5609483., 0.]
          ]

pl = np.array(pl)
pl[:, 0] += xshift
pl[:, 1] += yshift
pl[:, 0] -= 10.
pl[:, 1] -= 10.
surface_power=mu.refine_path(pl, length=14.)

# powerline segmentation  for grounding
powerline = np.array(powerline)
powerline[:, 0] += xshift
powerline[:, 1] += yshift
powerline= mu.refine_path(powerline, length=300.)

# groundings

polys, extra = M.get_prism_poly_coords(powerline, r=7.)
groundings = [np.copy(polys[0]), np.copy(polys[-1])]
inter_groundings = []
for j in range(len(extra)):
    inter_groundings.append(np.copy(extra[j]))

# %% surface mesh

insert_lines= [surface_power] + surface_pipes

# define surface mesh
M.build_surface(insert_line_tx=[tx1],
                insert_paths=groundings+inter_groundings,
                insert_lines= insert_lines
                )

#%% build mesh

# when choosing the parameters, the required memory should be taken into account
M.add_inv_domains(-800., invpoly, x_frame=3e3, y_frame=3e3, z_frame=2e3, cell_size=1e6) #cell_size=4e5

M.build_halfspace_mesh()

rx_air = param['rx_air']

#rx_air[:,2] = M.get_topo_vals(rx_air, z=60)  #included in the saved parameters

rx_common, rx_single, rx_tri = mu.refine_adaptive(rx_air, [tx1], r1=10., r2=5.,
                                                  r3=1, d3=200., d2=500,
                                                  #min_tx_dist=100.
                                                  )
M.add_rx(rx_common)

M.add_paths(rx_tri)

# M.extend_world(10., 10., 10.) # for min frequncy of 23Hz, probably not necessary

# %% add infra in the inv domain

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

##---- Gas pipes ----##

for pipe in pipes:
    pipe[:, 2] = M.get_topo_vals(pipe, z=-10)
    M.add_infra_prisms(pipe, r=5., cell_size=50., label='Pipe')#, marker=5)
 
# %% Defining the domain markers  
# flip domain markers for constant domain:
# To fixed the infra in the mesh for forward calculations

M.Omega.regionMarker(7).setMarker(2)
M.Omega.regionMarker(2).setMarker(7)

M.Omega.regionMarker(8).setMarker(3)
M.Omega.regionMarker(3).setMarker(8)

M.Omega.regionMarker(2).setMarker(5)
M.Omega.regionMarker(5).setMarker(2)

M.Omega.regionMarker(3).setMarker(6)
M.Omega.regionMarker(6).setMarker(3)

M.overwrite_markers=[7, 8]

# %% tetgem mesh

if p == 1:
    M.call_tetgen(tet_param='-pq1.2aA')
else:
    M.call_tetgen(tet_param='-pq1.6aA')

print("PLC region markers:", [rm.marker() for rm in M.Omega.regionMarkers()])
for rm in M.Omega.regionMarkers():
    print("marker:", rm.marker(), "pos:", [rm.x(), rm.y(), rm.z()])

# raise SystemExit()

# %% define inversion parameter

sig_bg = 1e-3
# Determine a suitable balance between the cross-section of prisms and electrical conductivity
# The final parameterization represents an equivalent conductive structure whose integrated \\
# conductance reproduces the expected EM coupling between the infra. and the surrounding subsurface.
sig_power=2000.0
sig_pipe = 200.0 
sig_fix=[sig_bg, sig_power, sig_pipe, sig_pipe, sig_pipe, sig_pipe]
skip_domains=[0, 1, 2, 3, 4, 5, 6]

max_iter = 21
synth_data = np.load('data/Bxyz_Fault_InfraToImport.npz', allow_pickle=True)

# %% set up forward operator


fop = MultiFWD(invmod, invmesh, saem_data=synth_data, skip_domains=skip_domains,
               sig_bg=sig_bg, sig_fix=sig_fix, p_fwd=21, n_cores=64, #max_procs=6,
               start_iter=0, min_freqs=3)

fop.setRegionProperties("*", limits=[1e-4, 1e0])

# %% set up inversion operator
inv = pg.Inversion(verbose=True)  # , debug=True)
inv.setForwardOperator(fop)
inv.setPostStep(fop.analyze)
# dT = pg.trans.TransSymLog(1e-3)
# inv.dataTrans = dT

# %% run inversion
invmodel = inv.run(fop.measured, fop.errors, lam=5., verbose=True,
                   startModel=fop.sig_0, maxIter=max_iter)

# save final conductivity vector
np.save(fop.inv_dir + 'final_inv_Fault_Infra_invDom.npy', invmodel)

# %% post-processing

pgmesh = fop.mesh()
pgmesh['sigma'] = invmodel
pgmesh['res'] = 1./invmodel

try:
    from saem.tools import coverage
    pgmesh['coverage'] = coverage(inv)
except ImportError:
    print("SAEM tools not found, coverage could not be computed")
    pass

pgmesh.exportVTK(fop.inv_dir + invmod + '_final_invmodel_Fault_Infra_invDom.vtk')

end_time = time.time()  # Record the end time
execution_time = end_time - start_time
print(f"Execution time: {execution_time} seconds")