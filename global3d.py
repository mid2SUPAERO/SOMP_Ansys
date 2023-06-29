from mpi4py import MPI

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from python.optimization import TopOpt
from python.postprocessor import Post3D

ANSYS_path = Path('mapdl')
script_dir = Path('python/')
res_dir    = Path('results/3d/')
mod_dir    = Path('models/')
TopOpt.load_paths(ANSYS_path, script_dir, res_dir, mod_dir)

# fiber: bamboo
rhofiber  = 700e-12 # t/mm^3
Efiber    = 17.5e3 # MPa
vfiber    = 0.04
CO2fiber  = 1.0565 # kgCO2/kg

# matrix: cellulose
rhomatrix = 990e-12 # t/mm^3
Ematrix   = 3.25e3
vmatrix   = 0.355 # MPa
CO2matrix = 3.8 # kgCO2/kg

Vfiber  = 0.5
Vmatrix = 1-Vfiber

Gfiber  = Efiber/(2*(1+vfiber))
Gmatrix = Ematrix/(2*(1+vmatrix))

Ex   = Efiber*Vfiber + Ematrix*Vmatrix
Ey   = Efiber*Ematrix / (Efiber*Vmatrix + Ematrix*Vfiber)
Gxy  = Gfiber*Gmatrix / (Gfiber*Vmatrix + Gmatrix*Vfiber)
nuxy = vfiber*Vfiber + vmatrix*Vmatrix
rho  = rhofiber*Vfiber + rhomatrix*Vmatrix

CO2mat = (rhofiber*Vfiber*CO2fiber + rhomatrix*Vmatrix*CO2matrix)/rho # kgCO2/kg
CO2veh = 1030 * 25 * 3.83 # kg_fuel/kg_transported/year * years * kgCO2/kg_fuel = kgCO2/kg

t0 = MPI.Wtime()

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# size should be odd to include 0 degrees
theta0 = np.linspace(-90, 90, num=size)

f = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
comp, niter, dt, footprint = [], [], [], []
for volfrac in f:
    jobname = str(int(10*volfrac)) + '_' + str(int(theta0[rank]))
    solver = TopOpt(inputfile='mbb3d', dim='3D', jobname=jobname,
                Ex=Ex, Ey=Ey, nuxy=nuxy, nuyz=vmatrix, Gxy=Gxy, volfrac=volfrac, r_rho=6, r_theta=16, theta0=theta0[rank],
                max_iter=100, move_rho=0.4, move_theta=10)

    solver.optim()
    print('{} - volfrac = {} - Elasped time: {:.2f}s'.format(rank, volfrac, solver.time))

    post = Post3D(solver)
    post.plot_convergence()
    post.plot_layer(layer=0)
    post.plot_layer(layer=1)

    plt.close('all')

    comp.append(solver.comp_hist[-1])
    niter.append(solver.mma.iter)
    dt.append(solver.time)
    footprint.append(1000 * post.CO2_footprint(rho, CO2mat, CO2veh))

comps      = comm.gather(comp)
niters     = comm.gather(niter)
dts        = comm.gather(dt)
footprints = comm.gather(footprint)

if rank == 0:
    mincomp = []
    for j in range(len(f)):
        print('\n**** volfrac = {:.1f} ****'.format(f[j]))
        print(' theta0    comp    iter    time     CO2')
        for i in range(size):
            print('{:7.1f} {:7.2f} {:7d} {:7.2f} {:7.2f}'.format(theta0[i],comps[i][j],niters[i][j],dts[i][j],footprints[i][j]))
        
        mincomp.append(np.amin(np.array(comps)[:,j]))

    f, mincomp = np.array(f), np.array(mincomp)
    plt.figure()
    plt.plot(f,f*mincomp)
    plt.ylabel('volfrac * compliance')
    plt.xlabel('volfrac')
    plt.savefig(res_dir/'optim.png')

    print('Total elapsed time: {:.2f}s'.format(MPI.Wtime()-t0))