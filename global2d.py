from mpi4py import MPI
from multiprocessing import Pool

from pathlib import Path
import numpy as np

from python.optimization import TopOpt2D
from python.postprocessor import PostProcessor

ANSYS_path = Path('mapdl')
script_dir = Path('python/')
res_dir    = Path('results/multi/')
mod_dir    = Path('models/')
TopOpt2D.load_paths(ANSYS_path, script_dir, res_dir, mod_dir)
TopOpt2D.set_processors(3)

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

Ex  = Efiber*Vfiber + Ematrix*Vmatrix
Ey  = Efiber*Ematrix / (Efiber*Vmatrix + Ematrix*Vfiber)
Gxy = Gfiber*Gmatrix / (Gfiber*Vmatrix + Gmatrix*Vfiber)
nu  = vfiber*Vfiber + vmatrix*Vmatrix
rho = rhofiber*Vfiber + rhomatrix*Vmatrix

CO2mat = (rhofiber*Vfiber*CO2fiber + rhomatrix*Vmatrix*CO2matrix)/rho # kgCO2/kg
CO2veh = 1030 * 25 * 3.83 # kg_fuel/kg_transported/year * years * kgCO2/kg_fuel = kgCO2/kg

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

instances_per_node = 5

Ntheta = instances_per_node * size
theta0 = np.linspace(-90, 90, num=Ntheta)

def launch(theta):
    solver = TopOpt2D(inputfile='mbb30_15', Ex=Ex, Ey=Ey, Gxy=Gxy, nu=nu, volfrac=0.3, rmin=1.5, theta0=theta, jobname=str(int(theta)))
    solver.optim()

    post = PostProcessor(solver)
    post.animate()
    post.plot()

    return solver

with Pool(instances_per_node) as p:
    solvers = p.map(launch,theta0[rank*instances_per_node:(rank+1)*instances_per_node])

comm.Barrier()
solvers = comm.Gather(solvers, root=0)
# TODO footprint

if rank == 0:
    solvers = [item for sublist in solvers for ite, in sublist]
    print(' theta0 comp iter time')
    for i in range(size):
        print('{:.1f} {:.4f} {:.0d} {:.2f}'.format(theta0[i],solvers[i].comp_hist[-1],solvers[i].mma.iter,solvers[i].time))

    import os, glob
    for filename in glob.glob('cleanup*'): os.remove(filename)
    for filename in glob.glob('*.bat'): os.remove(filename)
    for filename in glob.glob('*.err'): os.remove(filename)
    for filename in glob.glob('*.log'): os.remove(filename)
    for filename in glob.glob('*.out'): os.remove(filename)

