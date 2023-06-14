from mpi4py import MPI

from pathlib import Path
import numpy as np

from python.optimization import TopOpt2D
from python.postprocessor import PostProcessor

ANSYS_path = Path('mapdl')
script_dir = Path('python/')
res_dir    = Path('results/multi/')
mod_dir    = Path('models/')
TopOpt2D.load_paths(ANSYS_path, script_dir, res_dir, mod_dir)
TopOpt2D.set_processors(2)

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

theta0 = np.linspace(-90, 90, num=size)

solver = TopOpt2D(inputfile='mbb30_15', Ex=Ex, Ey=Ey, Gxy=Gxy, nu=nu, volfrac=0.3, rmin=1.5, theta0=theta0[rank], jobname=str(int(theta0[rank])))
solver.optim()

post = PostProcessor(solver)
post.animate()
post.plot()

foorprint = 1000 * solver.CO2_footprint(rho, CO2mat, CO2veh)

solvers    = comm.Gather(solver)
foorprints = comm.Gather(footprint)

if rank == 0:
    print(' theta0 comp iter time CO2')
    for i in range(size):
        print('{:.1f} {:.4f} {:.0d} {:.2f} {:.2f}'.format(theta0[i],solvers[i].comp_hist[-1],solvers[i].mma.iter,solvers[i].time,footprints[i]))

    import os, glob
    for filename in glob.glob('cleanup*'): os.remove(filename)
    for filename in glob.glob('*.bat'): os.remove(filename)
    for filename in glob.glob('*.err'): os.remove(filename)
    for filename in glob.glob('*.log'): os.remove(filename)
    for filename in glob.glob('*.out'): os.remove(filename)

