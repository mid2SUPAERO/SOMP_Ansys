from mpi4py import MPI

from pathlib import Path
import numpy as np
import time

from python.optimization import TopOpt
from python.postprocessor import Post2D

ANSYS_path = Path('C:/Program Files/ANSYS Inc/v202/ansys/bin/winx64/MAPDL.exe')
script_dir = Path('python/')
res_dir    = Path('results/bridge/')
mod_dir    = Path('models/')
TopOpt.load_paths(ANSYS_path, script_dir, res_dir, mod_dir)

Ex   = 113.6e3 # MPa
Ey   = 9.65e3 # MPa
Gxy  = 6e3 # MPa
nuxy = 0.334

t0 = MPI.Wtime()

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# size should be odd to include 0 degrees
theta0 = np.linspace(-90, 90, num=size)

solver = TopOpt(inputfile='bridge', dim='2D', jobname=str(int(theta0[rank])),
                Ex=Ex, Ey=Ey, nuxy=nuxy, nuyz=vmatrix, Gxy=Gxy, volfrac=0.4, r_rho=60, r_theta=90, theta0=theta0[rank],
                max_iter=100, move_rho=0.4, move_theta=20)
solver.set_solid_elem(np.where(solver.centers[:,1]>920)[0])
solver.optim()

print()
print('{} - Total elapsed time     {:7.2f}s'.format(rank, solver.time))
print('{} - FEA time               {:7.2f}s'.format(rank, solver.fea_time))
print('{} - Derivation time        {:7.2f}s'.format(rank, solver.deriv_time))
print('{} - Variable updating time {:7.2f}s'.format(rank, solver.mma.update_time))

post = Post2D(solver)
post.plot_convergence()
post.plot()

comp  = comm.gather(solver.comp_hist[-1])
niter = comm.gather(solver.mma.iter)
dt    = comm.gather(solver.time)

if rank == 0:
    print('\n theta0      comp    iter    time')
    for i in range(size):
        print('{:7.1f}  {:7.2f} {:7d} {:7.2f}'.format(theta0[i],comp[i],niter[i],dt[i]))
    
    print('\nTotal elapsed time: {:.2f}s'.format(time.time()-t0))

