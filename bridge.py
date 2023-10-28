from mpi4py import MPI

from pathlib import Path
import numpy as np
import time

from python.optimization import TopOpt
from python.postprocessor import Post2D

ANSYS_path = Path('mapdl')
script_dir = Path('python/')
res_dir    = Path('results/bridge/')
mod_dir    = Path('models/')
TopOpt.set_paths(ANSYS_path, script_dir, res_dir, mod_dir)

Ex   = 113.6e3 # MPa
Ey   = 9.65e3 # MPa
Gxy  = 6e3 # MPa
nuxy = 0.334

t0 = MPI.Wtime()

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# size-1 should be odd to include 0 degrees
theta0 = np.linspace(-90, 90, num=size-1)
theta0 = np.append(theta0, None) # last one works with random initial condition

jobname = str(int(theta0[rank])) if rank != size-1 else 'rand'
solver = TopOpt(inputfiles='bridge', dim='2D', jobname=jobname,
	Ex=Ex, Ey=Ey, nuxy=nuxy, nuyz=nuxy, Gxy=Gxy, volfrac=0.45, r_rho=70, r_theta=160, theta0=theta0[rank], max_iter=100, echo=False)
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
post.animate()

comp  = comm.gather(solver.comp_hist[-1])
niter = comm.gather(solver.mma.iter)
dt    = comm.gather(solver.time)

if rank == 0:
    print('\n theta0      comp    iter    time')
    for i in range(size-1):
        print('{:7.1f}  {:7.2f} {:7d} {:7.2f}'.format(theta0[i],comp[i],niter[i],dt[i]))
    print('  rand   {:7.2f} {:7d} {:7.2f}'.format(comp[-1],niter[-1],dt[-1]))
    print('\nTotal elapsed time: {:.2f}s'.format(MPI.Wtime()-t0))
