from mpi4py import MPI

from pathlib import Path
import numpy as np

from python.optimization import TopOpt2D
from python.postprocessor import Post2D

ANSYS_path = Path('mapdl')
script_dir = Path('python/')
res_dir    = Path('results/bridge/')
mod_dir    = Path('models/')
TopOpt2D.load_paths(ANSYS_path, script_dir, res_dir, mod_dir)
TopOpt2D.set_processors(2)

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

solver = TopOpt2D(inputfile='bridge', Ex=Ex, Ey=Ey, nuxy=nuxy, nuyz=nuxy, Gxy=Gxy, volfrac=0.4, r_rho=60, r_theta=90, theta0=theta0[rank], jobname=str(int(theta0[rank])))
solver.set_solid_elem(np.where(solver.centers[:,1]>920)[0])
solver.set_optim_options(max_iter=100)
solver.optim()
print('\n{} - Elasped time: {:.2f}s'.format(rank, solver.time))
print('{} - FEA time: {:.2f}s\n'.format(rank, solver.mma.fea_time))

post = Post2D(solver)
post.plot_convergence()
post.animate()
post.plot()

comp  = comm.gather(solver.comp_hist[-1])
niter = comm.gather(solver.mma.iter)
dt    = comm.gather(solver.time)

if rank == 0:
    print('\n theta0    comp    iter    time')
    for i in range(size):
        print('{:7.1f} {:7.2f} {:7d} {:7.2f}'.format(theta0[i],comp[i],niter[i],dt[i]))

    import os, glob
    for filename in glob.glob('*.bat'): os.remove(filename)
    for filename in glob.glob('*.err'): os.remove(filename)
    for filename in glob.glob('*.log'): os.remove(filename)
    for filename in glob.glob('*.out'): os.remove(filename)
    
    print('Total elapsed time: {:.2f}s'.format(MPI.Wtime()-t0))

