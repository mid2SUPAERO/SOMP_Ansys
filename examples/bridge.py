import os, sys
path = os.path.abspath(os.path.dirname(__file__).join('.'))
if path not in sys.path:
    sys.path.append(path)

from mpi4py import MPI
import numpy as np

from optim import TopOpt, Post3D

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
solver = TopOpt(inputfile='models/bridge.db', res_dir=f'results/bridge/{jobname}/', dim='2D', jobname=jobname, echo=False)
solver.set_material(Ex=Ex, Ey=Ey, nuxy=nuxy, nuyz=nuxy, Gxy=Gxy)
solver.set_volfrac(0.45)
solver.set_filters(r_rho=70, r_theta=160)
solver.set_solid_elem(np.where(solver.centers[:,1]>920)[0])

if rank != size-1:
    solver.set_initial_conditions('fix', theta0=theta0[rank])
else:
    solver.set_initial_conditions('random')

solver.set_optim_options(max_iter=100)
solver.run()

print()
print('rank = {}'.format(rank))
solver.print_timing()

post = Post2D(solver)
post.plot_convergence()
post.plot()
post.animate()

comp  = comm.gather(solver.comp_max_hist[-1])
niter = comm.gather(solver.mma.iter)
dt    = comm.gather(solver.time)

if rank == 0:
    print('\n theta0      comp    iter    time')
    for i in range(size-1):
        print('{:7.1f}  {:7.2f} {:7d} {:7.2f}'.format(theta0[i],comp[i],niter[i],dt[i]))
    print('  rand   {:7.2f} {:7d} {:7.2f}'.format(comp[-1],niter[-1],dt[-1]))
    print('\nTotal elapsed time: {:.2f}s'.format(MPI.Wtime()-t0))
