import os, sys
path = os.path.abspath(os.path.dirname(__file__).join('.'))
if path not in sys.path:
    sys.path.append(path)

from mpi4py import MPI

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from optim import TopOpt, Post3D

ANSYS_path = Path('mapdl')
res_dir    = Path('results/global/')
mod_dir    = Path('models/')
TopOpt.set_paths(ANSYS_path, res_dir, mod_dir)

# {t/mm^3, MPa, -, kgCO2/kg}
bamboo    = {'rho': 700e-12,  'E': 17.5e3,  'v': 0.04,  'CO2': 1.0565}
flax      = {'rho': 1470e-12, 'E': 53.5e3,  'v': 0.355, 'CO2': 0.44}
hemp      = {'rho': 1490e-12, 'E': 62.5e3,  'v': 0.275, 'CO2': 1.6}
hmcarbon  = {'rho': 2105e-12, 'E': 760e3,   'v': 0.105, 'CO2': 68.1}
lmcarbon  = {'rho': 1820e-12, 'E': 242.5e3, 'v': 0.105, 'CO2': 20.3}
sglass    = {'rho': 2495e-12, 'E': 89.5e3,  'v': 0.22,  'CO2': 2.905}
eglass    = {'rho': 2575e-12, 'E': 78.5e3,  'v': 0.22,  'CO2': 2.45}

cellulose = {'rho': 990e-12,  'E': 3.25e3, 'v': 0.355, 'CO2': 3.8}
pla       = {'rho': 1290e-12, 'E': 5.19e3, 'v': 0.39,  'CO2': 2.115}
petg      = {'rho': 1270e-12, 'E': 2.06e3, 'v': 0.403, 'CO2': 4.375}
epoxy     = {'rho': 1255e-12, 'E': 2.41e3, 'v': 0.399, 'CO2': 5.94}
polyester = {'rho': 1385e-12, 'E': 4.55e3, 'v': 0.35,  'CO2': 4.5}

names_f   = ['bamboo', 'flax', 'hemp', 'hmcarbon', 'lmcarbon', 'sglass', 'eglass']
fibers    = [bamboo, flax, hemp, hmcarbon, lmcarbon, sglass, eglass]

names_m   = ['cellulose', 'pla', 'petg', 'epoxy', 'polyester']
matrices  = [cellulose, pla, petg, epoxy, polyester]

CO2veh = 1030 * 25 * 3.83 # kg_fuel/kg_transported/year * years * kgCO2/kg_fuel = kgCO2/kg

volfrac = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55]
Vfiber  = [0.25, 0.5]

t0 = MPI.Wtime()

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

for name_f, fiber in zip(names_f, fibers):
    for name_m, matrix in zip(anmes_m, matrices):
        f = volfrac[rank//2]

        Vf = Vfiber[rank%2]
        Ex, Ey, nuxy, nuyz, Gxy, rho, CO2mat = TopOpt.rule_mixtures(fiber=fiber, matrix=matrix, Vfiber=Vf)

        jobname = '_'.join([str(int(100*Vf)), fiber, matrix, str(int(100*f))])
        solver = TopOpt(inputfiles='mbb3d_fine', dim='3D_layer', jobname=jobname, echo=False)
        solver.set_material(Ex=Ex, Ey=Ey, nuxy=nuxy, nuyz=nuxy, Gxy=Gxy)
        solver.set_volfrac(f)
        solver.set_filters(r_rho=4, r_theta=10)
        solver.set_initial_conditions('random')
        solver.set_optim_options(max_iter=80)
        solver.create_optimizer()
        solver.run()

        post = Post3D(solver)
        post.plot_convergence()
        post.plot_layer(layer=0)
        post.plot_layer(layer=1)
        post.plot_layer(layer=2)
        post.plot_layer(layer=3)
        post.plot(colorful=False)

        plt.close('all')

        comp      = comm.gather(solver.comp_hist[-1])
        dt        = comm.gather(solver.time)
        footprint = comm.gather(1000 * solver.get_CO2_footprint(rho, CO2mat, CO2veh))

        if rank == 0:
            print()
            print('0.25 {} and {}'.format(name_f, name_m))
            print('volfrac    comp    time     CO2')
            for k in range(0,size,2):
                print('{:7.2f} {:7.2f} {:7.2f} {:7.2f}'.format(volfrac[k//2],comp[k],dt[k],footprint[k]))

            print()
            print('0.5 {} and {}'.format(name_f, name_m))
            print('volfrac    comp    time     CO2')
            for k in range(1,size,2):
                print('{:7.2f} {:7.2f} {:7.2f} {:7.2f}'.format(volfrac[k//2],comp[k],dt[k],footprint[k]))
            
        comm.Barrier()

if rank == 0:
    print('Total elapsed time: {:.2f}s'.format(MPI.Wtime()-t0))
