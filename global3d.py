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
TopOpt.set_paths(ANSYS_path, script_dir, res_dir, mod_dir)

fibers   = ['bamboo', 'flax', 'hemp', 'hmcarbon', 'lmcarbon', 'sglass', 'eglass']
rhofiber = [700e-12, 1470e-12, 1490e-12, 2105e-12, 1820e-12, 2495e-12, 2575e-12] # t/mm^3
Efiber   = [17.5e3, 53.5e3, 62.5e3, 62.5e3, 760e3, 242.5e3, 89.5e3, 78.5e3] # MPa
nufiber  = [0.04, 0.355, 0.275, 0.105, 0.105, 0.22, 0.22]
CO2fiber = [1.0565, 0.44, 1.6, 68.1, 20.3, 2.905, 2.45] # kgCO2/kg

matrices  = ['cellulose', 'pla', 'petg', 'epoxy', 'polyester']
rhomatrix = [990e-12, 1290e-12, 1270e-12, 1255e-12, 1385e-12] # t/mm^3
Ematrix   = [3.25e3, 5.19e3, 2.06e3, 2.41e3, 4.55e3] # MPa
numatrix  = [0.355, 0.39, 0.403, 0.399, 0.35]
CO2matrix = [3.8, 2.115, 4.375, 5.94, 4.5] # kgCO2/kg

CO2veh = 1030 * 25 * 3.83 # kg_fuel/kg_transported/year * years * kgCO2/kg_fuel = kgCO2/kg

volfrac = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55]
Vfiber  = [0.25, 0.5]

t0 = MPI.Wtime()

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

for i, fiber in enumerate(fibers):
    for j, matrix in enumerate(matrices):
        rhof, Ef, nuf, CO2f = rhofiber[i], Efiber[i], nufiber[i], CO2fiber[i]
        rhom, Em, num, CO2m = rhomatrix[j], Ematrix[j], numatrix[j], CO2matrix[j]
        Gf, Gm = Ef/(2*(1+nuf)), Em/(2*(1+num))

        f = volfrac[rank//2]
        Vf, Vm = Vfiber[rank%2], 1-Vfiber[rank%2]

        Ex   = Ef*Vf + Em*Vm
        Ey   = Ef*Em / (Ef*Vm + Em*Vf)
        Gxy  = Gf*Gm / (Gf*Vm + Gm*Vf)
        nuxy = nuf*Vf + num*Vm
        rho  = rhof*Vf + rhom*Vm

        CO2mat = (rhof*Vf*CO2f + rhom*Vm*CO2m)/rho # kgCO2/kg

        jobname = '_'.join([str(int(100*Vf)), fiber, matrix, str(int(100*f))])
        solver = TopOpt(inputfile='mbb3d_fine', dim='3D', jobname=jobname,
            Ex=Ex, Ey=Ey, nuxy=nuxy, nuyz=num, Gxy=Gxy, volfrac=f, r_rho=4, r_theta=10, max_iter=80, echo=False)
        solver.optim()

        post = Post3D(solver)
        post.plot_convergence()
        post.plot_layer(layer=0)
        post.plot_layer(layer=1)
        post.plot_layer(layer=2)
        post.plot_layer(layer=3)
        post.plot()

        plt.close('all')

        comp      = comm.gather(solver.comp_hist[-1])
        dt        = comm.gather(solver.time)
        footprint = comm.gather(1000 * solver.CO2_footprint(rho, CO2mat, CO2veh))

        if rank == 0:
            print()
            print('0.25 {} and {}'.format(fiber, matrix))
            print('volfrac    comp    time     CO2')
            for k in range(0,size,2):
                print('{:7.2f} {:7.2f} {:7.2f} {:7.2f}'.format(volfrac[k//2],comp[k],dt[k],footprint[k]))
        
            plt.figure()
            plt.plot(volfrac,volfrac*np.array(comp[::2]))
            plt.ylabel('volfrac * compliance')
            plt.xlabel('volfrac')
            plt.savefig(res_dir/(''.join(['_'.join([str(25), fiber, matrix]), '.png'])))

            print()
            print('0.5 {} and {}'.format(fiber, matrix))
            print('volfrac    comp    time     CO2')
            for k in range(1,size,2):
                print('{:7.2f} {:7.2f} {:7.2f} {:7.2f}'.format(volfrac[k//2],comp[k],dt[k],footprint[k]))
        
            plt.figure()
            plt.plot(volfrac,volfrac*np.array(comp[1::2]))
            plt.ylabel('volfrac * compliance')
            plt.xlabel('volfrac')
            plt.savefig(res_dir/(''.join(['_'.join([str(50), fiber, matrix]), '.png'])))
            
        comm.Barrier()

if rank == 0:
    print('Total elapsed time: {:.2f}s'.format(MPI.Wtime()-t0))
