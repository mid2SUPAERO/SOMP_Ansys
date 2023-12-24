import os, sys
path = os.path.abspath(os.path.dirname(__file__).join('.'))
if path not in sys.path:
    sys.path.append(path)

import numpy as np

import matplotlib.pyplot as plt
import niceplots
plt.style.use(niceplots.get_style())

from optim import TopOpt, Post3D

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

print('-'*(21*7+1))
print(('|{:^20}'*7+'|').format('Fiber','Matrix','Compliance (N.mm)','Mass (g)','CO2 (kgCO2)','Iter','Time (s)'))
print('-'*(21*7+1))
for name_f, fiber in zip(names_f, fibers):
    for name_m, matrix in zip(names_m, matrices):
        Ex, Ey, nuxy, nuyz, Gxy, rho, CO2mat = TopOpt.rule_mixtures(fiber=fiber, matrix=matrix, Vfiber=0.5)

        jobname = '_'.join([name_f, name_m])
        solver = TopOpt(inputfile='models/mbb3d.db', res_dir=f'results/global/{jobname}/', dim='3D_layer', jobname=jobname, echo=False)
        solver.set_material(Ex=Ex, Ey=Ey, nuxy=nuxy, nuyz=nuxy, Gxy=Gxy)
        solver.set_volfrac(0.3)
        solver.set_filters(r_rho=8, r_theta=20)
        solver.set_initial_conditions('random')
        solver.set_optim_options(max_iter=150, tol=1e-3, continuation=True)
        solver.run()
        solver.save()

        post = Post3D(solver)
        post.plot_convergence()
        post.plot_layer(layer=0)
        post.plot_layer(layer=1)
        post.plot(colorful=False)
        plt.close('all')

        comp = solver.comp_max_hist[-1]
        mass = 1e6 * solver.get_mass(rho)
        co2  = 1e3 * solver.get_CO2_footprint(rho, CO2mat, CO2veh)
        iter = solver.mma.iter
        t    = solver.time

        print(('|{:^20}|{:^20}|{:^20.3f}|{:^20.3f}|{:^20.3f}|{:^20}|{:^20.2f}|').format(name_f,name_m,comp,mass,co2,iter,t))

print('-'*(21*7+1))
