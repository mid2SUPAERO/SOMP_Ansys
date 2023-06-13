import numpy as np

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from functools import partial

from .optimization import TopOpt

class PostProcessor():
    def __init__(self, solver):
        self.solver = solver
        
    def make_grid(self, result):
        x, y = np.meshgrid(np.unique(self.solver.centers[:,0]),np.unique(self.solver.centers[:,1]))
        z = np.zeros_like(x)
        for e in range(self.solver.num_elem):
            i = np.where(x[0,:] == self.solver.centers[e,0])[0][0]
            j = np.where(y[:,0] == self.solver.centers[e,1])[0][0]
            z[j,i] = result[e]
            
        return x, y, z
        
    def plot(self, iteration=-1, filename=None, save=True, fig=None, ax=None):
        if fig == None: fig, ax = plt.subplots(dpi=300)
        ax.cla()
        ax.set_aspect('equal', 'box')
        plt.title('Compliance = {:.4f}'.format(self.solver.comp_hist[iteration]))
        
        x, y, density = self.make_grid(self.solver.rho_hist[iteration])
        ax.pcolormesh(x, y, density, cmap='binary')
        
        _, _, theta = self.make_grid(self.solver.theta_hist[iteration])
        ax.quiver(x, y, np.cos(theta), np.sin(theta), color='white', pivot='mid', headwidth=0, headlength=0, headaxislength=0)

        if save:
            if filename is None: filename = self.solver.res_dir / 'design.png'
            plt.savefig(filename)
        
    def plot_orientation(self, iteration=-1, fig=None, ax=None):
        if fig == None: fig, ax = plt.subplots(dpi=300)
        ax.cla()
        ax.set_aspect('equal', 'box')
        plt.title('Compliance = {:.4f}'.format(self.solver.comp_hist[iteration])) 
        
        x, y, theta = self.make_grid(self.solver.theta_hist[iteration])
        ax.quiver(x, y, np.cos(theta), np.sin(theta), pivot='mid', headwidth=0, headlength=0, headaxislength=0)
       
    def plot_convergence(self, compliance_unit='N.mm'):
        plt.figure()
        plt.plot(self.solver.comp_hist)
        plt.xlabel('Iteration')
        plt.ylabel(f'Compliance [{compliance_unit}]')
        
    def animate(self, filename=None):
        if filename is None: filename = self.solver.res_dir / 'animation.gif'
        fig, ax = plt.subplots(dpi=300)
        anim = FuncAnimation(fig, partial(self.plot, save=False, fig=fig, ax=ax), frames=len(self.solver.rho_hist))
        anim.save(filename)
        
    def animate_orientation(self, filename=None):
        if filename is None: filename = self.solver.res_dir / 'animation_o.gif'
        fig, ax = plt.subplots(dpi=300)
        anim = FuncAnimation(fig, partial(self.plot_orientation, fig=fig, ax=ax), frames=len(self.solver.rho_hist))
        anim.save(filename)
        
    def CO2_footprint(self, rho, CO2mat, CO2veh):
        """
        rho: density
        CO2mat: mass CO2 emmited per mass material (material production)
        CO2veh: mass CO2 emitted per mass material during life (use in a vehicle)
                = mass fuel per mass transported per lifetime * service life * mass CO2 emmited per mass fuel
        """
        x = self.solver.rho_hist[-1]
        mass = rho * x.dot(self.solver.elemvol)
        return mass * (CO2mat + CO2veh)
