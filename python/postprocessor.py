import numpy as np

from abc import ABC, abstractmethod

from matplotlib import cm, colorbar
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from functools import partial

class PostProcessor(ABC):
    def __init__(self, solver):
        self.solver = solver
        
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
    
    def plot_convergence(self, compliance_unit='N.mm'):
        plt.figure()
        plt.plot(self.solver.comp_hist)
        plt.xlabel('Iteration')
        plt.ylabel(f'Compliance [{compliance_unit}]')
    
    @abstractmethod
    def plot(self, iteration=-1, filename=None, save=True, fig=None, ax=None):
        pass
    
    @abstractmethod
    def plot_orientation(self, filename=None, save=True, iteration=-1, fig=None, ax=None):
        pass
    
    @abstractmethod
    def animate(self, filename=None):
        pass
    
    @abstractmethod
    def animate_orientation(self, filename=None):
        pass

class Post2D(PostProcessor):
    def plot(self, iteration=-1, filename=None, save=True, fig=None, ax=None):
        if fig == None: fig, ax = plt.subplots(dpi=300)
        ax.cla()
        ax.set_aspect('equal', 'box')
        plt.title('Compliance = {:.4f}'.format(self.solver.comp_hist[iteration]))
        
        x, y, density = self.make_grid(self.solver.rho_hist[iteration], nodes=True)
        ax.pcolormesh(x, y, density, cmap='binary')
        
        x, y, theta = self.make_grid(self.solver.theta_hist[iteration], nodes=False)
        ax.quiver(x, y, np.cos(theta), np.sin(theta), color='white', pivot='mid', headwidth=0, headlength=0, headaxislength=0)

        if save:
            if filename is None: filename = self.solver.res_dir / 'design.png'
            plt.savefig(filename)
        
    def plot_orientation(self, iteration=-1, filename=None, save=True, fig=None, ax=None):
        if fig == None: fig, ax = plt.subplots(dpi=300)
        ax.cla()
        ax.set_aspect('equal', 'box')
        plt.title('Compliance = {:.4f}'.format(self.solver.comp_hist[iteration])) 
        
        x, y, theta = self.make_grid(self.solver.theta_hist[iteration])
        ax.quiver(x, y, np.cos(theta), np.sin(theta), pivot='mid', headwidth=0, headlength=0, headaxislength=0)
        
        if save:
            if filename is None: filename = self.solver.res_dir / 'orientation.png'
            plt.savefig(filename)
        
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
        
    ## --------------
    def make_grid(self, result, nodes):
        x, y = np.meshgrid(np.unique(self.solver.centers[:,0]),np.unique(self.solver.centers[:,1]))
        z = np.zeros_like(x)
        for e in range(self.solver.num_elem):
            i = np.where(x[0,:] == self.solver.centers[e,0])[0][0]
            j = np.where(y[:,0] == self.solver.centers[e,1])[0][0]
            z[j,i] = result[e]
        
        if nodes:
            x, y = np.meshgrid(np.unique(self.solver.node_coord[:,0]),np.unique(self.solver.node_coord[:,1]))
            
        return x, y, z

class Post3D(PostProcessor):
    def plot(self, iteration=-1, filename=None, save=True, fig=None, ax=None):
        data = self.solver.rho_hist[iteration]
        threshold = 0.5

        if ax is None:
            fig = plt.figure(dpi=300)
            ax = fig.add_axes([0.1, 0.1, 0.8, 0.8], projection='3d')
            ax_cb = fig.add_axes([0.9, 0.3, 0.02, 0.45])
            colorbar.ColorbarBase(ax_cb, cmap='binary', orientation='vertical')
        ax.set_box_aspect((np.amax(self.solver.node_coord[:,0]),np.amax(self.solver.node_coord[:,1]),np.amax(self.solver.node_coord[:,2])))
        plt.title('Compliance = {:.4f}'.format(self.solver.comp_hist[iteration]))
    
        cmap = cm.get_cmap('binary')
        for elem in range(len(data)):
            if data[elem] > threshold:
                self.plotCube(ax, elem, cmap(data[elem]), alpha=0.5)
                
        # x, y, z, theta = self.make_grid(self.solver.theta_hist[iteration])
        # ax.quiver(x, y, z, np.cos(theta), np.sin(theta), np.zeros_like(theta), color='red', pivot='mid', headwidth=0, headlength=0, headaxislength=0)
        
        if save:
            if filename is None: filename = self.solver.res_dir / 'design.png'
            plt.savefig(filename)
            
    def plot_orientation(self, filename=None, save=True, iteration=-1, fig=None, ax=None):
        pass
        # if ax is None:
        #     fig = plt.figure(dpi=300)
        #     ax = fig.add_axes([0.1, 0.1, 0.8, 0.8], projection='3d')
        #     ax.set_box_aspect((np.amax(self.solver.node_coord[:,0]),np.amax(self.solver.node_coord[:,1]),np.amax(self.solver.node_coord[:,2])))
        # ax.cla()
        # plt.title('Compliance = {:.4f}'.format(self.solver.comp_hist[iteration])) 
        
        # x, y, z, theta = self.make_grid(self.solver.theta_hist[iteration])
        # ax.quiver(x, y, z, np.cos(theta), np.sin(theta), np.zeros_like(theta), pivot='mid', headwidth=0, headlength=0, headaxislength=0)
        
        # if save:
        #     if filename is None: filename = self.solver.res_dir / 'orientation.png'
        #     plt.savefig(filename)
    
    def animate(self, filename=None):
        pass
    
    def animate_orientation(self, filename=None):
        pass
    
    ## --------------
    def plotCube(self, ax, elem, c, alpha=0.5):
        nodes = self.solver.elmnodes[elem,:]
        coord = self.solver.node_coord[nodes,:]
        
        # bottom z-
        x, y = np.meshgrid(coord[[0,2],0], coord[[0,2],1])
        z = np.array([[coord[0,2],coord[1,2]],[coord[3,2],coord[2,2]]])
        ax.plot_surface(x, y, z, color=c, rstride=1, cstride=1, alpha=alpha)
        # upper z+
        x, y = np.meshgrid(coord[[4,6],0], coord[[4,6],1])
        z = np.array([[coord[4,2],coord[5,2]],[coord[7,2],coord[6,2]]])
        ax.plot_surface(x, y, z, color=c, rstride=1, cstride=1, alpha=alpha)
        # left x-
        x, y = np.meshgrid(coord[[2,7],0], coord[[2,7],1])
        z = np.array([[coord[2,2],coord[3,2]],[coord[6,2],coord[7,2]]])
        ax.plot_surface(x, y, z, color=c, rstride=1, cstride=1, alpha=alpha)
        # right x+
        x, y = np.meshgrid(coord[[0,5],0], coord[[0,5],1])
        z = np.array([[coord[0,2],coord[1,2]],[coord[4,2],coord[5,2]]])
        ax.plot_surface(x, y, z, color=c, rstride=1, cstride=1, alpha=alpha)
        # front y+
        x, y = np.meshgrid(coord[[1,6],0], coord[[1,6],1])
        z = np.array([[coord[1,2],coord[5,2]],[coord[2,2],coord[6,2]]])
        ax.plot_surface(x, y, z, color=c, rstride=1, cstride=1, alpha=alpha)
        # back y-
        x, y = np.meshgrid(coord[[0,7],0], coord[[0,7],1])
        z = np.array([[coord[0,2],coord[4,2]],[coord[3,2],coord[7,2]]])
        ax.plot_surface(x, y, z, color=c, rstride=1, cstride=1, alpha=alpha)
        
    def make_grid(self, result):
        x, y, z = np.meshgrid(np.unique(self.solver.centers[:,0]),np.unique(self.solver.centers[:,1]),np.unique(self.solver.centers[:,2]))
        res = np.zeros_like(x)
        for e in range(self.solver.num_elem):
            i = np.where(x[0,:,:] == self.solver.centers[e,0])[0][0]
            j = np.where(y[:,0,:] == self.solver.centers[e,1])[0][0]
            k = np.where(z[:,:,0] == self.solver.centers[e,2])[0][0]
            res[j,i,k] = result[e]
            
        return x, y, z, res
