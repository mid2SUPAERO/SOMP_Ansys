import numpy as np

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from functools import partial

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

class Post3D():
    def __init__(self, solver):
        self.solver = solver

    # https://stackoverflow.com/questions/40853556/3d-discrete-heatmap-in-matplotlib
    def cuboid_data(center, size=1):
        # code taken from
        # http://stackoverflow.com/questions/30715083/python-plotting-a-wireframe-3d-cuboid?noredirect=1&lq=1
        # suppose axis direction: x: to left; y: to inside; z: to upper
        # get the (left, outside, bottom) point
        o = [a - b / 2 for a, b in zip(center, size)]
        # get the length, width, and height
        l = w = h = size
        x = [[o[0], o[0] + l, o[0] + l, o[0], o[0]],  # x coordinate of points in bottom surface
             [o[0], o[0] + l, o[0] + l, o[0], o[0]],  # x coordinate of points in upper surface
            [o[0], o[0] + l, o[0] + l, o[0], o[0]],  # x coordinate of points in outside surface
            [o[0], o[0] + l, o[0] + l, o[0], o[0]]]  # x coordinate of points in inside surface
        y = [[o[1], o[1], o[1] + w, o[1] + w, o[1]],  # y coordinate of points in bottom surface
             [o[1], o[1], o[1] + w, o[1] + w, o[1]],  # y coordinate of points in upper surface
             [o[1], o[1], o[1], o[1], o[1]],          # y coordinate of points in outside surface
             [o[1] + w, o[1] + w, o[1] + w, o[1] + w, o[1] + w]]    # y coordinate of points in inside surface
        z = [[o[2], o[2], o[2], o[2], o[2]],                        # z coordinate of points in bottom surface
             [o[2] + h, o[2] + h, o[2] + h, o[2] + h, o[2] + h],    # z coordinate of points in upper surface
             [o[2], o[2], o[2] + h, o[2] + h, o[2]],                # z coordinate of points in outside surface
             [o[2], o[2], o[2] + h, o[2] + h, o[2]]]                # z coordinate of points in inside surface
        return x, y, z

    def plotCubeAt(pos=(0,0,0), c="b", alpha=0.1, ax=None):
        # Plotting N cube elements at position pos
        if ax !=None:
            X, Y, Z = cuboid_data( (pos[0],pos[1],pos[2]) )
            ax.plot_surface(X, Y, Z, color=c, rstride=1, cstride=1, alpha=0.1)
    
    def plotMatrix(ax, x, y, z, data, cax=None, alpha=0.1):
        # plot a Matrix 
        norm = matplotlib.colors.Normalize(vmin=0., vmax=1.)
        colors = lambda i : matplotlib.cm.ScalarMappable(norm=norm,cmap = 'binary').to_rgba(data[i]) 
        for i in range(len(x)):
            plotCubeAt(pos=(x[i],y[i],z[i]), c=colors(i), alpha=alpha, ax=ax)

        if cax != None:
            cbar = matplotlib.colorbar.ColorbarBase(cax, cmap='binary', norm=norm, orientation='vertical')  
            cbar.set_ticks(np.unique(data))
            # set the colorbar transparent as well
            cbar.solids.set(alpha=alpha)     
    
    def plot(self, iteration=-1, filename=None, save=True):
        x = self.solver.centers[:,0]
        y = self.solver.centers[:,1]
        z = self.solver.centers[:,2]
        data = self.solver.rho_hist[iteration]

        fig = plt.figure(figsize=(10,4))
        ax = fig.add_axes([0.1, 0.1, 0.7, 0.8], projection='3d')
        ax_cb = fig.add_axes([0.8, 0.3, 0.05, 0.45])
        ax.set_aspect('equal')
    
        plotMatrix(ax, x, y, z, data, cax=ax_cb)
        if save:
            if filename is None: filename = self.solver.res_dir / 'design.png'
            plt.savefig(filename)
