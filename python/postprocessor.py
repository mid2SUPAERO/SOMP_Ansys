import numpy as np

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from functools import partial

class PostProcessor():
    def __init__(self, solver):
        self.solver = solver
        
    def plot_convergence(self, start_iter=0, filename=None, save=True):
        plt.figure()
        plt.plot(range(start_iter,len(self.solver.comp_hist)), self.solver.comp_hist[start_iter:])
        plt.xlabel('Iteration')
        plt.ylabel('Compliance')
        
        if save:
            if filename is None: filename = self.solver.res_dir / 'convergence.png'
            plt.savefig(filename)

class Post2D(PostProcessor):
    def plot(self, iteration=-1, colorful=False, filename=None, save=True, fig=None, ax=None):
        if fig == None: fig, ax = plt.subplots(dpi=300)
        ax.cla()
        ax.set_aspect('equal')
        plt.xlim(np.amin(self.solver.node_coord[:,0]),np.amax(self.solver.node_coord[:,0]))
        plt.ylim(np.amin(self.solver.node_coord[:,1]),np.amax(self.solver.node_coord[:,1]))
        plt.title('Compliance = {:.4f}'.format(self.solver.comp_hist[iteration]))
        
        x = self.solver.centers[:,0]
        y = self.solver.centers[:,1]
        rho   = self.solver.rho_hist[iteration]
        theta = self.solver.theta_hist[iteration]
        
        u = np.cos(theta)
        v = np.sin(theta)
        if colorful:
            color = [(0,np.abs(u[i]),np.abs(v[i])) for i in range(len(u))]
        else:
            color = 'black'
        
        ax.quiver(x, y, u, v, color=color, alpha=rho, pivot='mid', headwidth=0, headlength=0, headaxislength=0, linewidth=1.5)

        if save:
            if filename is None: filename = self.solver.res_dir / 'design.png'
            plt.savefig(filename)
        
    def animate(self, filename=None, colorful=False):
        if filename is None: filename = self.solver.res_dir / 'animation.gif'
        fig, ax = plt.subplots(dpi=300)
        anim = FuncAnimation(fig, partial(self.plot, colorful=colorful, save=False, fig=fig, ax=ax), frames=len(self.solver.rho_hist))
        anim.save(filename)

class Post3D(PostProcessor):
    def plot(self, iteration=-1, colorful=True, elev=None, azim=None, filename=None, save=True, fig=None, ax=None):
        if ax is None:
            fig = plt.figure(dpi=500)
            ax = fig.add_axes([0,0,1,1], projection='3d')
        ax.cla()
        deltax = np.amax(self.solver.node_coord[:,0]) - np.amin(self.solver.node_coord[:,0])
        deltay = np.amax(self.solver.node_coord[:,1]) - np.amin(self.solver.node_coord[:,1])
        deltaz = np.amax(self.solver.node_coord[:,2]) - np.amin(self.solver.node_coord[:,2])
        ax.set_box_aspect((deltax,deltay,deltaz))            
        ax.view_init(elev=elev, azim=azim)
        plt.title('Compliance = {:.4f}'.format(self.solver.comp_hist[iteration])) 
        
        x = self.solver.centers[:,0]
        y = self.solver.centers[:,1]
        z = self.solver.centers[:,2]
        theta = self.solver.theta_hist[iteration]
        alpha = self.solver.alpha_hist[iteration]
        u = np.cos(alpha)*np.cos(theta)
        v = np.cos(alpha)*np.sin(theta)
        w = np.sin(alpha)
        
        rho = self.solver.rho_hist[iteration]
        if colorful:
            color = [(np.abs(w[i]),np.abs(u[i]),np.abs(v[i])) for i in range(len(u))]
        else:
            color = 'black'
        
        ax.quiver(x, y, z, u, v, w, color=color, alpha=rho, pivot='middle', arrow_length_ratio=0, linewidth=1.5, length=3)
        
        if save:
            if filename is None: filename = self.solver.res_dir / 'design.png'
            plt.savefig(filename)
        
    def plot_layer(self, iteration=-1, layer=0, colorful=False, filename=None, save=True, fig=None, ax=None):
        if fig == None: fig, ax = plt.subplots(dpi=300)
        ax.cla()
        ax.set_aspect('equal')
        plt.xlim(np.amin(self.solver.node_coord[:,0]),np.amax(self.solver.node_coord[:,0]))
        plt.ylim(np.amin(self.solver.node_coord[:,1]),np.amax(self.solver.node_coord[:,1]))
        plt.title('Layer {} - Compliance = {:.4f}'.format(layer, self.solver.comp_hist[iteration]))
        
<<<<<<< HEAD
        z = np.unique(self.solver.centers[:,2])[layer]
        idx = np.where(self.solver.centers[:,2] == z)[0]
        
        x = self.solver.centers[idx,0]
        y = self.solver.centers[idx,1]
        rho   = self.solver.rho_hist[iteration][idx]
        theta = self.solver.theta_hist[iteration][idx]
        
        u = np.cos(theta)
        v = np.sin(theta)
        if colorful:
            color = [(0,np.abs(u[i]),np.abs(v[i])) for i in range(len(u))]
        else:
            color = 'black'
        
        ax.quiver(x, y, u, v, color=color, alpha=rho, pivot='mid', headwidth=0, headlength=0, headaxislength=0, linewidth=1.5)
=======
        x, y, density = self.make_grid(layer, self.solver.rho_hist[iteration], nodes=True)
        norm = colors.Normalize(vmin=0, vmax=1)
        ax.pcolormesh(x, y, density, cmap='binary', norm=norm)
        
        x, y, theta = self.make_grid(layer, self.solver.theta_hist[iteration], nodes=False)
        ax.quiver(x, y, np.cos(theta), np.sin(theta), color='white', pivot='mid', headwidth=0, headlength=0, headaxislength=0)
>>>>>>> b9e2e13110c51d37cde6f045b8067f9343d76e8f

        if save:
            if filename is None: filename = self.solver.res_dir / f'design_layer{layer}.png'
            plt.savefig(filename)
<<<<<<< HEAD
=======
            
    def plot_orientation(self, iteration=-1, threshold=0.3, elev=None, azim=None, filename=None, save=True, fig=None, ax=None):
        if ax is None:
            fig = plt.figure(dpi=500)
            ax = fig.add_axes([0,0,1,1], projection='3d')
            deltax = np.amax(self.solver.node_coord[:,0]) - np.amin(self.solver.node_coord[:,0])
            deltay = np.amax(self.solver.node_coord[:,1]) - np.amin(self.solver.node_coord[:,1])
            deltaz = np.amax(self.solver.node_coord[:,2]) - np.amin(self.solver.node_coord[:,2])
            ax.set_box_aspect((deltax,deltay,deltaz))            
            ax.view_init(elev=elev, azim=azim)
        ax.cla()
        plt.title('Compliance = {:.4f}'.format(self.solver.comp_hist[iteration])) 
        
        x = self.solver.centers[:,0]
        y = self.solver.centers[:,1]
        z = self.solver.centers[:,2]
        theta = self.solver.theta_hist[iteration]
        alpha = self.solver.alpha_hist[iteration]
        u = np.cos(alpha)*np.cos(theta)
        v = np.cos(alpha)*np.sin(theta)
        w = np.sin(alpha)
        
        rho = self.solver.rho_hist[iteration]
        x = x[rho > threshold]
        y = y[rho > threshold]
        z = z[rho > threshold]
        u = u[rho > threshold]
        v = v[rho > threshold]
        w = w[rho > threshold]
        color = [(u[i],v[i],w[i]) for i in range(len(u))]
        
        ax.quiver(x, y, z, u, v, w, color=color, pivot='middle', arrow_length_ratio=0, linewidth=0.3, length=3)
>>>>>>> b9e2e13110c51d37cde6f045b8067f9343d76e8f
        
    def animate(self, filename=None, colorful=True, elev=None, azim=None):
        if filename is None: filename = self.solver.res_dir / 'animation.gif'
        
        fig = plt.figure(dpi=500)
        ax = fig.add_axes([0,0,1,1], projection='3d')
        anim = FuncAnimation(fig, partial(self.plot, colorful=colorful, elev=elev, azim=azim, fig=fig, ax=ax), frames=len(self.solver.rho_hist))
        anim.save(filename)
        
    def animate_layer(self, layer, colorful=False, filename=None):
        if filename is None: filename = self.solver.res_dir / f'animation_layer{layer}.gif'
        fig, ax = plt.subplots(dpi=300)
<<<<<<< HEAD
        anim = FuncAnimation(fig, partial(self.plot_layer, layer=layer, colorful=colorful, save=False, fig=fig, ax=ax), frames=len(self.solver.rho_hist))
        anim.save(filename)
=======
        anim = FuncAnimation(fig, partial(self.plot_layer, layer=layer, save=False, fig=fig, ax=ax), frames=len(self.solver.rho_hist))
        anim.save(filename)
    
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
        
    def make_grid(self, layer, result, nodes):
        x, y = np.meshgrid(np.unique(self.solver.centers[:,0]),np.unique(self.solver.centers[:,1]))
        z = np.unique(self.solver.centers[:,2])[layer]
        res = np.zeros_like(x)
        for e in range(self.solver.num_elem):
            if not self.solver.centers[e,2] == z: continue
            i = np.where(x[0,:] == self.solver.centers[e,0])[0][0]
            j = np.where(y[:,0] == self.solver.centers[e,1])[0][0]
            res[j,i] = result[e]
        
        if nodes:
            x, y = np.meshgrid(np.unique(self.solver.node_coord[:,0]),np.unique(self.solver.node_coord[:,1]))
            
        return x, y, res
>>>>>>> b9e2e13110c51d37cde6f045b8067f9343d76e8f
