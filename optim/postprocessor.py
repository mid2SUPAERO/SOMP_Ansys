import numpy as np

from matplotlib import cm, colors
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from functools import partial

from stl import mesh
from mpl_toolkits import mplot3d
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset

class PostProcessor():
    def __init__(self, solver):
        self.solver = solver
        self.quiver_scale = 1/(0.8*np.mean(np.cbrt(self.solver.elemvol)))
        
    def plot_convergence(self, start_iter=0, penal=False, filename=None, save=True):
        if penal:
            fig, (ax1, ax2) = plt.subplots(2, 1)
        else:
            fig, ax1 = plt.subplots()
        
        for lc in range(self.solver.load_cases):
            ax1.plot(range(start_iter,len(self.solver.comp_hist[lc])), self.solver.comp_hist[lc][start_iter:], label=f'Load case {lc}')
                
        if not self.solver.load_cases == 1:
            ax1.legend()
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Compliance')
        
        if penal:
            ax2.plot(range(start_iter,len(self.solver.penal_hist)), self.solver.penal_hist[start_iter:])
            ax2.set_xlabel('Iteration')
            ax2.set_ylabel('Penalization factor')
        
        if save:
            if filename is None: filename = self.solver.res_root / 'convergence.png'
            plt.savefig(filename)

class Post2D(PostProcessor):
    def plot(self, iteration=-1, colorful=False, filename=None, save=True, fig=None, ax=None, zoom=None):
        if fig is None: fig, ax = plt.subplots(dpi=300)
        ax.cla()
        ax.set_aspect('equal')
        plt.xlim(np.amin(self.solver.node_coord[:,0]),np.amax(self.solver.node_coord[:,0]))
        plt.ylim(np.amin(self.solver.node_coord[:,1]),np.amax(self.solver.node_coord[:,1]))
        
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
        
        ax.quiver(x, y, u, v, color=color, alpha=rho, pivot='mid', headwidth=0, headlength=0, headaxislength=0, angles='xy', scale_units='xy', width=3e-3, scale=self.quiver_scale)
        
        if zoom is not None:
            axins = ax.inset_axes([zoom['xpos'],zoom['ypos'],zoom['width'],zoom['height']])
            axins.set_xlim(zoom['xmin'], zoom['xmax'])
            axins.set_ylim(zoom['ymin'], zoom['ymax'])
            axins.spines['bottom'].set_color(zoom['color'])
            axins.spines['top'].set_color(zoom['color'])
            axins.spines['right'].set_color(zoom['color'])
            axins.spines['left'].set_color(zoom['color'])
            ax.indicate_inset_zoom(axins, edgecolor=zoom['color'])
        
            axins.quiver(x, y, u, v, color=color, alpha=rho, pivot='mid', headwidth=0, headlength=0, headaxislength=0, angles='xy', scale_units='xy', width=1e-2, scale=self.quiver_scale)
            axins.set_xticks([])
            axins.set_yticks([])

        if save:
            if filename is None: filename = self.solver.res_root / 'design.png'
            plt.savefig(filename)
        
    def animate(self, filename=None, colorful=False):
        if filename is None: filename = self.solver.res_root / 'animation.gif'
        fig, ax = plt.subplots(dpi=300)
        anim = FuncAnimation(fig, partial(self.plot, colorful=colorful, save=False, fig=fig, ax=ax), frames=len(self.solver.rho_hist))
        anim.save(filename)

class Post3D(PostProcessor):
    def plot(self, iteration=-1, colorful=True, elev=None, azim=None, domain_stl=None, filename=None, save=True, fig=None, ax=None):
        if ax is None:
            fig = plt.figure(dpi=500)
            ax = fig.add_axes([0,0,1,1], projection='3d')
        ax.cla()
        
        maxdim = np.max(self.solver.node_coord, axis=0)
        mindim = np.min(self.solver.node_coord, axis=0)
        ax.set_box_aspect((maxdim-mindim))
        ax.view_init(elev=elev, azim=azim)

        if domain_stl is not None:
            domain_mesh = mesh.Mesh.from_file(domain_stl)
            ax.add_collection3d(mplot3d.art3d.Poly3DCollection(domain_mesh.vectors, alpha=0.1))
        
        x = self.solver.centers[:,0]
        y = self.solver.centers[:,1]
        z = self.solver.centers[:,2]
        theta = self.solver.theta_hist[iteration]
        alpha = self.solver.alpha_hist[iteration]
        
        # orientations in the printing coordinate system
        u = np.cos(alpha)*np.cos(theta)
        v = np.cos(alpha)*np.sin(theta)
        w = np.sin(alpha)
        
        rho = self.solver.rho_hist[iteration]
        if colorful:
            color = [(np.abs(w[i]),np.abs(u[i]),np.abs(v[i])) for i in range(len(u))]
            
            # plot coordinate system
            plt.quiver(*mindim, 1., 0., 0., color=(0.,1.,0.), length=0.1*np.max(maxdim), linewidth=2)
            plt.quiver(*mindim, 0., 1., 0., color=(0.,0.,1.), length=0.1*np.max(maxdim), linewidth=2)
            plt.quiver(*mindim, 0., 0., 1., color=(1.,0.,0.), length=0.1*np.max(maxdim), linewidth=2)
        else:
            color = 'black'     
            
        ax.quiver(x, y, z, u, v, w, color=color, alpha=rho, pivot='middle', arrow_length_ratio=0, linewidth=0.8, length=1/self.quiver_scale)
        
        if save:
            if filename is None: filename = self.solver.res_root / 'design.png'
            plt.savefig(filename)
        
    def plot_layer(self, iteration=-1, layer=0, colorful=False, filename=None, save=True, fig=None, ax=None, zoom=None):
        if fig is None: fig, ax = plt.subplots(dpi=300)
                
        idx = self.solver.layers[layer]
        plot_lim = self.solver.node_coord.T
        
        ax.cla()
        ax.set_aspect('equal')
        plt.xlim(np.amin(plot_lim[0,:]),np.amax(plot_lim[0,:]))
        plt.ylim(np.amin(plot_lim[1,:]),np.amax(plot_lim[1,:]))
        plt.title('Layer {}/{}'.format(layer+1,len(self.solver.layers)))
        
        print_coord = self.solver.centers[idx,:].T
        x, y = print_coord[0,:], print_coord[1,:]
        
        rho   = self.solver.rho_hist[iteration][idx]
        theta = self.solver.theta_hist[iteration][idx]
        u = np.cos(theta)
        v = np.sin(theta)
        
        if colorful:
            color = [(0,np.abs(u[i]),np.abs(v[i])) for i in range(len(u))]
        else:
            color = 'black'
        
        ax.quiver(x, y, u, v, color=color, alpha=rho, pivot='mid', headwidth=0, headlength=0, headaxislength=0, angles='xy', scale_units='xy', width=3e-3, scale=self.quiver_scale)
        
        if zoom is not None:
            axins = ax.inset_axes([zoom['xpos'],zoom['ypos'],zoom['width'],zoom['height']])
            axins.set_xlim(zoom['xmin'], zoom['xmax'])
            axins.set_ylim(zoom['ymin'], zoom['ymax'])
            axins.spines['bottom'].set_color(zoom['color'])
            axins.spines['top'].set_color(zoom['color'])
            axins.spines['right'].set_color(zoom['color'])
            axins.spines['left'].set_color(zoom['color'])
            ax.indicate_inset_zoom(axins, edgecolor=zoom['color'])
        
            axins.quiver(x, y, u, v, color=color, alpha=rho, pivot='mid', headwidth=0, headlength=0, headaxislength=0, angles='xy', scale_units='xy', width=1e-2, scale=self.quiver_scale)
            axins.set_xticks([])
            axins.set_yticks([])

        if save:
            if filename is None: filename = self.solver.res_root / f'design_layer{layer}.png'
            plt.savefig(filename)

    def plot_fill(self, iteration=-1, threshold=0.8, filename=None, save=True, fig=None, ax=None):
        data = self.solver.rho_hist[iteration]
        if ax is None:
            fig = plt.figure(dpi=500)
            ax = fig.add_axes([0,0,1,1], projection='3d')
            ax.set_box_aspect((np.amax(self.solver.node_coord[:,0]),np.amax(self.solver.node_coord[:,1]),np.amax(self.solver.node_coord[:,2])))
        ax.cla()
    
        cmap = cm.get_cmap('binary')
        for elem in range(len(data)):
            if data[elem] > threshold:
                nodes = self.solver.elmnodes[elem,:]
                coord = self.solver.node_coord[nodes,:]
                c = cmap(data[elem])
                alpha = data[elem]
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
                
        if save:
            if filename is None: filename = self.solver.res_root / 'design_fill.png'
            plt.savefig(filename)
        
    def animate(self, filename=None, colorful=True, elev=None, azim=None):
        if filename is None: filename = self.solver.res_root / 'animation.gif'
        
        fig = plt.figure(dpi=500)
        ax = fig.add_axes([0,0,1,1], projection='3d')
        anim = FuncAnimation(fig, partial(self.plot, colorful=colorful, elev=elev, azim=azim, fig=fig, ax=ax), frames=len(self.solver.rho_hist))
        anim.save(filename)
        
    def animate_layer(self, layer=0, colorful=False, filename=None):
        if filename is None: filename = self.solver.res_root / f'animation_layer{layer}.gif'
        fig, ax = plt.subplots(dpi=300)
        anim = FuncAnimation(fig, partial(self.plot_layer, layer=layer, colorful=colorful, save=False, fig=fig, ax=ax), frames=len(self.solver.rho_hist))
        anim.save(filename)
        
    def animate_print(self, colorful=False, filename=None):
        if filename is None: filename = self.solver.res_root / 'animation_print.gif'
        fig, ax = plt.subplots(dpi=300)
        anim = FuncAnimation(fig, lambda layer: self.plot_layer(iteration=-1, layer=layer, colorful=colorful, save=False, fig=fig, ax=ax), frames=len(self.solver.layers))
        anim.save(filename)
