import numpy as np
from scipy import interpolate

from matplotlib import cm, colors
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib import path
from matplotlib.animation import FuncAnimation
from functools import partial

from stl import mesh
from mpl_toolkits import mplot3d
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset

from ansys.dpf import core as dpf

class PostProcessor():
    def __init__(self, solver):
        self.solver = solver
        self.quiver_scale = 1/(0.8*self.solver.elem_size)
        
    def plot_convergence(self, start_iter=0, penal=False, filename=None, save=True):
        if penal:
            fig, (ax1, ax2) = plt.subplots(2, 1)
        else:
            fig, ax1 = plt.subplots()
        
        for lc in range(self.solver.num_load_cases):
            ax1.plot(range(start_iter,len(self.solver.comp_hist[lc])), self.solver.comp_hist[lc][start_iter:], label=f'Load case {lc}')
                
        if not self.solver.num_load_cases == 1:
            ax1.legend()
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Compliance')
        
        if penal:
            ax2.plot(range(start_iter,len(self.solver.penal_hist)), self.solver.penal_hist[start_iter:])
            ax2.set_xlabel('Iteration')
            ax2.set_ylabel('Penalization factor')
        
        if save:
            if filename is None: filename = self.solver.res_dir / 'convergence.png'
            plt.savefig(filename)

class Post2D(PostProcessor):
    def __init__(self, solver):
        if solver.dim != 'SIMP2D' and solver.dim != '2D':
            raise ValueError('solver does not contain a 2D optimization')
        super().__init__(solver)

    def plot_orientations(self, iteration=-1, colorful=False, filename=None, save=True, fig=None, ax=None, zoom=None):
        if fig is None: fig, ax = plt.subplots(dpi=300)
        ax.cla()
        ax.set_aspect('equal')
        plt.xlim(np.min(self.solver.node_coord[:,0]),np.max(self.solver.node_coord[:,0]))
        plt.ylim(np.min(self.solver.node_coord[:,1]),np.max(self.solver.node_coord[:,1]))
        
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
            if filename is None: filename = self.solver.res_dir / 'design.png'
            plt.savefig(filename)
        
    def animate(self, filename=None, colorful=False):
        if filename is None: filename = self.solver.res_dir / 'animation.gif'
        fig, ax = plt.subplots(dpi=300)
        anim = FuncAnimation(fig, partial(self.plot, colorful=colorful, save=False, fig=fig, ax=ax), frames=len(self.solver.rho_hist))
        anim.save(filename)

    def plot_fibers(self, iteration=-1, filename=None, save=True, fig=None, ax=None):
        if ax is None:
                fig, ax = plt.subplots(dpi=300)
                ax.set_aspect('equal')
                plt.xlim(np.amin(self.solver.node_coord[0,:]),np.amax(self.solver.node_coord[0,:]))
                plt.ylim(np.amin(self.solver.node_coord[1,:]),np.amax(self.solver.node_coord[1,:]))
        ax.cla()

        rho   = self.solver.rho_hist[iteration]
        theta = self.solver.theta_hist[iteration]
        u = np.cos(theta)
        v = np.sin(theta)
        
        xx = np.linspace(np.amin(self.solver.node_coord[0,:]), np.amax(self.solver.node_coord[0,:]), 50)
        yy = np.linspace(np.amin(self.solver.node_coord[1,:]), np.amax(self.solver.node_coord[1,:]), 50)
        xx, yy = np.meshgrid(xx, yy)

        inside = np.full(xx.shape, False)
        for i in range(self.solver.num_elem):
            center = self.solver.centers[:2,i].T
            xy = np.array([center + np.array([-self.solver.elem_size, -self.solver.elem_size]),
                            center + np.array([-self.solver.elem_size, self.solver.elem_size]),
                            center + np.array([self.solver.elem_size, self.solver.elem_size]),
                            center + np.array([self.solver.elem_size, -self.solver.elem_size])])
            elm_path = path.Path(xy)
            inside_elm = elm_path.contains_points(np.hstack((xx.flatten()[:,np.newaxis],yy.flatten()[:,np.newaxis]))).reshape(xx.shape)
            inside |= inside_elm
        outside = np.logical_not(inside)

        u_interp = interpolate.griddata(self.solver.centers[:2,:].T, u, (xx,yy), method='nearest')
        v_interp = interpolate.griddata(self.solver.centers[:2,:].T, v, (xx,yy), method='nearest')
        rho_interp = interpolate.griddata(self.solver.centers[:2,:].T, rho, (xx,yy), method='nearest')
        rho_interp[outside] = 0

        ax.streamplot(xx, yy, u_interp, v_interp, arrowstyle='-', linewidth=1, density=3, color=rho_interp, cmap=cm.binary)

        if save:
            if filename is None: filename = self.solver.res_dir / f'fibers.png'
            plt.savefig(filename)

    def animate_fibers(self, filename=None):
        if filename is None: filename = self.solver.res_dir / 'animation_fibers.gif'
        fig, ax = plt.subplots(dpi=300)
        anim = FuncAnimation(fig, partial(self.plot_fibers, save=False, fig=fig, ax=ax), frames=len(self.solver.rho_hist))
        anim.save(filename)

    def plot_density(self, iteration=-1, filename=None, save=True, fig=None, ax=None):
        if fig is None: fig, ax = plt.subplots(dpi=300)
        ax.cla()
        ax.set_aspect('equal')
        plt.xlim(np.min(self.solver.node_coord[:,0]),np.max(self.solver.node_coord[:,0]))
        plt.ylim(np.min(self.solver.node_coord[:,1]),np.max(self.solver.node_coord[:,1]))
        
        rho = self.solver.rho_hist[iteration]
        for i in range(self.solver.num_elem):
            xy = self.solver.node_coord[self.solver.elmnodes[i],:2]
            ax.add_patch(Polygon(xy, facecolor='k', edgecolor=None, alpha=rho[i]))

        if save:
            if filename is None: filename = self.solver.res_dir / 'density.png'
            plt.savefig(filename)

    def animate_density(self, filename=None):
        if filename is None: filename = self.solver.res_dir / 'animation_density.gif'
        fig, ax = plt.subplots(dpi=300)
        anim = FuncAnimation(fig, partial(self.plot_density, save=False, fig=fig, ax=ax), frames=len(self.solver.rho_hist))
        anim.save(filename)

class Post3D(PostProcessor):
    def __init__(self, solver):
        if solver.dim != 'SIMP3D' and solver.dim != '3D_layer' and solver.dim != '3D_free':
            raise ValueError('solver does not contain a 3D optimization')
        super().__init__(solver)

    def plot(self, iteration=-1, colorful=True, printability=False, elev=None, azim=None, domain_stl=None, filename=None, save=True, fig=None, ax=None):
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
        
        x, y, z = self.solver.centers.T
        theta = self.solver.theta_hist[iteration]
        alpha = self.solver.alpha_hist[iteration]
        
        # orientations in the printing coordinate system
        u = np.cos(alpha)*np.cos(theta)
        v = np.cos(alpha)*np.sin(theta)
        w = np.sin(alpha)

        # orientations in the global coordinate system
        u, v, w = self.__printing2global(np.vstack((u,v,w)).T).T
        
        rho = self.solver.rho_hist[iteration]
        if printability:
            color = ['black' if printable else 'red' for printable in self.solver.elm_printability]
            
            # plot print direction
            origin = np.where(self.solver.print_direction<0, maxdim, 0)
            plt.quiver(*origin, *self.solver.print_direction, label='Printing direction', color='r', length=0.2*np.max(maxdim), linewidth=2)
        elif colorful:
            color = [(np.abs(u[i]),np.abs(v[i]),np.abs(w[i])) for i in range(len(u))]
            
            # plot coordinate system
            plt.quiver(*mindim, 1., 0., 0., color=(1.,0.,0.), length=0.1*np.max(maxdim), linewidth=2)
            plt.quiver(*mindim, 0., 1., 0., color=(0.,1.,0.), length=0.1*np.max(maxdim), linewidth=2)
            plt.quiver(*mindim, 0., 0., 1., color=(0.,0.,1.), length=0.1*np.max(maxdim), linewidth=2)
        else:
            color = 'black'     
            
        ax.quiver(x, y, z, u, v, w, color=color, alpha=rho, pivot='middle', arrow_length_ratio=0, linewidth=0.8, length=1/self.quiver_scale)
        
        if save:
            if filename is None: filename = self.solver.res_dir / 'design.png'
            plt.savefig(filename)

    def plot_layer_density(self, iteration=-1, layer=0, filename=None, save=True, fig=None, ax=None):
        if fig is None: fig, ax = plt.subplots(dpi=300)
                
        idx = self.solver.layers[layer]
        plot_lim = self.__global2printing(self.solver.node_coord)
        
        ax.cla()
        ax.set_aspect('equal')
        plt.xlim(np.min(plot_lim[:,0]),np.max(plot_lim[:,0]))
        plt.ylim(np.min(plot_lim[:,1]),np.max(plot_lim[:,1]))
        plt.title('Layer {}/{}'.format(layer+1,len(self.solver.layers)))
        
        x, y, _ = self.__global2printing(self.solver.centers[idx,:]).T
        
        rho   = self.solver.rho_hist[iteration]
        for i in idx:
            xy = self.solver.node_coord[self.solver.elmnodes[i],:2]
            ax.add_patch(Polygon(xy, facecolor='k', edgecolor=None, alpha=rho[i]))

        if save:
            if filename is None: filename = self.solver.res_dir / f'density_layer{layer}.png'
            plt.savefig(filename)
        
    def plot_layer(self, iteration=-1, layer=0, colorful=False, printability=False, filename=None, save=True, fig=None, ax=None, zoom=None):
        if fig is None: fig, ax = plt.subplots(dpi=300)
                
        idx = self.solver.layers[layer]
        plot_lim = self.__global2printing(self.solver.node_coord)
        
        ax.cla()
        ax.set_aspect('equal')
        plt.xlim(np.min(plot_lim[:,0]),np.max(plot_lim[:,0]))
        plt.ylim(np.min(plot_lim[:,1]),np.max(plot_lim[:,1]))
        plt.title('Layer {}/{}'.format(layer+1,len(self.solver.layers)))
        
        x, y, _ = self.__global2printing(self.solver.centers[idx,:]).T

        rho   = self.solver.rho_hist[iteration][idx]
        theta = self.solver.theta_hist[iteration][idx]
        u = np.cos(theta)
        v = np.sin(theta)
        
        if printability:
            color = ['black' if printable else 'red' for printable in self.solver.elm_printability[idx]]
        elif colorful:
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
            if filename is None: filename = self.solver.res_dir / f'orientations_layer{layer}.png'
            plt.savefig(filename)

    def plot_fibers(self, iteration=-1, layer=None, global_frame=False, elev=None, azim=None, roll=None, domain_stl=None, filename=None, save=True, fig=None, ax=None):
        if filename is None:
            if layer is None:
                filename = self.solver.res_dir / f'fibers.png'
            else:
                filename = self.solver.res_dir / f'fibers_layer{layer}.png'

        if type(layer) is int: layer = (layer,)
        if layer is None: layer = range(len(self.solver.layers))
        plotted_layers = len(layer)
        norm = colors.Normalize(vmin=min(layer), vmax=max(layer))
        cmap = cm.jet if plotted_layers > 1 else cm.Greys_r

        plot_lim    = self.solver.node_coord if global_frame else self.__global2printing(self.solver.node_coord)
        print_coord = self.solver.centers if global_frame else self.__global2printing(self.solver.centers)

        if plotted_layers > 1:
            if ax is None:
                fig = plt.figure(dpi=500)
                ax = fig.add_axes([0,0,1,1], projection='3d')
            maxdim = np.max(plot_lim, axis=0)
            mindim = np.min(plot_lim, axis=0)
            ax.set_box_aspect((maxdim-mindim))
            ax.axes.set_xlim3d(left=0, right=maxdim[0]-mindim[0])
            ax.axes.set_ylim3d(bottom=0, top=maxdim[1]-mindim[1])
            ax.axes.set_zlim3d(bottom=0, top=maxdim[2]-mindim[2])
            ax.view_init(elev=elev, azim=azim, roll=roll)
        else:
            if ax is None: fig, ax = plt.subplots(dpi=300)
            ax.set_aspect('equal')
            plt.xlim(np.min(plot_lim[:,0]),np.max(plot_lim[:,0]))
            plt.ylim(np.min(plot_lim[:,1]),np.max(plot_lim[:,1]))
        ax.cla()
        if plotted_layers == 1: plt.title('Layer {}/{}'.format(layer[0]+1,len(self.solver.layers)))

        if plotted_layers > 1 and domain_stl is not None:
            domain_mesh = mesh.Mesh.from_file(domain_stl)
            if not global_frame:
                euler1, euler2 = self.solver.print_euler
                domain_mesh.rotate([0,0,1],-euler1)
                domain_mesh.rotate([1,0,0],-euler2)
            offset_x, offset_y, offset_z = np.min(domain_mesh.x)-np.min(self.solver.node_coord[:,0]), np.min(domain_mesh.y)-np.min(self.solver.node_coord[:,1]), np.min(domain_mesh.z)-np.min(self.solver.node_coord[:,2])
            domain_mesh.x -= offset_x
            domain_mesh.y -= offset_y
            domain_mesh.z -= offset_z
            ax.add_collection3d(mplot3d.art3d.Poly3DCollection(domain_mesh.vectors, alpha=0.1, linewidth=1))

        for layer in layer:
            if plotted_layers > 1 : fig_tmp, ax_tmp = plt.subplots()
    
            idx = self.solver.layers[layer]
            rho   = self.solver.rho_hist[iteration][idx]
            theta = self.solver.theta_hist[iteration][idx]
            u = np.cos(theta)
            v = np.sin(theta)
            
            xx = np.linspace(np.min(plot_lim[:,0]), np.max(plot_lim[:,0]), 50)
            yy = np.linspace(np.min(plot_lim[:,1]), np.max(plot_lim[:,1]), 50)
            xx, yy = np.meshgrid(xx, yy)

            inside = np.full(xx.shape, False)
            for i in idx:
                center = print_coord[i,:2].T
                xy = np.array([center + np.array([-self.solver.elem_size, -self.solver.elem_size]),
                                center + np.array([-self.solver.elem_size, self.solver.elem_size]),
                                center + np.array([self.solver.elem_size, self.solver.elem_size]),
                                center + np.array([self.solver.elem_size, -self.solver.elem_size])])
                elm_path = path.Path(xy)
                inside_elm = elm_path.contains_points(np.hstack((xx.flatten()[:,np.newaxis],yy.flatten()[:,np.newaxis]))).reshape(xx.shape)
                inside |= inside_elm
            outside = np.logical_not(inside)

            u_interp = interpolate.griddata(print_coord[idx,:2], u, (xx,yy), method='nearest')
            v_interp = interpolate.griddata(print_coord[idx,:2], v, (xx,yy), method='nearest')
            rho_interp = interpolate.griddata(print_coord[idx,:2], rho, (xx,yy), method='nearest')
            rho_interp[outside] = 0

            if plotted_layers > 1:
                res = ax_tmp.streamplot(xx, yy, u_interp, v_interp, arrowstyle='-', linewidth=1, density=3, color=rho_interp, cmap=cm.binary)
                plt.close(fig_tmp)

                for line in res.lines.get_paths():
                    x = line.vertices.T[0]
                    y = line.vertices.T[1]
                    z = self.solver.layer_thk*(layer+0.5) * np.ones_like(x)

                    rho_plot = interpolate.griddata(np.vstack((xx.flatten(),yy.flatten())).T, rho_interp.flatten(), (x[0],y[0]), method='nearest')

                    if global_frame:
                        x, y, z = self.__printing2global(np.vstack((x,y,np.zeros_like(x))).T).T
                        # x += self.solver.print_direction[0] * self.solver.layer_thk*(layer+0.5)
                        # y += self.solver.print_direction[1] * self.solver.layer_thk*(layer+0.5)
                        # z += self.solver.print_direction[2] * self.solver.layer_thk*(layer+0.5)

                    ax.plot(x, y, z, color=cmap(norm(layer)), alpha=rho_plot, linewidth=1)
            else:
                ax.streamplot(xx, yy, u_interp, v_interp, arrowstyle='-', linewidth=1, density=3, color=rho_interp, cmap=cm.binary)

        if save: plt.savefig(filename)

    def plot_iso(self, iteration=-1, threshold=0.8, elev=30, azim=-60, filename=None):
        if filename is None: filename = self.solver.res_dir / 'isosurface.png'

        rst_path = self.solver.res_dir / (self.solver.jobname + '.rst')
        model = dpf.Model(rst_path)

        density_field = dpf.fields_factory.field_from_array(self.solver.rho_hist[iteration])
        density_field.location = dpf.locations.elemental
        density_field.meshed_region = model.metadata.meshed_region
        
        density_field = dpf.operators.averaging.elemental_to_nodal(field=density_field).eval() # nodal interpolation
        iso_surface = dpf.operators.mesh.mesh_cut(field=density_field, iso_value=threshold, closed_surface=1, slice_surfaces=True).outputs.mesh()

        # setup camera for plot
        maxdim = np.max(self.solver.node_coord, axis=0)
        mindim = np.min(self.solver.node_coord, axis=0)
        elev = np.deg2rad(elev)
        azim = np.deg2rad(azim)

        center = 0.5*(maxdim+mindim)
        distance = 1.5 * np.sqrt(np.sum(np.square(maxdim-mindim)))
        rel_pos = np.array([distance*np.cos(elev)*np.cos(azim), distance*np.cos(elev)*np.sin(azim), distance*np.sin(elev)])
        cpos = [center + rel_pos, center, (0, 0, 1)] # camera position, focal point, up vector

        iso_surface.plot(show_edges=False, screenshot=filename, cpos=cpos)

    def animate(self, filename=None, colorful=True, printability=False, elev=None, azim=None):
        if filename is None: filename = self.solver.res_dir / 'animation.gif'
        
        fig = plt.figure(dpi=500)
        ax = fig.add_axes([0,0,1,1], projection='3d')
        anim = FuncAnimation(fig, partial(self.plot, colorful=colorful, printability=printability, elev=elev, azim=azim, save=False, fig=fig, ax=ax), frames=len(self.solver.rho_hist))
        anim.save(filename)
        
    def animate_layer(self, layer=0, colorful=False, filename=None):
        if filename is None: filename = self.solver.res_dir / f'animation_layer{layer}.gif'
        fig, ax = plt.subplots(dpi=300)
        anim = FuncAnimation(fig, partial(self.plot_layer, layer=layer, colorful=colorful, save=False, fig=fig, ax=ax), frames=len(self.solver.rho_hist))
        anim.save(filename)

    def animate_fibers(self, layer=None, filename=None):
        if filename is None:
            if layer is None:
                filename = self.solver.res_dir / f'animation_fibers.png'
            else:
                filename = self.solver.res_dir / f'animation_fibers_layer{layer}.png'

        if type(layer) is int: layer = (layer,)
        if layer is None: layer = range(len(self.solver.layers))
        plotted_layers = len(layer)

        if plotted_layers > 1:
            fig = plt.figure(dpi=500)
            ax = fig.add_axes([0,0,1,1], projection='3d')
        else:
            fig, ax = plt.subplots(dpi=300)
        anim = FuncAnimation(fig, partial(self.plot_fibers, layer=layer, save=False, fig=fig, ax=ax), frames=len(self.solver.rho_hist))
        anim.save(filename)
        
    def animate_print(self, fibers=True, colorful=False, printability=False, filename=None):
        if filename is None:
            if fibers:
                filename = self.solver.res_dir / 'animation_print_fibers.gif'
            else:
                filename = self.solver.res_dir / 'animation_print_orientations.gif'
        fig, ax = plt.subplots(dpi=300)
        if fibers:
            anim = FuncAnimation(fig, lambda layer: self.plot_fibers(iteration=-1, layer=layer, save=False, fig=fig, ax=ax), frames=len(self.solver.layers))
        else:
            anim = FuncAnimation(fig, lambda layer: self.plot_layer(iteration=-1, layer=layer, colorful=colorful, printability=printability, save=False, fig=fig, ax=ax), frames=len(self.solver.layers))
        anim.save(filename)

    # coord.shape = (npoints, 3)
    def __printing2global(self, coord):
        euler1, euler2 = self.solver.print_euler # rotations around x (-euler2) and z' (-euler1) respectively
        T = np.array([[np.cos(-euler1),-np.sin(-euler1),0],
                    [np.sin(-euler1),np.cos(-euler1),0],
                    [0,0,1]]) @ \
            np.array([[1,0,0],
                    [0,np.cos(-euler2),-np.sin(-euler2)],
                    [0,np.sin(-euler2),np.cos(-euler2)]])
            
        return T.dot(coord.T).T

    def __global2printing(self, coord):
        euler1, euler2 = self.solver.print_euler # rotations around z (euler1) and x' (euler2) respectively
        T = np.array([[1,0,0],
                    [0,np.cos(euler2),-np.sin(euler2)],
                    [0,np.sin(euler2),np.cos(euler2)]]) @ \
            np.array([[np.cos(euler1),-np.sin(euler1),0],
                    [np.sin(euler1),np.cos(euler1),0],
                    [0,0,1]])
        return T.dot(coord.T).T