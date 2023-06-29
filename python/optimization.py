import subprocess
import time
import os, glob
import numpy as np

from .filters import DensityFilter, OrientationFilter
from .mma import MMA

from .dkdt2d import dkdt2d
from .dkdt3d import dkdt3d

# Starting point:
# https://github.com/pep-pig/Topology-optimization-of-structure-via-simp-method
"""
Required APDL script files:
    ansys_meshdata_2d.txt: gets mesh properties for 2D optimization
    ansys_meshdata_3d.txt: gets mesh properties for 3D optimization
    ansys_solve.txt: calls finite element analysis and stores results

TopOpt2D: 4-node 2D quad, PLANE182 in a rectangular domain, with KEYOPT(3) = 3 plane stress with thk
TopOpt3D: 8-node 3D hex, SOLID185 in a cuboid domain

TopOpt(inputfile, Ex, Ey, nuxy, nuyz, Gxy, volfrac, r_rho, r_theta, theta0, max_iter, move_rho, move_theta, dim, jobname, echo):
    inputfile: name of the model file (without .db)
    Ex, Ey, nuxy, nuyz, Gxy: material properties
    volfrac: volume fraction constraint for the optimization
    r_rho: radius of the density filter (adjusts minimum feature size)
    r_theta: radius of the orientation filter (adjusts fiber curvature)
    theta0: initial orientation of the fibers, in degrees
    max_iter: number of iterations
    move_rho: move limit for densities
    move_theta: move limit for orientations, in degrees
    dim: optimization type, '2D' or '3D'
    jobname: subfolder of TopOpt.res_dir to store results for this optim. Default: stores results directly on TopOpt.res_dir
    echo: boolean, print status at each iteration?

Configure Ansys: Same configuration for all TopOpt objects
    set_paths(ANSYS_path, script_dir, res_dir, mod_dir): paths as pathlib.Path

Configure optimization:
    set_solid_elem(solid_elem): list of elements whose densities will be fixed on 1. Element indexing starting at 0
    
Optimization function: optim()
"""
class TopOpt():
    def set_paths(ANSYS_path, script_dir, res_dir, mod_dir):
        """
        ANSYS_path: MAPDL executable path
        script_dir: folder with .py files and .txt APDL scripts
        res_dir: folder to store results
        mod_dir: folder with the .db file (geometry, mesh, constraints, loads)
        """
        TopOpt.ANSYS_path = ANSYS_path
        TopOpt.script_dir = script_dir
        TopOpt.res_dir    = res_dir
        TopOpt.mod_dir    = mod_dir
    
    def __init__(self, inputfile, Ex, Ey, nuxy, nuyz, Gxy, volfrac, r_rho, r_theta, theta0, max_iter=200, move_rho=0.3, move_theta=5, dim='2D', jobname=None, echo=True):
        self.dim  = dim
        self.echo = echo
        self.dkdt = dkdt2d if dim == '2D' else dkdt3d
        
        self.jobname = jobname
        if jobname is None:
            self.res_dir = TopOpt.res_dir
        else:
            self.res_dir = TopOpt.res_dir / jobname
        self.res_dir.mkdir(parents=True, exist_ok=True)
        
        self.meshdata_cmd, self.result_cmd = self.build_apdl_scripts(inputfile)
        self.num_elem, self.num_node, self.centers, self.elemvol, self.elmnodes, self.node_coord = self.get_mesh_data()
    
        self.Ex     = Ex
        self.Ey     = Ey
        self.nuxy   = nuxy
        self.nuyz   = nuyz
        self.Gxy    = Gxy
        self.theta0 = np.deg2rad(theta0)
        
        self.volfrac = volfrac
        self.r_rho   = r_rho
        self.r_theta = r_theta
        self.penal   = 3
        
        self.max_iter   = max_iter
        self.move_rho   = move_rho
        self.move_theta = np.deg2rad(move_theta)
        self.rho_min    = 1e-3
        self.solid_elem = []
        
        self.density_filter     = DensityFilter(self.r_rho, self.centers)
        self.orientation_filter = OrientationFilter(self.r_theta, self.centers)
        
        self.fea_time   = 0
        self.deriv_time = 0
        self.mma        = self.create_optimizer()
    
    def build_apdl_scripts(self, inputfile):
        self.title = inputfile if self.jobname is None else self.jobname.replace('-','m')
        meshdata_base = 'ansys_meshdata_2d.txt' if self.dim == '2D' else 'ansys_meshdata_3d.txt'

        with open(self.res_dir/'ansys_meshdata.txt', 'w') as f:
            f.write(f"RESUME,'{inputfile}','db','{TopOpt.mod_dir}',0,0\n")
            f.write(f"/CWD,'{self.res_dir}'\n")
            f.write(f"/FILENAME,{self.title},1\n")
            f.write(f"/TITLE,{self.title}\n")
            f.write(open(TopOpt.script_dir/meshdata_base).read())
            
        with open(self.res_dir/'ansys_solve.txt', 'w') as f:
            f.write(f"RESUME,'{inputfile}','db','{TopOpt.mod_dir}',0,0\n")
            f.write(f"/CWD,'{self.res_dir}'\n")
            f.write(f"/FILENAME,{self.title},1\n")
            f.write(f"/TITLE,{self.title}\n")
            f.write(open(TopOpt.script_dir/'ansys_solve.txt').read())
                  
        meshdata_cmd = [TopOpt.ANSYS_path, '-b', '-i', self.res_dir/'ansys_meshdata.txt', '-o', self.res_dir/'meshdata.out', '-smp']
        result_cmd   = [TopOpt.ANSYS_path, '-b', '-i', self.res_dir/'ansys_solve.txt', '-o', self.res_dir/'solve.out', '-smp']

        if not self.jobname is None:
            meshdata_cmd += ['-j', self.title]
            result_cmd   += ['-j', self.title]
            
        return meshdata_cmd, result_cmd
    
    def get_mesh_data(self):
        subprocess.run(self.meshdata_cmd)
        
        num_elem, num_node = np.loadtxt(self.res_dir/'elements_nodes_counts.txt', dtype=int) # num_elm num_nodes
        centers            = np.loadtxt(self.res_dir/'elements_centers.txt')[:, 1:] # label x y z
        elmvol             = np.loadtxt(self.res_dir/'elements_volume.txt')[:,1] # label elmvol
        elmnodes           = np.loadtxt(self.res_dir/'elements_nodes.txt', dtype=int) - 1 # n1 n2 n3 n4 ...
        node_coord         = np.loadtxt(self.res_dir/'node_coordinates.txt') # x y z
        
        return num_elem, num_node, centers, elmvol, elmnodes, node_coord
    
    def create_optimizer(self):
        rho    = self.volfrac * np.ones(self.num_elem)
        theta  = self.theta0 * np.ones(self.num_elem)
        self.x = np.concatenate((rho,theta))

        xmin = np.concatenate((self.rho_min*np.ones_like(rho), -np.pi*np.ones_like(theta)))
        xmax = np.concatenate((np.ones_like(rho), np.pi*np.ones_like(theta)))
        move = np.concatenate((self.move_rho*np.ones(self.num_elem), self.move_theta*np.ones(self.num_elem)))
        
        mma = MMA(self.fea,self.sensitivities,self.constraint,self.dconstraint,xmin,xmax,move)
        
        self.rho_hist   = []
        self.theta_hist = []
        self.comp_hist  = []
        
        return mma
    
    def fea(self, x):
        t0 = time.time()
        rho, theta = np.split(x,2)
        theta = self.orientation_filter.filter(rho,theta)
        
        # Generate file with material properties for each element
        Ex   = rho**self.penal * self.Ex
        Ey   = rho**self.penal * self.Ey
        nuxy = self.nuxy * np.ones(self.num_elem)
        nuyz = self.nuyz * np.ones(self.num_elem)
        Gxy  = rho**self.penal * self.Gxy
        Gyz  = Ey/(2*(1+nuyz))
        material = np.array([Ex, Ey, nuxy, nuyz, Gxy, Gyz, np.rad2deg(theta)]).T
        np.savetxt(self.res_dir/'material.txt', material, fmt=' %-.7E', newline='\n')
        
        # Solve
        subprocess.run(self.result_cmd)
        energy = np.loadtxt(self.res_dir/'strain_energy.txt', dtype=float) # strain_energy
        c = 2*np.sum(energy)

        # Save history
        self.rho_hist.append(rho)
        self.theta_hist.append(theta)
        self.comp_hist.append(c)

        if self.echo: print("compliance = {:10.4f}".format(c))
        self.fea_time += time.time() - t0
        return c
    
    def sensitivities(self, x):
        t0 = time.time()
        rho, theta = np.split(x,2)
        
        # dc/drho
        energy = np.loadtxt(self.res_dir/'strain_energy.txt', dtype=float) # strain_energy
        uku    = 2*energy/rho**self.penal # K: stiffness matrix with rho=1
        dcdrho = -self.penal * rho**(self.penal-1) * uku
        dcdrho = self.density_filter.filter(rho, dcdrho)
        dcdrho[self.solid_elem] = 0
        
        # dc/dtheta
        u = np.loadtxt(self.res_dir/'nodal_solution_u.txt', dtype=float) # ux uy uz
        if self.dim == '2D':
            u = u[:,[0,1]] # drop z displacement
        dcdt = np.zeros(self.num_elem)
        for i in range(self.num_elem):
            dkdt = self.dkdt(self.Ex,self.Ey,self.nuxy,self.nuyz,self.Gxy,theta[i],self.elemvol[i])
            ue = u[self.elmnodes[i,:],:].flatten()         
            dcdt[i] = -rho[i]**self.penal * ue.dot(dkdt.dot(ue))
        
#         import matplotlib.pyplot as plt
#         x, y = np.meshgrid(np.unique(self.centers[:,0]),np.unique(self.centers[:,1]))
#         res1, res2 = np.zeros_like(x), np.zeros_like(x)
#         for e in range(self.num_elem):
#             i = np.where(x[0,:] == self.centers[e,0])[0][0]
#             j = np.where(y[:,0] == self.centers[e,1])[0][0]
#             res1[j,i] = dcdt[e]
#             if self.mma.iter > 1: res2[j,i] = self.theta_hist[-1][e] - self.theta_hist[-2][e]
        
#         x, y = np.meshgrid(np.unique(self.node_coord[:,0]),np.unique(self.node_coord[:,1]))
#         plt.subplot(2,1,1)
#         plt.pcolormesh(x,y,res1,cmap='coolwarm')
#         plt.subplot(2,1,2)
#         plt.pcolormesh(x,y,res2,cmap='coolwarm')
#         plt.show()
        
        self.deriv_time += time.time() - t0
        return np.concatenate((dcdrho, dcdt))
    
    # sum(rho.v)/(volfrac.V) - 1 <= 0
    def constraint(self, x):
        rho, _ = np.split(x,2)
        return rho.dot(self.elemvol)/self.volfrac/np.sum(self.elemvol) - 1
    
    def dconstraint(self, x):
        return np.concatenate((self.elemvol/self.volfrac/np.sum(self.elemvol),np.zeros(self.num_elem)))
    
    def set_solid_elem(self, solid_elem):
        self.solid_elem    = solid_elem
        self.x[solid_elem] = 1
        
    def optim(self):
        t0 = time.time()
        for _ in range(self.max_iter):
            if self.echo: print("Starting iteration {:3d}...".format(self.mma.iter+1), end=' ')
            xnew = self.mma.iterate(self.x)

            rho, theta = np.split(xnew,2)
            theta = self.orientation_filter.filter(rho,theta)
            self.x = np.concatenate((rho,theta))
            
        # Evaluating result from last iteration
        if self.echo: print("Final design             ", end=' ')
        self.fea(self.x)
        
        self.clear_files()
        self.time = time.time() - t0
        return rho, theta
    
    # clear temporary Ansys files
    def clear_files(self):
        for filename in glob.glob('cleanup*'): os.remove(filename)
        for filename in glob.glob(self.title+'.*'): os.remove(filename)
