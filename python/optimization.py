import subprocess
import time
from pathlib import Path
from abc import ABC, abstractmethod

import numpy as np

from .filters import MeshIndependenceFilter, OrientationRegularizationFilter, GaussianFilter
from .mma import MMA

# Starting point:
# https://github.com/pep-pig/Topology-optimization-of-structure-via-simp-method
"""
Required APDL script files:
    ansys_meshdata.txt: gets mesh properties
    ansys_solve.txt: calls finite element analysis and stores results

TopOpt2D: 4-node 2D quad, PLANE182 in a rectangular domain
TopOpt3D: 8-node 3D hex, SOLID185 in a cuboid domain

TopOpt2D/TopOpt3D(inputfile, Ex, nu, volfrac, rmin, penal, theta0, jobname):
    inputfile: name of the model file (without .db)
    Ex, Ey, Gxy, nu: material properties
    volfrac: volume fraction constraint for the optimization
    rmin: radius of the filter (adjusts minimum feature size)
    theta0: initial orientation of the fibers, in degrees
    jobname: subfolder of TopOpt.res_dir to store results for this optim. Defaults to no subfolder, stores results directly on TopOpt.res_dir

Configure Ansys:
    load_paths(ANSYS_path, script_dir, res_dir, mod_dir): paths as pathlib.Path
    set_processors(np): np - number of processors for Ansys
Optimization function: optim()
"""
class TopOpt(ABC):
    def load_paths(ANSYS_path, script_dir, res_dir, mod_dir):
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

    # Running on Shared Memory Parallel
    np = 2
    def set_processors(np):
        TopOpt.np = np
    
    def __init__(self, inputfile, Ex, Ey, Gxy, nu, volfrac, rmin, theta0, jobname=None):
        if jobname is None:
            self.res_dir = TopOpt.res_dir
        else:
            self.res_dir = TopOpt.res_dir / jobname
        self.res_dir.mkdir(parents=True, exist_ok=True)
        
        self.meshdata_cmd, self.result_cmd = self.build_apdl_scripts(inputfile)
        subprocess.run(self.meshdata_cmd)
        
        self.num_elem, self.num_node = self.count_mesh()
        self.centers, self.elemvol, self.elmnodes, self.node_coord = self.get_mesh_data()
    
        self.Ex     = Ex
        self.Ey     = Ey
        self.Gxy    = Gxy
        self.nu     = nu
        self.theta0 = np.deg2rad(theta0)
        
        self.volfrac = volfrac
        self.rmin    = rmin
        self.penal   = 3
        self.move    = np.concatenate((0.4*np.ones(self.num_elem),5./360*np.ones(self.num_elem)))
        
        self.sensitivity_filter = MeshIndependenceFilter(self.rmin, self.num_elem, self.centers)
        self.orientation_filter = OrientationRegularizationFilter(self.rmin, self.num_elem, self.centers)

        self.max_iter = 200
        self.rho_min  = 1e-3
        
        self.optim_setup()
        self.rho_hist   = []
        self.theta_hist = []
        self.comp_hist  = []
    
    def build_apdl_scripts(self, inputfile):
        with open(self.res_dir/'ansys_meshdata.txt', 'w') as f:
            f.write(f"RESUME,'{inputfile}','db','{TopOpt.mod_dir}',0,0\n")
            f.write(f"/CWD,'{self.res_dir}'\n")
            f.write(f"/FILENAME,{inputfile},1\n")
            f.write(f"/TITLE,{inputfile}\n")
            f.write(open(TopOpt.script_dir/'ansys_meshdata.txt').read())
            
        with open(self.res_dir/'ansys_solve.txt', 'w') as f:
            f.write(f"RESUME,'{inputfile}','db','{TopOpt.mod_dir}',0,0\n")
            f.write(f"/CWD,'{self.res_dir}'\n")
            f.write(f"/FILENAME,{inputfile},1\n")
            f.write(f"/TITLE,{inputfile}\n")
            f.write(open(TopOpt.script_dir/'ansys_solve.txt').read())
            
        meshdata_cmd = [TopOpt.ANSYS_path, '-b', '-i', self.res_dir/'ansys_meshdata.txt', '-o', self.res_dir/'meshdata.out']
        result_cmd = [TopOpt.ANSYS_path, '-b', '-i', self.res_dir/'ansys_solve.txt', '-o', self.res_dir/'solve.out', '-smp', '-np', str(TopOpt.np)]
            
        return meshdata_cmd, result_cmd
    
    def count_mesh(self):
        count = np.loadtxt(self.res_dir/'elements_nodes_counts.txt', dtype=int) # num_elm num_nodes
        return count[0], count[1]
    
    def get_mesh_data(self):
        centers    = np.loadtxt(self.res_dir/'elements_centers.txt')[:, 1:] # label x y z
        elmvol     = np.loadtxt(self.res_dir/'elements_volumn.txt')[:,1] # label elmvol
        elmnodes   = np.loadtxt(self.res_dir/'elements_nodes.txt', dtype=int) - 1 # n1 n2 n3 n4 ...
        node_coord = np.loadtxt(self.res_dir/'node_coordinates.txt') # x y z
        return centers, elmvol, elmnodes, node_coord
    
    def optim_setup(self):
        rho   = self.volfrac * np.ones(self.num_elem)
        theta = self.theta0 * np.ones(self.num_elem)
        self.x = np.concatenate((rho,theta))

        xmin = np.concatenate((self.rho_min*np.ones_like(rho), -np.pi*np.ones_like(theta)))
        xmax = np.concatenate((np.ones_like(rho), np.pi*np.ones_like(theta)))    
        self.mma = MMA(self.fea,self.sensitivities,self.constraint,self.dconstraint,xmin,xmax,self.move)
    
    def fea(self, x):
        rho, theta = np.split(x,2)
        
        # Generate file with material properties for each element
        Ex  = rho**self.penal * self.Ex
        Ey  = rho**self.penal * self.Ey
        Gxy = rho**self.penal * self.Gxy
        nu  = self.nu * np.ones(self.num_elem)
        material = np.array([Ex, Ey, Gxy, nu, np.rad2deg(theta)]).T
        np.savetxt(self.res_dir/'material.txt', material, fmt=' %-.7E', newline='\n')
        
        # Solve
        subprocess.run(self.result_cmd)
        energy = np.loadtxt(self.res_dir/'strain_energy.txt', dtype=float) # strain_energy
        c = 2*np.sum(energy)

        self.rho_hist.append(rho)
        self.theta_hist.append(theta)
        self.comp_hist.append(c)
    
        return c
    
    @abstractmethod
    def sensitivities(self, x):
        pass
    
    # sum(rho.v)/(volfrac.V) - 1 <= 0
    def constraint(self, x):
        rho, _ = np.split(x,2)
        return rho.dot(self.elemvol)/self.volfrac/np.sum(self.elemvol) - 1
    
    def dconstraint(self, x):
        return np.concatenate((self.elemvol/self.volfrac/np.sum(self.elemvol),np.zeros(self.num_elem)))
    
    def optim(self):
        t0 = time.time()
        for _ in range(self.max_iter):
            print("Starting iteration {:3d}...".format(self.mma.iter+1), end=' ')
            xnew = self.mma.iterate(self.x)
            print("compliance = {:.4f}".format(self.comp_hist[-1]))
            
            # Relative variation of the compliance moving average
            convergence = np.abs((sum(self.comp_hist[-5:])-sum(self.comp_hist[-10:-5]))/sum(self.comp_hist[-10:-5])) if self.mma.iter > 9 else 1
            if convergence < 1e-3: break

            rho, theta = np.split(xnew,2)
            theta = self.orientation_filter.filter(theta)
            self.x = np.concatenate((rho,theta))
            
        # Evaluating result from last iteration
        self.fea(self.x)
        
        self.time = time.time() - t0
        return rho, theta
    
class TopOpt2D(TopOpt):
    def sensitivities(self, x):
        rho, theta = np.split(x,2)
        
        # dc/drho
        energy = np.loadtxt(self.res_dir/'strain_energy.txt', dtype=float) # strain_energy
        uku = 2*energy/rho**self.penal # K: stiffness matrix with rho=1
        dcdrho = -self.penal * rho**(self.penal-1) * uku
        dcdrho = self.sensitivity_filter.filter(rho, dcdrho)
        
        # dc/dtheta
        u = np.loadtxt(self.res_dir/'nodal_solution_u.txt', dtype=float) # ux uy (uz)
        dcdt = np.zeros(self.num_elem)
        for i in range(self.num_elem):
            from .dkdt2d import dkdt2d
            dkdt = dkdt2d(self.Ex,self.Ey,self.nu,theta[i],self.elemvol[i])
            
            nodes = self.elmnodes[i,:]
            ue = [u[nodes[0],0], u[nodes[0],1], u[nodes[1],0], u[nodes[1],1], u[nodes[2],0], u[nodes[2],1], u[nodes[3],0], u[nodes[3],1]]
            ue = np.array(ue)
            
            dcdt[i] = -rho[i]**self.penal * ue.dot(dkdt.dot(ue))
            
        dcdt = self.sensitivity_filter.filter(rho, dcdt)
        
        return np.concatenate((dcdrho, dcdt))
    
class TopOpt3D(TopOpt):
    def sensitivities(self, x):
        rho, theta = np.split(x,2)
        
        # dc/drho
        energy = np.loadtxt(self.res_dir/'strain_energy.txt', dtype=float) # strain_energy
        uku = 2*energy/rho**self.penal # K: stiffness matrix with rho=1
        dcdrho = -self.penal * rho**(self.penal-1) * uku
        dcdrho = self.sensitivity_filter.filter(rho, dcdrho)
        
        # dc/dtheta
        u = np.loadtxt(self.res_dir/'nodal_solution_u.txt', dtype=float) # ux uy (uz)
        dcdt = np.zeros(self.num_elem)
        for i in range(self.num_elem):
            from .dkdt3d import dkdt3d
            dkdt = dkdt3d(self.Ex,self.Ey,self.nu,theta[i],self.elemvol[i])
            
            nodes = self.elmnodes[i,:]
            ue = [u[nodes[0],0], u[nodes[0],1], u[nodes[0],2], u[nodes[1],0], u[nodes[1],1], u[nodes[1],2], u[nodes[2],0], u[nodes[2],1], u[nodes[2],2], u[nodes[3],0], u[nodes[3],1], u[nodes[3],2], u[nodes[4],0], u[nodes[4],1], u[nodes[4],2], u[nodes[5],0], u[nodes[5],1], u[nodes[5],2], u[nodes[6],0], u[nodes[6],1], u[nodes[6],2], u[nodes[7],0], u[nodes[7],1], u[nodes[7],2]]
            ue = np.array(ue)
            
            dcdt[i] = -rho[i]**self.penal * ue.dot(dkdt.dot(ue))
            
        dcdt = self.sensitivity_filter.filter(rho, dcdt)

        return np.concatenate((dcdrho, dcdt))