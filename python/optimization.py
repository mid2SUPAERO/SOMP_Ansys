import subprocess
import time
import os, glob
import numpy as np

from .dstiffness import dk2d, dk3d
from .filters import DensityFilter, OrientationFilter
from .mma import MMA

# Starting point:
# https://github.com/pep-pig/Topology-optimization-of-structure-via-simp-method
"""
Required APDL script files:
    ansys_meshdata_2d.txt: gets mesh properties for 2D optimization
    ansys_meshdata_3d.txt: gets mesh properties for 3D optimization
    ansys_solve.txt: calls finite element analysis and stores results

TopOpt2D: 4-node 2D quad, PLANE182 in a rectangular domain, with KEYOPT(3) = 3 plane stress with thk
TopOpt3D: 8-node 3D hex, SOLID185 in a cuboid domain

TopOpt(inputfile, Ex, Ey, nuxy, nuyz, Gxy, volfrac, r_rho, r_theta, theta0, alpha0, max_iter, dim, jobname, echo):
    inputfile: name of the model file (without .db)
    Ex, Ey, nuxy, nuyz, Gxy: material properties
    volfrac: volume fraction constraint for the optimization
    r_rho: radius of the density filter (adjusts minimum feature size)
    r_theta: radius of the orientation filter (adjusts fiber curvature)
    theta0: initial orientation (around z) of the fibers, in degrees. Default: random distribution
    alpha0: initial orientation (around x) of the fibers, in degrees. Default: random distribution
    max_iter: number of iterations
    dim: optimization type, '2D', '3D_layer' or '3D_free'
    jobname: subfolder of TopOpt.res_dir to store results for this optim. Default: stores results directly on TopOpt.res_dir
    echo: boolean, print status at each iteration?

Configure Ansys: Same configuration for all TopOpt objects
    set_paths(ANSYS_path, script_dir, res_dir, mod_dir): paths as pathlib.Path

Configure optimization:
    set_solid_elem(solid_elem): list of elements whose densities will be fixed on 1. Element indexing starting at 0
    
Optimization function: optim()

Design evaluation:
    mass(rho): final design mass
    disp_max(): maximum nodal displacement
    CO2_footprint(rho, CO2mat, CO2veh): final design carbon footprint, considering material production and use in a vehicle
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
    
    def __init__(self, inputfile, Ex, Ey, nuxy, nuyz, Gxy, volfrac, r_rho=0, r_theta=0, theta0=None, alpha0=None, max_iter=200, dim='3D_layer', jobname=None, echo=True):
        self.dim  = dim
        self.echo = echo

        # dkdt, dkda = dk(Ex,Ey,nuxy,nuyz,Gxy,theta,alpha,elmvol)
        if dim == '2D':
            self.dk = dk2d
        elif dim == '3D_layer':
            self.dk = dk3d
        elif dim == '3D_free':
            self.dk = dk3d
        
        self.jobname = jobname
        if jobname is None:
            self.res_dir = TopOpt.res_dir
        else:
            self.res_dir = TopOpt.res_dir / jobname
        self.res_dir.mkdir(parents=True, exist_ok=True)
        
        self.meshdata_cmd, self.result_cmd = self.build_apdl_scripts(inputfile)
        self.num_elem, self.num_node, self.centers, self.elemvol, self.elmnodes, self.node_coord = self.get_mesh_data()
    
        self.Ex   = Ex
        self.Ey   = Ey
        self.nuxy = nuxy
        self.nuyz = nuyz
        self.Gxy  = Gxy
        
        if theta0 is None:
            # Random numbers between -pi/2 and pi/2
            self.theta0 = np.pi * np.random.random(self.num_elem) - np.pi/2
        else:
            self.theta0 = np.deg2rad(theta0)
            
        if alpha0 is None:
            # Random numbers between -pi and pi
            self.alpha0 = 2*np.pi * np.random.random(self.num_elem) - np.pi
        else:
            self.alpha0 = np.deg2rad(alpha0)
        
        self.max_iter   = max_iter
        self.rho_min    = 1e-3
        self.penal      = 3
        self.volfrac    = volfrac
        self.r_rho      = r_rho
        self.r_theta    = r_theta
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
        self.x = rho
        
        xmin = self.rho_min*np.ones_like(rho)
        xmax = np.ones_like(rho)
        
        # add theta variable
        if self.dim == '2D' or self.dim == '3D_layer' or self.dim == '3D_free':
            theta  = self.theta0 * np.ones(self.num_elem)
            self.x = np.concatenate((rho,theta))
            xmin = np.concatenate((xmin, -np.pi*np.ones_like(theta)))
            xmax = np.concatenate((xmax, np.pi*np.ones_like(theta)))
        
        # add alpha variable
        if self.dim == '3D_free':
            alpha = self.alpha0 * np.ones(self.num_elem)
            self.x = np.concatenate((self.x,alpha))
            xmin = np.concatenate((xmin, -2*np.pi*np.ones_like(alpha)))
            xmax = np.concatenate((xmax, 2*np.pi*np.ones_like(alpha)))
        
        mma = MMA(self.fea,self.sensitivities,self.constraint,self.dconstraint,xmin,xmax)
        
        self.rho_hist   = []
        self.theta_hist = []
        self.alpha_hist = []
        self.comp_hist  = []
        
        return mma
    
    def fea(self, x):
        t0 = time.time()
        
        if self.dim == 'SIMP':
            rho = x.copy()
            theta = np.zeros_like(rho)
            alpha = np.zeros_like(rho)
        elif self.dim == '2D' or self.dim == '3D_layer':
            rho, theta = np.split(x,2)
            alpha = np.zeros_like(theta)
        elif self.dim == '3D_free':
            rho, theta, alpha = np.split(x,3)
            
        theta, alpha = self.orientation_filter.filter(rho,theta,alpha)
        
        # Generate file with material properties for each element
        Ex   = rho**self.penal * self.Ex
        Ey   = rho**self.penal * self.Ey
        nuxy = self.nuxy * np.ones(self.num_elem)
        nuyz = self.nuyz * np.ones(self.num_elem)
        Gxy  = rho**self.penal * self.Gxy
        Gyz  = Ey/(2*(1+nuyz))
        material = np.array([Ex, Ey, nuxy, nuyz, Gxy, Gyz, np.rad2deg(theta), np.deg2rad(alpha)]).T
        np.savetxt(self.res_dir/'material.txt', material, fmt=' %-.7E', newline='\n')
        
        # Solve
        subprocess.run(self.result_cmd)
        energy = np.loadtxt(self.res_dir/'strain_energy.txt', dtype=float) # strain_energy
        c = 2*np.sum(energy)

        # Save history
        self.rho_hist.append(rho)
        self.theta_hist.append(theta)
        self.alpha_hist.append(alpha)
        self.comp_hist.append(c)

        if self.echo: print('compliance = {:10.4f}'.format(c))
        self.fea_time += time.time() - t0
        return c
    
    def sensitivities(self, x):
        t0 = time.time()
        
        if self.dim == 'SIMP':
            rho = x.copy()
            theta = np.zeros_like(rho)
            alpha = np.zeros_like(rho)
        elif self.dim == '2D' or self.dim == '3D_layer':
            rho, theta = np.split(x,2)
            alpha = np.zeros_like(theta)
        elif self.dim == '3D_free':
            rho, theta, alpha = np.split(x,3)
        
        # dc/drho
        energy = np.loadtxt(self.res_dir/'strain_energy.txt', dtype=float) # strain_energy
        uku    = 2*energy/rho**self.penal # K: stiffness matrix with rho=1
        dcdrho = -self.penal * rho**(self.penal-1) * uku
        dcdrho = self.density_filter.filter(rho, dcdrho)
        
        if self.dim == 'SIMP':
            self.deriv_time += time.time() - t0
            return dcdrho
        
        # dc/dtheta and dc/dalpha
        u = np.loadtxt(self.res_dir/'nodal_solution_u.txt', dtype=float) # ux uy uz
        if self.dim == '2D':
            u = u[:,[0,1]] # drop z dof
            
        dcdt = np.zeros(self.num_elem)
        dcda = np.zeros(self.num_elem)
        for i in range(self.num_elem):
            dkdt, dkda = self.dk(self.Ex,self.Ey,self.nuxy,self.nuyz,self.Gxy,theta[i],alpha[i],self.elemvol[i])
            ue = u[self.elmnodes[i,:],:].flatten()
            dcdt[i] = -rho[i]**self.penal * ue.dot(dkdt.dot(ue))
            dcda[i] = -rho[i]**self.penal * ue.dot(dkda.dot(ue))
        
        # concatenate all sensitivities
        if self.dim == '2D' or self.dim == '3D_layer':
            dc = np.concatenate((dcdrho,dcdt))
        elif self.dim == '3D_free':
            dc = np.concatenate((dcdrho,dcdt,dcda))
        
        self.deriv_time += time.time() - t0
        return dc
    
    # sum(rho.v)/(volfrac.V) - 1 <= 0
    def constraint(self, x):
        rho = x[:self.num_elem]
        return rho.dot(self.elemvol)/self.volfrac/np.sum(self.elemvol) - 1
    
    def dconstraint(self, x):
        dcons = np.zeros_like(x)
        dcons[:self.num_elem] = self.elemvol/self.volfrac/np.sum(self.elemvol)
        return dcons
    
    def set_solid_elem(self, solid_elem):
        self.solid_elem    = solid_elem
        self.x[solid_elem] = 1
        
    def optim(self):
        t0 = time.time()
        for _ in range(self.max_iter):
            if self.echo: print('Iteration {:3d}... '.format(self.mma.iter), end=' ')
            xnew = self.mma.iterate(self.x)
                
            if self.dim == 'SIMP':
                rho = xnew.copy()
                theta = np.zeros_like(rho)
                alpha = np.zeros_like(rho)
            elif self.dim == '2D' or self.dim == '3D_layer':
                rho, theta = np.split(xnew,2)
                alpha = np.zeros_like(theta)
            elif self.dim == '3D_free':
                rho, theta, alpha = np.split(xnew,3)
            
            rho[self.solid_elem] = 1
            theta, alpha = self.orientation_filter.filter(rho,theta,alpha)
            if self.dim == 'SIMP':
                self.x = rho
            elif self.dim == '2D' or self.dim == '3D_layer':
                self.x = np.concatenate((rho,theta))
            elif self.dim == '3D_free':
                self.x = np.concatenate((rho,theta,alpha))
                
        # Evaluate result from last iteration
        if self.echo: print('Iteration {:3d}... '.format(self.mma.iter), end=' ')
        self.fea(self.x)
        
        self.clear_files()
        self.time = time.time() - t0
        return self.x
    
    # clear temporary Ansys files
    def clear_files(self):
        for filename in glob.glob('cleanup*'): os.remove(filename)
        for filename in glob.glob(self.title + '.*'): os.remove(filename)
    
    def mass(self, rho):
        x = self.rho_hist[-1]
        mass = rho * x.dot(self.elemvol)
        return mass
    
    def disp_max(self):
        u = np.loadtxt(self.res_dir/'nodal_solution_u.txt', dtype=float) # ux uy uz
        u = np.linalg.norm(u, axis=1)
        return np.amax(u)
    
    def CO2_footprint(self, rho, CO2mat, CO2veh):
        """
        rho: density
        CO2mat: mass CO2 emmited per mass material (material production)
        CO2veh: mass CO2 emitted per mass material during life (use in a vehicle)
                = mass fuel per mass transported per lifetime * service life * mass CO2 emmited per mass fuel
        """
        x = self.rho_hist[-1]
        mass = rho * x.dot(self.elemvol)
        return mass * (CO2mat + CO2veh)
