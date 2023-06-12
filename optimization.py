import subprocess
import time

import numpy as np

from filters import MeshIndependenceFilter, OrientationRegularizationFilter, GaussianFilter
from mma import MMA

# https://github.com/pep-pig/Topology-optimization-of-structure-via-simp-method
"""
Element type: 4-node 2D quad, PLANE182 in a rectangular domain

Required APDL script files:
    ansys_meshdata.txt: gets mesh properties
    ansys_solve.txt: calls finite element analysis and stores results

TopOpt(inputfile, Ex, nu, volfrac, rmin, penal, theta0):
    inputfile: name of the model file (without .db)
    Ex, Ey, Gxy, nu: material properties
    volfrac: volume fraction constraint for the optimization
    rmin: radius of the filter (adjusts minimum feature size)
    theta0: initial orientation of the fibers, in degrees

Configure Ansys:
    load_paths(ANSYS_path, res_dir, mod_dir)
    set_processors(np)
Optimization function: optim()
"""
class TopOpt():
    def load_paths(ANSYS_path, res_dir, mod_dir):
        """
        ANSYS_path: MAPDL executable path
        res_dir: folder to store results
        mod_dir: folder with the .db file (geometry, mesh, constraints, loads)
        """
        TopOpt.ANSYS_path = ANSYS_path
        TopOpt.res_dir = res_dir
        TopOpt.mod_dir = mod_dir

    # Running on Shared Memory Parallel
    np = 2
    def set_processors(np):
        TopOpt.np = np
    
    def __init__(self, inputfile, Ex, Ey, Gxy, nu, volfrac, rmin, theta0):
        self.meshdata_cmd = [TopOpt.ANSYS_path, '-b', '-i', 'ansys_meshdata.txt', '-o', TopOpt.res_dir+'meshdata.out']
        self.result_cmd = [TopOpt.ANSYS_path, '-b', '-i', 'ansys_solve.txt', '-o', TopOpt.res_dir+'solve.out', '-smp', '-np', str(TopOpt.np)]
        
        TopOpt.write_pathfile(inputfile)
        subprocess.run(self.meshdata_cmd)
        
        self.num_elem, self.num_node = TopOpt.count_mesh()
        self.centers, self.elemvol, self.elmnodes, self.node_coord = TopOpt.get_mesh_data()
    
        self.Ex     = Ex
        self.Ey     = Ey
        self.Gxy    = Gxy
        self.nu     = nu
        self.theta0 = np.deg2rad(theta0)
        
        self.volfrac = volfrac
        self.rmin    = rmin
        self.penal   = 3
        self.move    = np.concatenate((0.4*np.ones(self.num_elem),0.01*np.ones(self.num_elem)))
        
        self.sensitivity_filter = MeshIndependenceFilter(self.rmin, self.num_elem, self.centers)
        self.orientation_filter = OrientationRegularizationFilter(self.rmin, self.num_elem, self.centers)

        self.max_iter = 200
        self.rho_min  = 1e-3
        
        self.optim_setup()
        self.rho_hist   = []
        self.theta_hist = []
        self.comp_hist  = []
    
    def write_pathfile(inputfile):
        with open(TopOpt.res_dir+'path.txt', 'w') as f:
            f.write(f"/CWD,'{TopOpt.res_dir}'\n")
            f.write(f"/FILENAME,{inputfile},1\n")
            f.write(f"/TITLE,{inputfile}\n")
            f.write(f"RESUME,'{inputfile}','db','{TopOpt.mod_dir}',0,0\n")
    
    def count_mesh():
        count = np.loadtxt(TopOpt.res_dir+'elements_nodes_counts.txt', dtype=int) # num_elm num_nodes
        return count[0], count[1]
    
    def get_mesh_data():
        centers    = np.loadtxt(TopOpt.res_dir+'elements_centers.txt')[:, 1:] # label x y z
        elmvol     = np.loadtxt(TopOpt.res_dir+'elements_volumn.txt')[:,1] # label elmvol
        elmnodes   = np.loadtxt(TopOpt.res_dir+'elements_nodes.txt', dtype=int) - 1 # n1 n2 n3 n4 ...
        node_coord = np.loadtxt(TopOpt.res_dir+'node_coordinates.txt') # x y z
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
        np.savetxt(TopOpt.res_dir+'material.txt', material, fmt=' %-.7E', newline='\n')
        
        # Solve
        subprocess.run(self.result_cmd)
        energy = np.loadtxt(TopOpt.res_dir+'strain_energy.txt', dtype=float) # strain_energy
        c = 2*np.sum(energy)

        self.rho_hist.append(rho)
        self.theta_hist.append(theta)
        self.comp_hist.append(c)
    
        return c
    
    def sensitivities(self, x):
        rho, theta = np.split(x,2)
        
        # dc/drho
        energy = np.loadtxt(TopOpt.res_dir+'strain_energy.txt', dtype=float) # strain_energy
        uku = 2*energy/rho**self.penal # K: stiffness matrix with rho=1
        dcdrho = -self.penal * rho**(self.penal-1) * uku
        dcdrho = self.sensitivity_filter.filter(rho, dcdrho)
        
        # dc/dtheta
        Ex = self.Ex
        Ey = self.Ey
        nuxy = self.nu
        
        u = np.loadtxt(TopOpt.res_dir+'nodal_solution_u.txt', dtype=float) # ux uy (uz)
        dcdt = np.zeros(self.num_elem)
        for i in range(self.num_elem):
            T = theta[i]
            dkdt = np.zeros((8,8))
            dkdt[0,0] = -((2*Ex**2*np.cos(T)*np.sin(T))/3 - (2*Ex*Ey*np.cos(T)*np.sin(T))/3 + (2*Ex*Ey*nuxy*np.cos(T)**2)/3 - (2*Ex*Ey*nuxy*np.sin(T)**2)/3)/(- Ey*nuxy**2 + Ex)
            dkdt[0,1] = -((Ex*Ey*np.cos(2*T))/4 - (Ex**2*np.cos(2*T))/4 + (Ex*Ey*nuxy*np.sin(2*T))/2)/(- Ey*nuxy**2 + Ex)
            dkdt[0,2] = ((2*Ex**2*np.cos(T)*np.sin(T))/3 - (2*Ex*Ey*np.cos(T)*np.sin(T))/3 + (2*Ex*Ey*nuxy*np.cos(T)**2)/3 - (2*Ex*Ey*nuxy*np.sin(T)**2)/3)/(- Ey*nuxy**2 + Ex)
            dkdt[0,3] = -(Ex*Ey*np.cos(2*T) - Ex**2*np.cos(2*T) + 2*Ex*Ey*nuxy*np.sin(2*T))/(4*(- Ey*nuxy**2 + Ex))
            dkdt[0,4] = ((Ex**2*np.cos(T)*np.sin(T))/3 - (Ex*Ey*np.cos(T)*np.sin(T))/3 + (Ex*Ey*nuxy*np.cos(T)**2)/3 - (Ex*Ey*nuxy*np.sin(T)**2)/3)/(- Ey*nuxy**2 + Ex)
            dkdt[0,5] = (Ex*Ey*np.cos(2*T) - Ex**2*np.cos(2*T) + 2*Ex*Ey*nuxy*np.sin(2*T))/(4*(- Ey*nuxy**2 + Ex))
            dkdt[0,6] = -((Ex**2*np.cos(T)*np.sin(T))/3 - (Ex*Ey*np.cos(T)*np.sin(T))/3 + (Ex*Ey*nuxy*np.cos(T)**2)/3 - (Ex*Ey*nuxy*np.sin(T)**2)/3)/(- Ey*nuxy**2 + Ex)
            dkdt[0,7] = ((Ex*Ey*np.cos(2*T))/4 - (Ex**2*np.cos(2*T))/4 + (Ex*Ey*nuxy*np.sin(2*T))/2)/(- Ey*nuxy**2 + Ex)
            dkdt[1,1] = (2*Ex**2*np.cos(T)*np.sin(T) - 2*Ex*Ey*np.cos(T)*np.sin(T) + 2*Ex*Ey*nuxy*np.cos(T)**2 - 2*Ex*Ey*nuxy*np.sin(T)**2)/(- 3*Ey*nuxy**2 + 3*Ex)
            dkdt[1,2] = (Ex*Ey*np.cos(2*T) - Ex**2*np.cos(2*T) + 2*Ex*Ey*nuxy*np.sin(2*T))/(4*(- Ey*nuxy**2 + Ex))
            dkdt[1,3] = (4*((Ex**2*np.cos(T)*np.sin(T))/4 - (Ex*Ey*np.cos(T)*np.sin(T))/4 + (Ex*Ey*nuxy*np.cos(T)**2)/4 - (Ex*Ey*nuxy*np.sin(T)**2)/4))/(3*(- Ey*nuxy**2 + Ex))
            dkdt[1,4] = (Ex*Ey*np.cos(2*T) - Ex**2*np.cos(2*T) + 2*Ex*Ey*nuxy*np.sin(2*T))/(4*(- Ey*nuxy**2 + Ex))
            dkdt[1,5] = -(4*((Ex**2*np.cos(T)*np.sin(T))/4 - (Ex*Ey*np.cos(T)*np.sin(T))/4 + (Ex*Ey*nuxy*np.cos(T)**2)/4 - (Ex*Ey*nuxy*np.sin(T)**2)/4))/(3*(- Ey*nuxy**2 + Ex))
            dkdt[1,6] = -((Ex*Ey*np.cos(2*T))/4 - (Ex**2*np.cos(2*T))/4 + (Ex*Ey*nuxy*np.sin(2*T))/2)/(- Ey*nuxy**2 + Ex)
            dkdt[1,7] = -(4*Ex**2*np.cos(T)*np.sin(T) - 4*Ex*Ey*np.cos(T)*np.sin(T) + 4*Ex*Ey*nuxy*np.cos(T)**2 - 4*Ex*Ey*nuxy*np.sin(T)**2)/(- 6*Ey*nuxy**2 + 6*Ex)
            dkdt[2,2] = -((2*Ex**2*np.cos(T)*np.sin(T))/3 - (2*Ex*Ey*np.cos(T)*np.sin(T))/3 + (2*Ex*Ey*nuxy*np.cos(T)**2)/3 - (2*Ex*Ey*nuxy*np.sin(T)**2)/3)/(- Ey*nuxy**2 + Ex)
            dkdt[2,3] = ((Ex*Ey*np.cos(2*T))/4 - (Ex**2*np.cos(2*T))/4 + (Ex*Ey*nuxy*np.sin(2*T))/2)/(- Ey*nuxy**2 + Ex)
            dkdt[2,4] = -((Ex**2*np.cos(T)*np.sin(T))/3 - (Ex*Ey*np.cos(T)*np.sin(T))/3 + (Ex*Ey*nuxy*np.cos(T)**2)/3 - (Ex*Ey*nuxy*np.sin(T)**2)/3)/(- Ey*nuxy**2 + Ex)
            dkdt[2,5] = -((Ex*Ey*np.cos(2*T))/4 - (Ex**2*np.cos(2*T))/4 + (Ex*Ey*nuxy*np.sin(2*T))/2)/(- Ey*nuxy**2 + Ex)
            dkdt[2,6] = ((Ex**2*np.cos(T)*np.sin(T))/3 - (Ex*Ey*np.cos(T)*np.sin(T))/3 + (Ex*Ey*nuxy*np.cos(T)**2)/3 - (Ex*Ey*nuxy*np.sin(T)**2)/3)/(- Ey*nuxy**2 + Ex)
            dkdt[2,7] = -(Ex*Ey*np.cos(2*T) - Ex**2*np.cos(2*T) + 2*Ex*Ey*nuxy*np.sin(2*T))/(4*(- Ey*nuxy**2 + Ex))
            dkdt[3,3] = (2*Ex**2*np.cos(T)*np.sin(T) - 2*Ex*Ey*np.cos(T)*np.sin(T) + 2*Ex*Ey*nuxy*np.cos(T)**2 - 2*Ex*Ey*nuxy*np.sin(T)**2)/(- 3*Ey*nuxy**2 + 3*Ex)
            dkdt[3,4] = ((Ex*Ey*np.cos(2*T))/4 - (Ex**2*np.cos(2*T))/4 + (Ex*Ey*nuxy*np.sin(2*T))/2)/(- Ey*nuxy**2 + Ex)
            dkdt[3,5] = -(4*Ex**2*np.cos(T)*np.sin(T) - 4*Ex*Ey*np.cos(T)*np.sin(T) + 4*Ex*Ey*nuxy*np.cos(T)**2 - 4*Ex*Ey*nuxy*np.sin(T)**2)/(- 6*Ey*nuxy**2 + 6*Ex)
            dkdt[3,6] = -(Ex*Ey*np.cos(2*T) - Ex**2*np.cos(2*T) + 2*Ex*Ey*nuxy*np.sin(2*T))/(4*(- Ey*nuxy**2 + Ex))
            dkdt[3,7] = -(4*((Ex**2*np.cos(T)*np.sin(T))/4 - (Ex*Ey*np.cos(T)*np.sin(T))/4 + (Ex*Ey*nuxy*np.cos(T)**2)/4 - (Ex*Ey*nuxy*np.sin(T)**2)/4))/(3*(- Ey*nuxy**2 + Ex))
            dkdt[4,4] = -((2*Ex**2*np.cos(T)*np.sin(T))/3 - (2*Ex*Ey*np.cos(T)*np.sin(T))/3 + (2*Ex*Ey*nuxy*np.cos(T)**2)/3 - (2*Ex*Ey*nuxy*np.sin(T)**2)/3)/(- Ey*nuxy**2 + Ex)
            dkdt[4,5] = -((Ex*Ey*np.cos(2*T))/4 - (Ex**2*np.cos(2*T))/4 + (Ex*Ey*nuxy*np.sin(2*T))/2)/(- Ey*nuxy**2 + Ex)
            dkdt[4,6] = ((2*Ex**2*np.cos(T)*np.sin(T))/3 - (2*Ex*Ey*np.cos(T)*np.sin(T))/3 + (2*Ex*Ey*nuxy*np.cos(T)**2)/3 - (2*Ex*Ey*nuxy*np.sin(T)**2)/3)/(- Ey*nuxy**2 + Ex)
            dkdt[4,7] = -(Ex*Ey*np.cos(2*T) - Ex**2*np.cos(2*T) + 2*Ex*Ey*nuxy*np.sin(2*T))/(4*(- Ey*nuxy**2 + Ex))
            dkdt[5,5] = (2*Ex**2*np.cos(T)*np.sin(T) - 2*Ex*Ey*np.cos(T)*np.sin(T) + 2*Ex*Ey*nuxy*np.cos(T)**2 - 2*Ex*Ey*nuxy*np.sin(T)**2)/(- 3*Ey*nuxy**2 + 3*Ex)
            dkdt[5,6] = (Ex*Ey*np.cos(2*T) - Ex**2*np.cos(2*T) + 2*Ex*Ey*nuxy*np.sin(2*T))/(4*(- Ey*nuxy**2 + Ex))
            dkdt[5,7] = (4*((Ex**2*np.cos(T)*np.sin(T))/4 - (Ex*Ey*np.cos(T)*np.sin(T))/4 + (Ex*Ey*nuxy*np.cos(T)**2)/4 - (Ex*Ey*nuxy*np.sin(T)**2)/4))/(3*(- Ey*nuxy**2 + Ex))
            dkdt[6,6] = -((2*Ex**2*np.cos(T)*np.sin(T))/3 - (2*Ex*Ey*np.cos(T)*np.sin(T))/3 + (2*Ex*Ey*nuxy*np.cos(T)**2)/3 - (2*Ex*Ey*nuxy*np.sin(T)**2)/3)/(- Ey*nuxy**2 + Ex)
            dkdt[6,7] = ((Ex*Ey*np.cos(2*T))/4 - (Ex**2*np.cos(2*T))/4 + (Ex*Ey*nuxy*np.sin(2*T))/2)/(- Ey*nuxy**2 + Ex)
            dkdt[7,7] = (2*Ex**2*np.cos(T)*np.sin(T) - 2*Ex*Ey*np.cos(T)*np.sin(T) + 2*Ex*Ey*nuxy*np.cos(T)**2 - 2*Ex*Ey*nuxy*np.sin(T)**2)/(- 3*Ey*nuxy**2 + 3*Ex)
            dkdt = dkdt + dkdt.T - np.diag(dkdt.diagonal()) # symmetric matrix
            dkdt = dkdt * self.elemvol[i]
            
            nodes = self.elmnodes[i,:]
            ue = [u[nodes[0],0], u[nodes[0],1], u[nodes[1],0], u[nodes[1],1], u[nodes[2],0], u[nodes[2],1], u[nodes[3],0], u[nodes[3],1]]
            ue = np.array(ue)
            
            dcdt[i] = -rho[i]**self.penal * ue.dot(dkdt.dot(ue))
        
        return np.concatenate((dcdrho, dcdt))
    
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
            
        # Post-processing last iteration
#         rho, theta = np.split(self.x,2)
#         theta = GaussianFilter(self.num_elem, self.centers).filter(theta)
#         self.x = np.concatenate((rho,theta))
        self.fea(self.x)
        self.time = time.time() - t0

        return rho, theta
