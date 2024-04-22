import subprocess
import time
import os, glob
from pathlib import Path
import numpy as np
import json, jsonpickle

from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

from .dstiffness import dk2d, dk3d
from .filters import DensityFilter, OrientationFilter
from .mma import MMA

# Starting point: https://github.com/pep-pig/Topology-optimization-of-structure-via-simp-method
class TopOpt():
    def rule_mixtures(*, fiber, matrix, Vfiber):
        rhofiber  = fiber['rho']
        Efiber    = fiber['E']
        vfiber    = fiber['v']
        CO2fiber  = fiber['CO2']

        rhomatrix = matrix['rho']
        Ematrix   = matrix['E']
        vmatrix   = matrix['v']
        CO2matrix = matrix['CO2']

        Vmatrix = 1-Vfiber

        Gfiber  = Efiber/(2*(1+vfiber))
        Gmatrix = Ematrix/(2*(1+vmatrix))

        Ex   = Efiber*Vfiber + Ematrix*Vmatrix
        Ey   = Efiber*Ematrix / (Efiber*Vmatrix + Ematrix*Vfiber)
        nuxy = vfiber*Vfiber + vmatrix*Vmatrix
        nuyz = nuxy * (1-nuxy*Ey/Ex)/(1-nuxy)
        Gxy  = Gfiber*Gmatrix / (Gfiber*Vmatrix + Gmatrix*Vfiber)
        
        rho    = rhofiber*Vfiber + rhomatrix*Vmatrix
        CO2mat = (rhofiber*Vfiber*CO2fiber + rhomatrix*Vmatrix*CO2matrix)/rho

        return Ex, Ey, nuxy, nuyz, Gxy, rho, CO2mat

    # Example of Python FEA function
    # https://www.topopt.mek.dtu.dk/apps-and-software/topology-optimization-codes-written-in-python
    def mbb2d(*, nelx, nely, elsize, elthk, Ex, Ey, nuxy, Gxy, force):
        ndof = 2*(nelx+1)*(nely+1)
        edofMat=np.zeros((nelx*nely,8),dtype=int)
        for elx in range(nelx):
            for ely in range(nely):
                el = ely+elx*nely
                n1=(nely+1)*elx+ely
                n2=(nely+1)*(elx+1)+ely
                edofMat[el,:]=np.array([2*n1+2, 2*n1+3, 2*n2+2, 2*n2+3,2*n2, 2*n2+1, 2*n1, 2*n1+1])
        # Construct the index pointers for the coo format
        iK = np.kron(edofMat,np.ones((8,1))).flatten()
        jK = np.kron(edofMat,np.ones((1,8))).flatten()

        # BC's and support
        dofs=np.arange(2*(nelx+1)*(nely+1))
        fixed=np.union1d(dofs[0:2*(nely+1):2],np.array([2*(nelx+1)*(nely+1)-1]))
        free=np.setdiff1d(dofs,fixed)

        # Solution and RHS vectors
        f=np.zeros((ndof,1))
        u=np.zeros((ndof,1))

        # Set load
        f[1,0]=force

        def fea_fun(x, penal):
            # Setup and solve FE problem
            rho   = x[:nelx*nely]
            theta = x[nelx*nely:]

            C = np.zeros((nelx*nely,3,3))
            C[:,0,0] = rho**penal * Ex**2/(Ex - Ey*nuxy**2)
            C[:,0,1] = rho**penal * (Ex*Ey*nuxy)/(Ex - Ey*nuxy**2)
            C[:,0,2] = 0
            C[:,1,1] = rho**penal * (Ex*Ey)/(Ex - Ey*nuxy**2)
            C[:,1,2] = 0
            C[:,2,2] = rho**penal * Gxy
            C[:,1,0] = C[:,0,1]
            C[:,2,0] = C[:,0,2]
            C[:,2,1] = C[:,1,2]

            c  = np.cos(theta)
            s  = np.sin(theta)
            Tt = np.zeros((nelx*nely,6,6))
            Tt[:,0,0] = c**2
            Tt[:,0,1] = s**2
            Tt[:,0,5] = -2*c*s
            Tt[:,1,0] = s**2
            Tt[:,1,1] = c**2
            Tt[:,1,5] = 2*c*s
            Tt[:,2,2] = 1
            Tt[:,3,3] = c
            Tt[:,3,4] = s
            Tt[:,4,3] = -s
            Tt[:,4,4] = c
            Tt[:,5,0] = c*s
            Tt[:,5,1] = -c*s
            Tt[:,5,5] = c**2 - s**2
            Tt = Tt[:,[0,1,5],:][:,:,[0,1,5]]

            points  = [-1/np.sqrt(3), 1/np.sqrt(3)]
            weights = [1, 1]
            sK = np.zeros((nelx*nely,8,8))
            for xi, wi in zip(points, weights):
                for xj, wj in zip(points, weights):
                        B = np.zeros((3,8))
                        B[0][0] = xj/4 - 1/4
                        B[0][2] = 1/4 - xj/4
                        B[0][4] = xj/4 + 1/4
                        B[0][6] = - xj/4 - 1/4
                        B[1][1] = xi/4 - 1/4
                        B[1][3] = - xi/4 - 1/4
                        B[1][5] = xi/4 + 1/4
                        B[1][7] = 1/4 - xi/4
                        B[2][0] = xi/4 - 1/4
                        B[2][1] = xj/4 - 1/4
                        B[2][2] = - xi/4 - 1/4
                        B[2][3] = 1/4 - xj/4
                        B[2][4] = xi/4 + 1/4
                        B[2][5] = xj/4 + 1/4
                        B[2][6] = 1/4 - xi/4
                        B[2][7] = - xj/4 - 1/4
                        sK += wi * wj * B.T @ Tt @ C @ Tt.transpose((0,2,1)) @ B
            sK *= elthk
            KE = sK.copy()
            sK = sK.reshape((nelx*nely,64)).T
            sK=sK.flatten(order='F')
            K = coo_matrix((sK,(iK,jK)),shape=(ndof,ndof)).tocsc()
            # Remove constrained dofs from matrix
            K = K[free,:][:,free]
            # Solve system
            u[free,0]=spsolve(K,f[free,0])
            energy= 0.5*((u[edofMat].reshape(nelx*nely,8) @ KE)[np.arange(nelx*nely),np.arange(nelx*nely)] * u[edofMat].reshape(nelx*nely,8)).sum(1)
            c = 2*energy.sum()

            u_reshape = u.reshape(((nelx+1)*(nely+1),2))
            u_reshape = np.hstack((u_reshape, np.zeros(((nelx+1)*(nely+1),1))))

            return c, u_reshape, energy
        
        def meshdata_fun():
            num_elem = nelx*nely
            num_node = (nelx+1)*(nely+1)
            centers  = np.zeros((num_elem,3))
            elmnodes = np.zeros((num_elem,4)).astype(int)
            for elx in range(nelx):
                for ely in range(nely):
                    el = ely+elx*nely
                    n1=(nely+1)*elx+ely
                    n2=(nely+1)*(elx+1)+ely
                    centers[el,[0,1]] = elsize*np.array([elx+0.5,nely-1-ely+0.5])
                    elmnodes[el,:] = np.array([n1+1,n2+1,n2,n1]).astype(int)
            elemvol = elsize**2 * elthk * np.ones(num_elem)
            node_coord = np.zeros((num_node,3))
            for nx in range(nelx+1):
                for ny in range(nely+1):
                    nn = ny+nx*(nely+1)
                    node_coord[nn,[0,1]] = elsize*np.array([nx,nely-ny])

            return num_elem, num_node, centers, elemvol, elmnodes, node_coord

        return fea_fun, meshdata_fun

    @staticmethod
    def set_paths(ANSYS_path, res_dir, mod_dir):
        TopOpt.ANSYS_path = ANSYS_path # MAPDL executable path
        TopOpt.res_dir    = res_dir    # folder to store results
        TopOpt.mod_dir    = mod_dir    # folder with the .db file (geometry, mesh, constraints, loads)
        TopOpt.script_dir = Path(os.path.abspath(os.path.dirname(__file__)))
    
    def __init__(self, *, inputfiles=None, dim='3D_layer', jobname=None, echo=True, fea_fun=None, meshdata_fun=None):
        try: TopOpt.ANSYS_path
        except AttributeError: raise Exception('TopOpt object creation failed. Define paths with TopOpt.set_paths()') from None

        if inputfiles is None and (fea_fun is None or meshdata_fun is None):
            raise Exception('Arguments inputfiles or fea_fun must be defined')
        
        self.dim  = dim
        self.echo = echo

        self.inputfiles   = inputfiles
        self.fea_fun      = fea_fun
        self.meshdata_fun = meshdata_fun
        
        if inputfiles is not None: # use Ansys
            # define load cases
            if isinstance(inputfiles,str): inputfiles = (inputfiles,)
            self.load_cases     = len(inputfiles)
            self.comp_max_order = 8               # using 8-norm as a differentiable max function of all compliances
            
            # create results folders
            self.jobname = jobname
            self.res_dir = []
            for lc, inputfile in enumerate(inputfiles):
                if jobname is None:
                    self.res_dir.append(TopOpt.res_dir / ('load_case_' + str(lc+1)))
                else:
                    self.res_dir.append(TopOpt.res_dir / jobname / ('load_case_' + str(lc+1)))
                self.res_dir[lc].mkdir(parents=True, exist_ok=True)
            self.res_root = self.res_dir[0].parent
            
            # get model properties
            self.inputfiles = inputfiles
            self.meshdata_cmd, self.initial_orientations_cmd, self.result_cmd = self.__build_apdl_scripts()
            self.num_elem, self.num_node, self.centers, self.elemvol, self.elmnodes, self.node_coord = self.__get_mesh_data()
        else:
            self.load_cases     = 1               # only supports single load case for Python FEA function
            self.comp_max_order = 8               # using 8-norm as a differentiable max function of all compliances

            self.jobname = jobname
            self.title   = 'file' if self.jobname is None else self.jobname
            self.res_dir = []
            for lc in range(self.load_cases):
                if jobname is None:
                    self.res_dir.append(TopOpt.res_dir / ('load_case_' + str(lc+1)))
                else:
                    self.res_dir.append(TopOpt.res_dir / jobname / ('load_case_' + str(lc+1)))
                self.res_dir[lc].mkdir(parents=True, exist_ok=True)
            self.res_root = self.res_dir[0].parent

            self.fea_fun = fea_fun
            self.num_elem, self.num_node, self.centers, self.elemvol, self.elmnodes, self.node_coord = meshdata_fun()
        
        # initial setup with default parameters
        self.set_material()
        self.set_volfrac()
        self.set_filters()
        self.set_solid_elem()
        self.set_initial_conditions()
        self.set_optim_options()

        self.__set_layers()
        
    def set_material(self, *, Ex=1, Ey=1, nuxy=0.3, nuyz=0.3, Gxy=1/(2*(1+0.3))):
        self.Ex   = Ex
        self.Ey   = Ey
        self.nuxy = nuxy
        self.nuyz = nuyz
        self.Gxy  = Gxy
        
        # update derivative function (dependent on material)
        self.dk = self.__get_dk()
        
    def set_volfrac(self, volfrac=0.3):
        self.rho_min = 1e-3
        self.volfrac = volfrac
        
    def set_filters(self, *, r_rho=0, r_theta=0):
        self.r_rho          = r_rho
        self.density_filter = DensityFilter(r_rho, self.centers)
        
        self.r_theta            = r_theta
        self.orientation_filter = OrientationFilter(r_theta, self.centers)
        
    def set_solid_elem(self, solid_elem=[]):
        self.solid_elem = solid_elem
        
    def __set_layers(self):
        # slice domain in layers normal to print_direction
        layer_thk = np.mean(np.cbrt(self.elemvol)) # thickness: approximately mean element edge length
        elm_height = self.centers[:,2]
        elm_layer = (elm_height/layer_thk).astype(int)
        
        layers = [[] for _ in range(np.amax(elm_layer)+1)]
        for i in range(self.num_elem):
            layers[elm_layer[i]].append(i)

        self.layers    = layers
        self.layer_thk = layer_thk
        
    def set_initial_conditions(self, angle_type='random', **kwargs):
        # initial angles are given
        if angle_type == 'fix':
            theta0 = kwargs.pop('theta0')
            if self.dim == '3D_free':
                alpha0 = kwargs.pop('alpha0')
            else:
                alpha0 = kwargs.get('alpha0',0)

            self.theta0 = np.deg2rad(theta0) * np.ones(self.num_elem)
            self.alpha0 = np.deg2rad(alpha0) * np.ones(self.num_elem)
        
        # orientations distributed around the given values
        elif angle_type == 'noise':
            theta0 = kwargs.pop('theta0')
            if self.dim == '3D_free':
                alpha0 = kwargs.pop('alpha0')
            else:
                alpha0 = kwargs.get('alpha0',0)

            self.theta0 = np.random.default_rng().normal(np.deg2rad(theta0), np.pi/10, self.num_elem)
            self.alpha0 = np.random.default_rng().normal(np.deg2rad(alpha0), np.pi/10, self.num_elem)
        
        # random numbers between -pi/2 and pi/2
        elif angle_type == 'random':
            self.theta0 = np.random.default_rng().uniform(-np.pi/2, np.pi/2, self.num_elem)
            self.alpha0 = np.random.default_rng().uniform(-np.pi/2, np.pi/2, self.num_elem)
        
        # inital orientations are the principal directions for an isotropic base case
        elif angle_type == 'principal':
            if self.echo: print('Calculating initial orientations...')
            subprocess.run(self.initial_orientations_cmd)
            angles = np.loadtxt(self.res_root/'initial_principal_angles.txt')

            # initial angle for an element: average of angles on its nodes
            self.theta0 = np.deg2rad(np.mean(angles[self.elmnodes,0], axis=1))
            self.alpha0 = np.deg2rad(np.mean(angles[self.elmnodes,1], axis=1))

        else:
            raise ValueError('Unsupported value for angle_type')
            
    def set_optim_options(self, *, max_iter=200, tol=0, continuation=False, move=0.2, max_grey=0.3, void_thr=0.1, filled_thr=0.9):
        self.max_iter = max_iter
        self.tol      = tol
        self.move     = move
        
        self.max_grey   = max_grey
        self.void_thr   = void_thr
        self.filled_thr = filled_thr
        
        self.continuation = continuation
        if continuation:
            self.penal      = 1
            self.penal_step = 0.3
        else:
            self.penal      = 3
        
    def run(self):
        try: self.mma
        except AttributeError: self.__create_optimizer()
    
        t0 = time.time()
        for it in range(self.max_iter):
            if self.echo: print('Iteration {:3d}... '.format(it), end=' ')
            xnew = self.mma.iterate(self.x)
            
            if it >= 1 and np.abs(self.comp_max_hist[-1]-self.comp_max_hist[-2])/self.comp_max_hist[-2] < self.tol:
                if not self.continuation:
                    break
                
                if self.get_greyness() < self.max_grey:
                    break
                else:
                    self.penal += self.penal_step
                    self.beta += self.beta_step
                
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
        else:        
            # Evaluate result from last iteration
            if self.echo: print('Iteration {:3d}... '.format(self.mma.iter), end=' ')
            self.fea(self.x)
        
        self.greyness = self.get_greyness()
        
        if self.echo:
            print()
            print('Greyness = {:.3f}'.format(self.greyness))
            print()
        
        self.time += time.time() - t0
        return self.x
            
    def print_timing(self):
        print('Total elapsed time     {:7.2f}s'.format(self.time))
        print('FEA time               {:7.2f}s'.format(self.fea_time))
        print('Derivation time        {:7.2f}s'.format(self.deriv_time))
        print('Variable updating time {:7.2f}s'.format(self.mma.update_time))

    def save(self, filename=None):
        if filename is None: filename = self.res_root / 'topopt.json'
        
        json_str = json.dumps(jsonpickle.encode(self))
        with open(filename, 'w') as f:
            f.write(json_str)
    
    @staticmethod
    def load(filename):
        with open(filename, 'r') as f:
            data = json.load(f)
        return jsonpickle.decode(data)
    
    def get_greyness(self):
        return self.__get_greyness(self.rho_hist[-1])
    
    def get_mass(self, rho):
        x = self.rho_hist[-1]
        mass = rho * x.dot(self.elemvol)
        return mass
    
    def get_max_disp(self, load_case=1):
        u = np.loadtxt(self.res_dir[load_case-1]/'nodal_solution_u.txt', dtype=float) # ux uy uz
        u = np.linalg.norm(u, axis=1)
        return np.amax(u)
    
    def get_CO2_footprint(self, rho, CO2mat, CO2veh):
        return self.get_mass(rho) * (CO2mat + CO2veh)

    # -------------------------------------------- Optimization functions --------------------------------------------
    def fea(self, x):
        try: self.mma
        except AttributeError: self.__create_optimizer() # creates vectors to save compliances if running standalone

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

        if self.fea_fun is not None:
            c = []
            for lc in range(self.load_cases):
                comp, u, energy = self.fea_fun(x, self.penal)
                c.append(comp)
                self.comp_hist[lc].append(c[-1])

                np.savetxt(self.res_dir[lc]/'nodal_solution_u.txt', u, fmt=' %-.7E', newline='\n')
                np.savetxt(self.res_dir[lc]/'strain_energy.txt', energy, fmt=' %-.7E', newline='\n')
                
                if self.echo: print('c_{} = {:10.4f}'.format(lc+1,self.comp_hist[lc][-1]), end=', ')
        else:   
            # Generate file with 1000 discrete materials
            rho_disc = np.linspace(0.001, 1, 1000)
            Ex   = rho_disc**self.penal * self.Ex
            Ey   = rho_disc**self.penal * self.Ey
            nuxy = self.nuxy * np.ones(1000)
            nuyz = self.nuyz * np.ones(1000)
            Gxy  = rho_disc**self.penal * self.Gxy
            Gyz  = Ey/(2*(1+nuyz))
            materials = np.array([Ex, Ey, nuxy, nuyz, Gxy, Gyz]).T
            for lc in range(self.load_cases):
                np.savetxt(self.res_dir[lc]/'materials.txt', materials, fmt=' %-.7E', newline='\n')
            
            # Generate file with material properties for each element
            props = np.array([1000*rho, np.rad2deg(theta), np.rad2deg(alpha)]).T
            for lc in range(self.load_cases):
                np.savetxt(self.res_dir[lc]/'elem_props.txt', props, fmt='%5d %-.7E %-.7E', newline='\n')
            
            # Solve
            c = []
            for lc in range(self.load_cases):
                subprocess.run(self.result_cmd[lc])
                energy = np.loadtxt(self.res_dir[lc]/'strain_energy.txt', dtype=float) # strain_energy
                c.append(2*np.sum(energy))
                self.comp_hist[lc].append(c[-1])
                
                if self.echo: print('c_{} = {:10.4f}'.format(lc+1,self.comp_hist[lc][-1]), end=', ')
            
        comp_max = np.linalg.norm(np.array(c), ord=self.comp_max_order)

        # Save history
        self.rho_hist.append(rho)
        self.theta_hist.append(theta)
        self.alpha_hist.append(alpha)
        self.comp_max_hist.append(comp_max)
        self.penal_hist.append(self.penal)

        self.fea_time += time.time() - t0
        self.__clear_files()
        if self.echo: print()
        return comp_max
    
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
        
        # dcmax/drho = sum(ci**(n-1).cmax**(1-n).dcirho)
        dcdrho = np.zeros_like(rho)
        for lc in range(self.load_cases):
            energy = np.loadtxt(self.res_dir[lc]/'strain_energy.txt', dtype=float) # strain_energy
            uku    = 2*energy/rho**self.penal # K: stiffness matrix with rho=1
            dcidrho = -self.penal * rho**(self.penal-1) * uku
            dcidrho = self.density_filter.filter(rho, dcidrho)
            dcdrho += self.comp_hist[lc][-1]**(self.comp_max_order-1) * dcidrho
        dcdrho *= self.comp_max_hist[-1]**(1-self.comp_max_order)
        
        if self.dim == 'SIMP':
            self.deriv_time += time.time() - t0
            return dcdrho
        
        # dc/dtheta and dc/dalpha
        dcdt = np.zeros_like(theta)
        dcda = np.zeros_like(alpha)
        for lc in range(self.load_cases):
            u = np.loadtxt(self.res_dir[lc]/'nodal_solution_u.txt', dtype=float) # ux uy uz
            if self.dim == '2D':
                u = u[:,[0,1]] # drop z dof

            dcidt = np.zeros_like(theta)
            dcida = np.zeros_like(alpha)
            for i in range(self.num_elem):
                dkdt, dkda = self.dk(theta[i],alpha[i],self.elemvol[i])
                ue = u[self.elmnodes[i,:],:].flatten()
                dcidt[i] = -rho[i]**self.penal * ue.dot(dkdt.dot(ue))
                dcida[i] = -rho[i]**self.penal * ue.dot(dkda.dot(ue))
                
            dcdt += self.comp_hist[lc][-1]**(self.comp_max_order-1) * dcidt
            dcda += self.comp_hist[lc][-1]**(self.comp_max_order-1) * dcida
            
        dcdt *= self.comp_max_hist[-1]**(1-self.comp_max_order)
        dcda *= self.comp_max_hist[-1]**(1-self.comp_max_order)

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
    
    # -------------------------------------------- Internal functions --------------------------------------------
    def __build_apdl_scripts(self):
        inputfiles = self.inputfiles
        self.title = 'file' if self.jobname is None else self.jobname
        meshdata_base = 'ansys_meshdata_2d.txt' if self.dim == '2D' else 'ansys_meshdata_3d.txt'

        # meshdata script
        with open(self.res_root/'ansys_meshdata.txt', 'w') as f:
            f.write(f"RESUME,'{inputfiles[0]}','db','{TopOpt.mod_dir}',0,0\n")
            f.write(f"/CWD,'{self.res_root}'\n")
            f.write(f"/FILENAME,{self.title},1\n")
            f.write(f"/TITLE,{self.title}\n")
            f.write(open(TopOpt.script_dir/meshdata_base).read())
        
        meshdata_cmd = [TopOpt.ANSYS_path, '-b', '-i', self.res_root/'ansys_meshdata.txt', '-o', self.res_root/'meshdata.out', '-smp']
        if self.jobname is not None:
            meshdata_cmd += ['-j', self.title]
            
        # initial orientations script
        with open(self.res_root/'ansys_initial_orientations.txt', 'w') as f:
            f.write(f"RESUME,'{inputfiles[0]}','db','{TopOpt.mod_dir}',0,0\n")
            f.write(f"/CWD,'{self.res_root}'\n")
            f.write(f"/FILENAME,{self.title},1\n")
            f.write(f"/TITLE,{self.title}\n")
            f.write(open(TopOpt.script_dir/'ansys_initial_orientations.txt').read())
            
        initial_orientations_cmd = [TopOpt.ANSYS_path, '-b', '-i', self.res_root/'ansys_initial_orientations.txt', '-o', self.res_root/'orientations.out', '-smp']
        if self.jobname is not None:
            initial_orientations_cmd += ['-j', self.title]
            
        # solve scripts
        result_cmd = []
        for lc, inputfile in enumerate(inputfiles):
            with open(self.res_dir[lc]/'ansys_solve.txt', 'w') as f:
                f.write(f"RESUME,'{inputfile}','db','{TopOpt.mod_dir}',0,0\n")
                f.write(f"/CWD,'{self.res_dir[lc]}'\n")
                f.write(f"/FILENAME,{self.title},1\n")
                f.write(f"/TITLE,{self.title}\n")
                f.write(open(TopOpt.script_dir/'ansys_solve.txt').read())
        
            cmd = [TopOpt.ANSYS_path, '-b', '-i', self.res_dir[lc]/'ansys_solve.txt', '-o', self.res_dir[lc]/'solve.out', '-smp']
            if self.jobname is not None:
                cmd += ['-j', self.title]
            result_cmd.append(cmd)
            
        return meshdata_cmd, initial_orientations_cmd, result_cmd
    
    def __get_mesh_data(self):
        subprocess.run(self.meshdata_cmd)
        self.__clear_files()
        
        num_elem, num_node = np.loadtxt(self.res_root/'elements_nodes_counts.txt', dtype=int) # num_elm num_nodes
        centers            = np.loadtxt(self.res_root/'elements_centers.txt')[:, 1:]          # label x y z
        elmvol             = np.loadtxt(self.res_root/'elements_volume.txt')[:,1]             # label elmvol
        elmnodes           = np.loadtxt(self.res_root/'elements_nodes.txt', dtype=int) - 1    # n1 n2 n3 n4 ...
        node_coord         = np.loadtxt(self.res_root/'node_coordinates.txt')                 # x y z
        
        return num_elem, num_node, centers, elmvol, elmnodes, node_coord

    def __create_optimizer(self):
        rho    = self.volfrac * np.ones(self.num_elem)
        self.x = rho
        self.x[self.solid_elem] = 1
        
        xmin = self.rho_min*np.ones_like(rho)
        xmax = np.ones_like(rho)
        
        # add theta variable
        if self.dim == '2D' or self.dim == '3D_layer' or self.dim == '3D_free':
            theta  = self.theta0
            self.x = np.concatenate((rho,theta))
            xmin = np.concatenate((xmin, -np.pi*np.ones_like(theta)))
            xmax = np.concatenate((xmax, np.pi*np.ones_like(theta)))
        
        # add alpha variable
        if self.dim == '3D_free':
            alpha = self.alpha0
            self.x = np.concatenate((self.x,alpha))
            xmin = np.concatenate((xmin, -np.pi*np.ones_like(alpha)))
            xmax = np.concatenate((xmax, np.pi*np.ones_like(alpha)))
        
        self.mma = MMA(self.fea,self.sensitivities,self.constraint,self.dconstraint,xmin,xmax,self.move)
        
        self.rho_hist      = []
        self.theta_hist    = []
        self.alpha_hist    = []
        self.comp_hist     = [[] for _ in range(self.load_cases)]
        self.comp_max_hist = []
        self.penal_hist    = []
        self.beta_hist     = []
        
        self.time       = 0
        self.fea_time   = 0
        self.deriv_time = 0

    def __get_dk(self):
        # sensitivities: dkdt, dkda = dk(theta,alpha,elmvol)
        if self.dim == '2D':
            dk = lambda theta,alpha,elmvol: dk2d(self.Ex,self.Ey,self.nuxy,self.nuyz,self.Gxy,theta,elmvol)
        elif self.dim == '3D_layer' or self.dim == '3D_free':
            dk = lambda theta,alpha,elmvol: dk3d(self.Ex,self.Ey,self.nuxy,self.nuyz,self.Gxy,theta,alpha,elmvol)
        elif self.dim == 'SIMP':
            dk = None

        return dk
    
    def __get_greyness(self, rho):
        return np.count_nonzero((rho > self.void_thr) & (rho < self.filled_thr))/self.num_elem

    def __clear_files(self):
        # clear Ansys temporary files
        for filename in glob.glob('cleanup*'): os.remove(filename)
        for filename in glob.glob(self.title + '.*'): os.remove(filename)