import subprocess
import time
import os, glob
from pathlib import Path
import numpy as np
import json, jsonpickle

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

    @staticmethod
    def set_paths(ANSYS_path, res_dir, mod_dir):
        TopOpt.ANSYS_path = ANSYS_path # MAPDL executable path
        TopOpt.res_dir    = res_dir    # folder to store results
        TopOpt.mod_dir    = mod_dir    # folder with the .db file (geometry, mesh, constraints, loads)
        TopOpt.script_dir = Path(os.path.abspath(os.path.dirname(__file__)))
    
    def __init__(self, *, inputfiles, dim='3D_layer', jobname=None, echo=True):
        try: TopOpt.ANSYS_path
        except AttributeError: raise Exception('TopOpt object creation failed. Define paths with TopOpt.set_paths()') from None
        
        self.dim  = dim
        self.echo = echo
        
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
        
        # initial setup with default parameters
        self.set_material()
        self.set_volfrac()
        self.set_filters()
        self.set_solid_elem()
        self.set_print_direction()
        self.set_initial_conditions()
        self.set_optim_options()
        
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
            
    def set_print_direction(self, *, print_direction=(0.,0.,1.), overhang_angle=45, overhang_constraint=False):
        # normalize print_direction
        print_direction = np.array(print_direction)
        print_direction /= np.linalg.norm(print_direction)
                     
        # define euler angles of printing coordinate system
        x, y, z = print_direction
        euler1 = -np.arctan2(x,y) # -arctan(x/y), around z
        euler2 = -np.arctan2(np.hypot(x,y),z) # -arctan(sqrt(x^2+y^2)/z), around x'
        euler  = np.array([euler1, euler2]).T
        for lc in range(self.load_cases):
            np.savetxt(self.res_dir[lc]/'print_direction.txt', np.rad2deg(euler), fmt=' %-.7E', newline='\n')
                     
        # slice domain in layers normal to print_direction
        layer_thk = np.mean(np.cbrt(self.elemvol)) # thickness: approximately mean element edge length
        elm_height = np.dot(self.centers, print_direction)
        elm_layer = (elm_height/layer_thk).astype(int)
        
        layers = [[] for _ in range(np.amax(elm_layer)+1)]
        for i in range(self.num_elem):
            layers[elm_layer[i]].append(i)
        
        # define neighborhoods for overhang projection
        h   = np.mean(np.cbrt(self.elemvol)) # approximate element size
        r_s = self.r_rho
        if r_s > 0:
            local, support, boundary = self.__overhang_neighborhoods(r_s, layers, print_direction, np.deg2rad(overhang_angle))
                     
        self.print_direction     = print_direction
        self.print_euler         = euler
        self.layers              = layers
        self.layer_thk           = layer_thk
        self.overhang_angle      = np.deg2rad(overhang_angle)
        self.overhang_constraint = overhang_constraint
        self.r_s                 = r_s
        
        if r_s > 0:
            self.T                   = 1/(np.pi/2-self.overhang_angle) * h/(2*r_s) # Heaviside threshold
            self.betaT               = 25 # Heaviside parameter
            self.overhang_local      = local
            self.overhang_support    = support
            self.overhang_boundary   = boundary

        # update derivative function (dependent on print_direction)
        self.dk = self.__get_dk()
        
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
            self.beta       = 5
            self.beta_step  = 5
        else:
            self.penal      = 3
            self.beta       = 25 # overhang projection parameter
        
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
        
        if self.r_s > 0:
            self.print_score, self.elm_printability = self.get_printability()
        self.greyness = self.get_greyness()
        
        if self.echo:
            print()
            if self.r_s > 0:
                print('Printability score = {:.3f}'.format(self.print_score))
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
    
    def get_printability(self):
        _, _, _, elm_printability, score = self.__printability(self.rho_hist[-1])
        return score, elm_printability
    
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
            
        if self.overhang_constraint: # design variable in x is psi instead of rho
            psi = rho.copy()
            phi = self.__printability(psi)[0]
            rho = 1 - np.exp(-self.beta*phi) + phi*np.exp(-self.beta)
            rho[rho < self.rho_min] = self.rho_min
            
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
        self.beta_hist.append(self.beta)

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
            
        if self.overhang_constraint: # design variable in x is psi instead of rho
            psi = rho.copy()
            phi, rho_s, mu_s = self.__printability(psi)[0:3]
            rho = 1 - np.exp(-self.beta*phi) + phi*np.exp(-self.beta)
            rho[rho < self.rho_min] = self.rho_min
        
        # dcmax/drho = sum(ci**(n-1).cmax**(1-n).dcirho)
        dcdrho = np.zeros_like(rho)
        for lc in range(self.load_cases):
            energy = np.loadtxt(self.res_dir[lc]/'strain_energy.txt', dtype=float) # strain_energy
            uku    = 2*energy/rho**self.penal # K: stiffness matrix with rho=1
            dcidrho = -self.penal * rho**(self.penal-1) * uku
            dcidrho = self.density_filter.filter(rho, dcidrho)
            dcdrho += self.comp_hist[lc][-1]**(self.comp_max_order-1) * dcidrho
        dcdrho *= self.comp_max_hist[-1]**(1-self.comp_max_order)
        
        if self.overhang_constraint:
            drhodphi = np.zeros_like(rho)
            dphidpsi = np.zeros((self.num_elem,self.num_elem)) # dphidpsi[i,j] = dphii/dpsij
            for layer in self.layers:
                for eli in layer:
                    drhodphi = self.beta*np.exp(-self.beta*phi[eli]) + np.exp(-self.beta)

                    dphidpsi[eli,eli] = rho_s[eli]
                    for elj in self.overhang_boundary[eli]:
                        dphidpsi[eli,elj] = psi[eli] * self.beta/np.cosh(self.beta*(mu_s[eli]-self.T))**2/(np.tanh(self.beta*self.T) + np.tanh(self.beta*(1-self.T))) * 1/len(self.overhang_support[eli]) * np.sum(dphidpsi[:,elj])
            
            dcdpsi = np.dot(dcdrho*drhodphi, dphidpsi)
            dcdrho = dcdpsi # renaming to fit into the standard algorithm
        
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
        if self.overhang_constraint: # design variable in x is psi instead of rho
            psi = rho.copy()
            phi = self.printability(psi)[0]
            rho = 1 - np.exp(-self.beta*phi) + phi*np.exp(-self.beta)
            rho[rho < self.rho_min] = self.rho_min
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
    
    def __overhang_neighborhoods(self, r_s, layers, print_direction, overhang_angle):
        local   = [[] for _ in range(self.num_elem)]
        support = [[] for _ in range(self.num_elem)]
        
        def distances(matrixA, matrixB):
            A = np.matrix(matrixA)
            B = np.matrix(matrixB)
            Btrans = B.transpose()
            vecProd = A * Btrans
            SqA =  A.getA()**2
            sumSqA = np.matrix(np.sum(SqA, axis=1))
            sumSqAEx = np.tile(sumSqA.transpose(), (1, vecProd.shape[1]))    
            SqB = B.getA()**2
            sumSqB = np.sum(SqB, axis=1)
            sumSqBEx = np.tile(sumSqB, (vecProd.shape[0], 1))    
            SqED = sumSqBEx + sumSqAEx - 2*vecProd   
            elmDis = (np.maximum(0,SqED).getA())**0.5
            return np.matrix(elmDis)
        
        for ei in range(self.num_elem):
            ii = np.where(abs(self.centers[:,0] - self.centers[ei][0]) < r_s)[0]
            jj = np.where(abs(self.centers[ii,1] - self.centers[ei][1]) < r_s)[0]
            kk = np.where(abs(self.centers[ii[jj],2] - self.centers[ei][2]) < r_s)[0]
            ll = np.where(np.logical_not(ii[jj][kk] == ei))[0]
            
            v_ij = self.centers[ii[jj][kk][ll]] - self.centers[ei]
            d = distances(self.centers[ei], self.centers[ii[jj][kk][ll]])
            v_ij = (v_ij.T/d).T
            angles = np.arccos(v_ij @ -print_direction)
            mm = np.where(angles <= np.pi/2 - overhang_angle)[0]
            
            local[ei]   = ii[jj][kk]
            support[ei] = ii[jj][kk][ll][mm]
        
        boundary = support.copy()
        for layer in layers[2:]:
            for ei in layer:
                for ej in support[ei]:
                    boundary[ei] = np.concatenate((boundary[ei], boundary[ej]))
                boundary[ei] = np.unique(boundary[ei])
                        
        return local, support, boundary

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
            dk = lambda theta,alpha,elmvol: dk3d(self.Ex,self.Ey,self.nuxy,self.nuyz,self.Gxy,theta,alpha,elmvol,self.print_euler)

        return dk
    
    def __get_greyness(self, rho):
        return np.count_nonzero((rho > self.void_thr) & (rho < self.filled_thr))/self.num_elem
    
    def __printability(self, psi):
        phi  = np.copy(psi)
        beta = self.beta
        T    = self.T
        
        rho_s = np.ones_like(phi)
        mu_s  = np.ones_like(phi)
        for layeri in self.layers[1:]:
            for eli in layeri:
                neighbors = self.overhang_support[eli]
                mu_s[eli] = np.mean(phi[neighbors]) if len(neighbors) > 0 else 0
                rho_s[eli] = (np.tanh(beta*T) + np.tanh(beta*(mu_s[eli]-T)))/(np.tanh(beta*T) + np.tanh(beta*(1-T)))
                phi[eli] *= rho_s[eli] # unsupported element cannot support above elements
        
        score = np.average(rho_s, weights=psi)
        elm_printability = rho_s > 0.5
        
        return phi, rho_s, mu_s, elm_printability, score

    def __clear_files(self):
        # clear Ansys temporary files
        for filename in glob.glob('cleanup*'): os.remove(filename)
        for filename in glob.glob(self.title + '.*'): os.remove(filename)