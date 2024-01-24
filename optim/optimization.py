import time
import os, shutil, glob
from pathlib import Path
import numpy as np
import json, jsonpickle

from ansys.mapdl import core as pymapdl

from .dstiffness import dk2d, dk3d
from .filters import DensityFilter, OrientationFilter
from .mma import MMA

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
    
    def __init__(self, *, inputfile, res_dir, load_cases=None, dim='3D_layer', jobname='file', echo=True, **kwargs):
        self.inputfile = Path(inputfile)
        self.res_dir   = Path(res_dir)
        self.dim       = dim
        self.jobname   = jobname
        self.echo      = echo

        self.res_dir.mkdir(parents=True, exist_ok=True)
        
        # define load cases
        if load_cases is None:
            self.num_load_cases = 1
            self.load_cases     = None
        else:
            if isinstance(load_cases,str): load_cases = (load_cases,)
            self.load_cases     = load_cases
            self.num_load_cases = len(load_cases)
            for i, lc in enumerate(load_cases):
                filename = '{}.s{:02d}'.format(self.jobname, i+1)
                shutil.copyfile(Path(lc), self.res_dir/filename)

        self.comp_max_order = 8 # using 8-norm as a differentiable max function of all compliances
        
        self.num_elem, self.num_node, self.centers, self.elem_size, self.elemvol, self.elmnodes, self.node_coord = [
            kwargs.pop(x,None) for x in ['num_elem','num_node', 'centers', 'elem_size', 'elemvol', 'elmnodes', 'node_coord']
        ]
        if self.num_elem is None:
            self.num_elem, self.num_node, self.centers, self.elem_size, self.elemvol, self.elmnodes, self.node_coord = self.__get_mesh_data()
        
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
        euler = [euler1, euler2] if self.dim != 'SIMP2D' and self.dim != '2D' else [0., 0.]
                     
        # slice domain in layers normal to print_direction
        elm_height = np.dot(self.centers, print_direction)
        elm_layer  = (elm_height/self.elem_size).astype(int)
        layers     = [np.where(elm_layer==i)[0] for i in range(np.max(elm_layer)+1)]
        
        # define neighborhoods for overhang projection
        r_s = 1.5*self.r_rho
        if r_s > 0:
            local, support, boundary = self.__overhang_neighborhoods(r_s, layers, print_direction, np.deg2rad(overhang_angle))
                     
        self.print_direction     = print_direction
        self.print_euler         = euler
        self.layers              = layers
        self.layer_thk           = self.elem_size
        self.overhang_angle      = np.deg2rad(overhang_angle)
        self.overhang_constraint = overhang_constraint
        self.r_s                 = r_s
        
        if r_s > 0:
            self.T                   = 1/(np.pi/2-self.overhang_angle) * self.elem_size/(2*r_s) # Heaviside threshold
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
            self.penal_step = 0.5
            self.beta       = 0
            self.beta_step  = 0
        else:
            self.penal      = 3
            self.beta       = 25
        
    def run(self):
        try: self.mma
        except AttributeError: self.__create_optimizer()
    
        t0 = time.time()
        
        self.mapdl = pymapdl.launch_mapdl(jobname=self.jobname, run_location=self.res_dir.absolute(), override=True)
        self.mapdl.resume(fname=self.inputfile.absolute())

        for it in range(self.max_iter):
            if self.echo: print('Iteration {:3d}... '.format(self.mma.iter), end=' ')
            
            # Handle instability on MAPDL connection: if it is broken, creates a new instance and retries
            for attempt in range(5):
                try:
                    xnew = self.mma.iterate(self.current_state['x'])
                except pymapdl.errors.MapdlExitedError:
                    self.mapdl = pymapdl.launch_mapdl(jobname=self.jobname, run_location=self.res_dir.absolute(), override=True)
                    self.mapdl.resume(fname=self.inputfile.absolute())
                else:
                    break
            else:
                raise ConnectionError('Too many failed attempts to reconnect to MAPDL')
            
            if it >= 1 and np.abs(self.comp_max_hist[-1]-self.comp_max_hist[-2])/self.comp_max_hist[-2] < self.tol:
                if not self.continuation:
                    break
                
                if self.get_greyness() < self.max_grey and self.penal > 3:
                    break
                else:
                    self.penal += self.penal_step
                    self.beta  += self.beta_step
                
            self.__update_state(xnew)
        else:        
            # Evaluate result from last iteration
            if self.echo: print('Iteration {:3d}... '.format(self.mma.iter), end=' ')
            self.fea()

        self.mapdl.exit()
        del self.mapdl
        self.__clear_files()
        
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
        return self.current_state['x']
            
    def print_timing(self):
        print('Total elapsed time     {:7.2f}s'.format(self.time))
        print('FEA time               {:7.2f}s'.format(self.fea_time))
        print('Derivation time        {:7.2f}s'.format(self.deriv_time))
        print('Variable updating time {:7.2f}s'.format(self.mma.update_time))

    def save(self, filename=None):
        if filename is None: filename = self.res_dir / 'topopt.json'
        
        json_str = json.dumps(jsonpickle.encode(self.__dict__))
        with open(filename, 'w') as f:
            f.write(json_str)
    
    @staticmethod
    def load(filename):
        with open(filename, 'r') as f:
            data = json.load(f)
        
        dict_rebuilt = jsonpickle.decode(data)
        solver_rebuilt = TopOpt(**dict_rebuilt)
        solver_rebuilt.__dict__ = dict_rebuilt

        # recover callables
        solver_rebuilt.dk = solver_rebuilt.__get_dk()
        try:
            solver_rebuilt.mma.fobj   = solver_rebuilt.fea
            solver_rebuilt.mma.dfobj  = solver_rebuilt.sensitivities
            solver_rebuilt.mma.const  = solver_rebuilt.constraint
            solver_rebuilt.mma.dconst = solver_rebuilt.dconstraint
        except:
            pass
        
        return solver_rebuilt
    
    def get_printability(self):
        if self.overhang_constraint:
            rho_s = self.current_state['rho_s']
        else:
            rho_s = self.__print_eval(self.current_state['rho'])[1]
        
        score = np.average(rho_s, weights=self.current_state['rho'])
        elm_printability = rho_s > 0.5

        return score, elm_printability
    
    def get_greyness(self):
        return self.__get_greyness(self.current_state['rho'])
    
    def get_mass(self, dens):
        mass = dens * np.dot(self.current_state['rho'], self.elemvol)
        return mass
    
    def get_max_disp(self, load_case=1):
        u = self.disp_hist[load_case-1][-1]
        u = np.linalg.norm(u, axis=1)
        return np.max(u)
    
    def get_CO2_footprint(self, dens, CO2mat, CO2veh):
        return self.get_mass(dens) * (CO2mat + CO2veh)

    # -------------------------------------------- Optimization functions --------------------------------------------
    def fea(self, x=None):
        t0 = time.time()

        if x is None:
            mapdl = self.mapdl
        else: # running outside optimization
            self.__create_optimizer()
            self.__update_state(x)
            mapdl = pymapdl.launch_mapdl(jobname=self.jobname, run_location=self.res_dir.absolute(), override=True)
            mapdl.resume(fname=self.inputfile.absolute())

        rho   = self.current_state['rho']
        theta = self.current_state['theta']
        alpha = self.current_state['alpha']

        # Generate 1000 discrete materials
        NUM_MAT = 1000
        rho_disc = np.linspace(0.001, 1, NUM_MAT)
        Ex   = rho_disc**self.penal * self.Ex
        Ey   = rho_disc**self.penal * self.Ey
        nuxy = self.nuxy * np.ones(NUM_MAT)
        nuyz = self.nuyz * np.ones(NUM_MAT)
        Gxy  = rho_disc**self.penal * self.Gxy
        Gyz  = Ey/(2*(1+nuyz))

        mapdl.prep7()
        with mapdl.non_interactive:
            for i in range(NUM_MAT):
                mapdl.mp('ex',i+1,Ex[i])
                mapdl.mp('ey',i+1,Ey[i])
                mapdl.mp('ez',i+1,Ey[i])
                mapdl.mp('prxy',i+1,nuxy[i])
                mapdl.mp('prxz',i+1,nuxy[i])
                mapdl.mp('pryz',i+1,nuyz[i])
                mapdl.mp('gxy',i+1,Gxy[i])
                mapdl.mp('gxz',i+1,Gxy[i])
                mapdl.mp('gyz',i+1,Gyz[i])

            for i in range(self.num_elem):
                mapdl.emodif(i+1,'mat',int(NUM_MAT*rho[i]))
                mapdl.clocal(i+100,0,*self.centers[i,:],*np.rad2deg(self.print_euler),0)      # printing plane
                mapdl.clocal(i+100,0,*[0.,0.,0.],np.rad2deg(theta[i]),np.rad2deg(alpha[i]),0) # material orientation
                mapdl.emodif(i+1,'esys',i+100)
                mapdl.csys(0)
        
            mapdl.slashsolu()

            if self.load_cases is None:
                mapdl.solve()
            else:
                mapdl.kuse(1)
                mapdl.lssolve(1,self.num_load_cases,1)

            mapdl.post1()

        c = []
        for lc in range(self.num_load_cases):
            mapdl.set(lc+1)
            mapdl.etable('energy','sene')
            energy = mapdl.get_array(entity='elem', item1='etable', it1num='energy')

            disp = np.vstack((
                    mapdl.get_array('node', item1='u', it1num='x'),
                    mapdl.get_array('node', item1='u', it1num='y'),
                    mapdl.get_array('node', item1='u', it1num='z'),
                )).T

            c += [2*energy.sum()]
            self.comp_hist[lc]   += [c[-1]]
            self.energy_hist[lc] += [energy]
            self.disp_hist[lc]   += [disp]
            
            if self.echo:
                print('c{} = {:10.4f}'.format('' if self.num_load_cases == 1 else '_' + str(lc+1), self.comp_hist[lc][-1]),
                    end='' if lc == self.num_load_cases-1 else ', ')
            
        if x is not None: mapdl.exit()
        comp_max = np.linalg.norm(np.array(c), ord=self.comp_max_order)

        # Save history
        self.rho_hist      += [rho]
        self.theta_hist    += [theta]
        self.alpha_hist    += [alpha]
        self.comp_max_hist += [comp_max]
        self.penal_hist    += [self.penal]
        self.beta_hist     += [self.beta]

        self.fea_time += time.time() - t0
        if self.echo: print()

        return comp_max
    
    def sensitivities(self):
        t0 = time.time()
        
        rho   = self.current_state['rho']
        theta = self.current_state['theta']
        alpha = self.current_state['alpha']
        
        # dcmax/drho = sum(ci**(n-1).cmax**(1-n).dcirho)
        dcdrho = np.zeros_like(rho)
        for lc in range(self.num_load_cases):
            energy = self.energy_hist[lc][-1]
            uku    = 2*energy/rho**self.penal # K: stiffness matrix with rho=1
            dcidrho = -self.penal * rho**(self.penal-1) * uku
            dcidrho = self.density_filter.filter(rho, dcidrho)
            dcdrho += self.comp_hist[lc][-1]**(self.comp_max_order-1) * dcidrho
        dcdrho *= self.comp_max_hist[-1]**(1-self.comp_max_order)
        
        if self.overhang_constraint:
            drhodpsi = self.current_state['drhodpsi']
            dcdpsi = dcdrho @ drhodpsi
            dcdrho = dcdpsi # renaming to fit into the standard algorithm
        
        if self.dim == 'SIMP2D' or self.dim == 'SIMP3D':
            self.deriv_time += time.time() - t0
            return dcdrho
        
        # dc/dtheta and dc/dalpha
        dcdt = np.zeros_like(theta)
        dcda = np.zeros_like(alpha)
        for lc in range(self.num_load_cases):
            u = self.disp_hist[lc][-1]
            if self.dim == '2D' or self.dim == 'SIMP2D': u = u[:,:2] # drop z dof

            dkdt, dkda = self.dk(theta,alpha,self.elemvol)
            ue = u[self.elmnodes,:]
            ue = ue.reshape(ue.shape[0],-1)[:,:,np.newaxis]

            dcidt = -rho**self.penal * (ue.transpose(0,2,1) @ dkdt @ ue).flatten()
            dcida = -rho**self.penal * (ue.transpose(0,2,1) @ dkda @ ue).flatten()
                
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
    def constraint(self):
        rho = self.current_state['rho']
        return np.dot(rho,self.elemvol)/self.volfrac/np.sum(self.elemvol) - 1
    
    def dconstraint(self):
        dcons = np.zeros_like(self.current_state['x'])
        dcons[:self.num_elem] = self.elemvol/self.volfrac/np.sum(self.elemvol) # dg/drho

        if self.overhang_constraint: # design variable in x is psi instead of rho
            dcons[:self.num_elem] @= self.current_state['drhodpsi'] # dg/dpsi

        return dcons
    
    # -------------------------------------------- Internal functions --------------------------------------------  
    def __get_mesh_data(self):
        mapdl = pymapdl.launch_mapdl(jobname=self.jobname, run_location=self.res_dir.absolute(), override=True)
        mapdl.resume(fname=self.inputfile.absolute())
        
        num_elem   = mapdl.mesh.n_elem
        num_node   = mapdl.mesh.n_node
        node_coord = mapdl.mesh.nodes
        elmnodes   = np.array(mapdl.mesh.elem)[:,10:] - 1
        
        centers    = np.vstack((
                        mapdl.get_array('elem', item1='cent', it1num='x'),
                        mapdl.get_array('elem', item1='cent', it1num='y'),
                        mapdl.get_array('elem', item1='cent', it1num='z'),
                    )).T

        if self.dim == '2D' or self.dim == 'SIMP2D':
            elem_area = mapdl.get_array(entity='elem', item1='geom')
            thk       = mapdl.get_value(entity='rcon', entnum=1, item1='const', it1num=1)
            elemvol   = thk*elem_area
            elem_size = np.mean(np.sqrt(elem_area))
        else:
            elemvol   = mapdl.get_array(entity='elem', item1='geom')
            elem_size = np.mean(np.cbrt(elemvol))

        mapdl.exit()
        self.__clear_files()
        
        return num_elem, num_node, centers, elem_size, elemvol, elmnodes, node_coord
    
    def __overhang_neighborhoods(self, r_s, layers, print_direction, overhang_angle):
        local   = [[] for _ in range(self.num_elem)]
        support = [[] for _ in range(self.num_elem)]
        
        for ei in range(self.num_elem):
            ii = np.where(abs(self.centers[:,0] - self.centers[ei][0]) < r_s)[0]
            jj = np.where(abs(self.centers[ii,1] - self.centers[ei][1]) < r_s)[0]
            kk = np.where(abs(self.centers[ii[jj],2] - self.centers[ei][2]) < r_s)[0]
            ll = np.where(np.logical_not(ii[jj][kk] == ei))[0]
            
            v_ij = self.centers[ii[jj][kk][ll]] - self.centers[ei]
            d = np.sqrt(np.sum(np.square(self.centers[ii[jj][kk][ll]] - self.centers[ei]), axis=1))
            v_ij = (v_ij.T/d).T
            angles = v_ij @ -print_direction
            angles = np.arccos(np.where(angles > 1, 1, np.where(angles < -1, -1, angles)))
            angles = np.asarray(angles).flatten()
            mm = np.where(angles <= np.pi/2 - overhang_angle + 1e-3)[0]
            
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
        rho = self.volfrac * np.ones(self.num_elem)

        if self.overhang_constraint: # all elements filled to avoid numerical problems caused by very high compliances
            rho = np.ones(self.num_elem)
        
        x    = rho
        xmin = np.ones_like(rho) * self.rho_min
        xmax = np.ones_like(rho)
        
        # add theta variable
        if self.dim == '2D' or self.dim == '3D_layer' or self.dim == '3D_free':
            theta = self.theta0
            x     = np.concatenate((x,theta))
            xmin  = np.concatenate((xmin, -np.pi*np.ones_like(theta)))
            xmax  = np.concatenate((xmax, np.pi*np.ones_like(theta)))
        
        # add alpha variable
        if self.dim == '3D_free':
            alpha = self.alpha0
            x     = np.concatenate((x,alpha))
            xmin  = np.concatenate((xmin, -np.pi*np.ones_like(alpha)))
            xmax  = np.concatenate((xmax, np.pi*np.ones_like(alpha)))
        
        self.__update_state(x, apply_filters=False)
        if self.overhang_constraint:
            # asyinit = 0.01
            # asyincr = 1.15
            # asydecr = 0.6
            asyinit = 0.2
            asyincr = 1.2
            asydecr = 0.7
        else:
            asyinit = 0.2
            asyincr = 1.2
            asydecr = 0.7
        self.mma = MMA(self.fea,self.sensitivities,self.constraint,self.dconstraint,xmin,xmax,self.move,asyinit,asyincr,asydecr)
        
        self.rho_hist      = []
        self.theta_hist    = []
        self.alpha_hist    = []
        self.comp_max_hist = []
        self.penal_hist    = []
        self.beta_hist     = []

        self.comp_hist     = [[] for _ in range(self.num_load_cases)]
        self.energy_hist   = [[] for _ in range(self.num_load_cases)]
        self.disp_hist     = [[] for _ in range(self.num_load_cases)]
        
        self.time       = 0
        self.fea_time   = 0
        self.deriv_time = 0

    def __get_dk(self):
        if self.dim == 'SIMP2D' or self.dim == 'SIMP3D': return None

        # sensitivities: dkdt, dkda = dk(theta,alpha,elmvol)
        if self.dim == '2D':
            dk = lambda theta,alpha,elmvol: dk2d(self.Ex,self.Ey,self.nuxy,self.nuyz,self.Gxy,theta,elmvol)
        elif self.dim == '3D_layer' or self.dim == '3D_free':
            dk = lambda theta,alpha,elmvol: dk3d(self.Ex,self.Ey,self.nuxy,self.nuyz,self.Gxy,theta,alpha,elmvol,self.print_euler)

        return dk

    def __update_state(self, x, apply_filters=True):
        self.current_state = dict()

        if self.dim == 'SIMP2D' or self.dim == 'SIMP3D':
            rho = x.copy()
            theta = np.zeros_like(rho)
            alpha = np.zeros_like(rho)
        elif self.dim == '2D' or self.dim == '3D_layer':
            rho, theta = np.split(x,2)
            alpha = np.zeros_like(theta)
        elif self.dim == '3D_free':
            rho, theta, alpha = np.split(x,3)

        rho[self.solid_elem] = 1
        if apply_filters: theta, alpha = self.orientation_filter.filter(rho,theta,alpha)

        if self.overhang_constraint: # design variable in x is psi instead of rho
            psi = rho.copy()
            phi, rho_s, mu_s = self.__print_eval(psi)
            
            phi = self.density_filter.filter(np.ones_like(phi), phi) # direct filtering
            rho = 1 - np.exp(-self.beta*phi) + phi*np.exp(-self.beta)
            rho[rho < self.rho_min] = self.rho_min
            rho[rho > 1] = 1 # correct floating point error

            drhodphi = self.beta*np.exp(-self.beta*phi) + np.exp(-self.beta) # drhodphi[i] = drho_i/dphi_i
            drhodphi *= self.density_filter.filter(np.ones_like(phi), np.ones_like(phi))
            dphidpsi = np.zeros((self.num_elem,self.num_elem)) # dphidpsi[i,j] = dphi_i/dpsi_j
            drhodmu_s = self.betaT/np.cosh(self.betaT*(mu_s-self.T))**2/(np.tanh(self.betaT*self.T) + np.tanh(self.betaT*(1-self.T)))
            for layer in self.layers:
                dphidpsi[layer,layer] = rho_s[layer]
                for eli in layer:
                    if len(self.overhang_support[eli]) == 0: continue
                    dmu_sdphi_sup = 1/len(self.overhang_support[eli])
                    dphidpsi[eli,self.overhang_boundary[eli]] = psi[eli] * drhodmu_s[eli] * dmu_sdphi_sup * np.sum(dphidpsi[self.overhang_support[eli],:][:,self.overhang_boundary[eli]], axis=0)

            drhodpsi = np.diag(drhodphi) @ dphidpsi # drhodpsi[i,j] = drho_i/dpsi_j

            self.current_state['psi']      = psi
            self.current_state['phi']      = phi
            self.current_state['rho_s']    = rho_s
            self.current_state['mu_s']     = mu_s
            self.current_state['drhodpsi'] = drhodpsi

        if self.overhang_constraint: x_mod = psi.copy()
        else:                        x_mod = rho.copy()
            
        if self.dim == '2D' or self.dim == '3D_layer' or self.dim == '3D_free':
            x_mod = np.concatenate((x_mod,theta))
        if self.dim == '3D_free':
            x_mod = np.concatenate((x_mod,alpha))

        self.current_state['x']     = x_mod
        self.current_state['rho']   = rho
        self.current_state['theta'] = theta
        self.current_state['alpha'] = alpha
    
    def __get_greyness(self, rho):
        return np.count_nonzero((rho > self.void_thr) & (rho < self.filled_thr))/self.num_elem
    
    def __print_eval(self, psi):
        phi  = psi.copy()
        beta = self.betaT
        T    = self.T
        
        rho_s = np.ones_like(phi)
        mu_s  = np.ones_like(phi)
        for layeri in self.layers[1:]:
            for eli in layeri:
                neighbors = self.overhang_support[eli]
                mu_s[eli] = np.mean(phi[neighbors]) if len(neighbors) > 0 else 0

            rho_s[layeri] = (np.tanh(beta*T) + np.tanh(beta*(mu_s[layeri]-T)))/(np.tanh(beta*T) + np.tanh(beta*(1-T)))
            phi[layeri]  *= rho_s[layeri]
        
        return phi, rho_s, mu_s

    def __clear_files(self):
        # clear Ansys temporary files
        for filename in list(set(glob.glob(f'{self.res_dir/self.jobname}.*'))
            - set(glob.glob(f'{self.res_dir/self.jobname}.db'))
            - set(glob.glob(f'{self.res_dir/self.jobname}.rst'))
            - set(glob.glob(f'{self.res_dir/self.jobname}.s[0-9]*'))): os.remove(filename)
        for filename in glob.glob(f"{self.res_dir/'ds_file.*'}"): os.remove(filename)
        for filename in glob.glob(f"{self.res_dir/'.__tmp__.*'}"): os.remove(filename)
        for filename in glob.glob(f"{self.res_dir/'*_tmp_*'}"): os.remove(filename)