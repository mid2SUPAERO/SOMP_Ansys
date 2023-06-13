import multiprocessing

from python.optimization import TopOpt2D
from python.postprocessor import PostProcessor

ANSYS_path = "mapdl"
script_dir = "python/"
res_dir    = "results/"
mod_dir    = "models/"
TopOpt2D.load_paths(ANSYS_path, script_dir, res_dir, mod_dir)
TopOpt2D.set_processors(3)

# fiber: bamboo
rhofiber  = 700e-12 # t/mm^3
Efiber    = 17.5e3 # MPa
vfiber    = 0.04
CO2fiber  = 1.0565 # kgCO2/kg

# matrix: cellulose
rhomatrix = 990e-12 # t/mm^3
Ematrix   = 3.25e3
vmatrix   = 0.355 # MPa
CO2matrix = 3.8 # kgCO2/kg

Vfiber  = 0.5
Vmatrix = 1-Vfiber

Gfiber  = Efiber/(2*(1+vfiber))
Gmatrix = Ematrix/(2*(1+vmatrix))

Ex  = Efiber*Vfiber + Ematrix*Vmatrix
Ey  = Efiber*Ematrix / (Efiber*Vmatrix + Ematrix*Vfiber)
Gxy = Gfiber*Gmatrix / (Gfiber*Vmatrix + Gmatrix*Vfiber)
nu  = vfiber*Vfiber + vmatrix*Vmatrix
rho = rhofiber*Vfiber + rhomatrix*Vmatrix

Ntheta = 3
theta0 = np.linspace(-90, 90, num=Ntheta)

def optim(theta):
    solver = TopOpt2D(inputfile='mbb2d', Ex=Ex, Ey=Ey, Gxy=Gxy, nu=nu, volfrac=0.3, rmin=6, theta0=theta)
    solver.optim()
    return solver

with multiprocessing.Pool() as pool:
    for solver in pool.map(optim,theta0):
        post = PostProcessor(solver)
        # important data to save: theta0, compliance, iterations, execution time, CO2 footprint
