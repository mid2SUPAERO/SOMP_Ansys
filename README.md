# SOMP_Ansys

## Dependencies

The code was tested with the following libraries in Python >= 3.7:
- NumPy >= 1.21.5
- SciPy >= 1.6.2
- Matplotlib >= 3.5.1

Examples may have additional dependencies:
- mpi4py 3.0.1

## Examples

- `SimpleExample.ipynb`: Jupyter notebook with an example of 2D and 3D optimizations
- `global3d.py`: example of a complete 3D optimization. Uses MPI to launch parallel processes with different volume fraction constraints and materials
- `SimpleExample.ipynb`: Jupyter notebook comparing different materials

## Usage 

### Ansys configuration

- `TopOpt.load_paths(ANSYS_path, script_dir, res_dir, mod_dir)`: configures the paths to be used by all `TopOpt` objects
  - `ANSYS_path : pathlib.Path`: MAPDL executable path
  - `script_dir : pathlib.Path`: folder with .py files and .txt APDL scripts
  - `res_dir : pathlib.Path`: folder to store results. Individual job results can be stored in subfolders of it
  - `mod_dir : pathlib.Path`: folder with the .db file (geometry, mesh, constraints, loads)

### Model files

- For 2D optimization: use 4-node 2D quad elements (PLANE182) in a rectangular domain, with KEYOPT(3) = 3 (plane stress with thk)
- For 3D optimization: use 8-node 3D hex elements (SOLID185) in a cuboid domain

### Class `TopOpt`

- `TopOpt(inputfile, Ex, Ey, nuxy, nuyz, Gxy, volfrac, r_rho, r_theta, theta0, alpha0, max_iter, dim, jobname, echo)`
  - `inputfile`: name of the model file (without .db)
  - `Ex`, `Ey`, `nuxy`, `nuyz`, `Gxy`: material properties (considered transverse isotropic, symmetry plane $yz$)
  - `volfrac`: volume fraction constraint for the optimization
  - `r_rho`: radius of the density filter (adjusts minimum feature size)
  - `r_theta`: radius of the orientation filter (adjusts fiber curvature)
  - `max_iter`: number of iterations
  - `theta0`: initial orientation (around z) of the fibers, in degrees. Default: random distribution
  - `alpha0`: initial orientation (around x) of the fibers, in degrees. Default: random distribution
  - `dim`: optimization type
    - `'SIMP'`: SIMP method, 2D or 3D optimization
    - `'2D'`: 2D optimization
    - `'3D_layer'`: 3D optimization, structure will be printed in layers and fibers can rotate in the plane $xy$
    - `'3D_free'`: 3D optimization, fiber orientations defined by rotations around $z$ and $x$ respectively
  - `jobname`: optional. Subfolder of `TopOpt.res_dir` to store results for this optim. Default: no subfolder, stores results directly on `TopOpt.res_dir`
  - `echo`: boolean. Print compliance at each iteration?

- `set_solid_elem(self, solid_elem)`: list of elements whose densities will be fixed on 1. Indexing starting at 0

- `optim(self)`: runs the optimization, saves all results within the `TopOpt`object

- `mass(self, rho)`: returns the mass of the final design
  - `rho`: density
  
- `disp_max(self)`: returns the maximum nodal displacement

- `CO2_footprint(self, rho, CO2mat, CO2veh)`: returns the CO2 footprint of the final design
  - `rho`: density
  - `CO2mat`: mass CO2 emmited per mass material (material production)
  - `CO2veh`: mass CO2 emitted per mass material during life (use in a vehicle) = mass fuel per mass transported per lifetime * service life * mass CO2 emmited per mass fuel

### Class `PostProcessor`

- `Post2D(solver)`, `Post3D(solver)`
- `plot_convergence(self, starting_iteration=0)`: plots the convergence history
- `plot(self, iteration=-1, colorful=True, elev=None, azim=None, filename=None, save=True, fig=None, ax=None)`: plots the configuration (densities and orientations)
- `animate(self, filename=None, colorful=True, elev=None, azim=None)`: creates an animation with `self.plot` history
- `plot_layer(self, iteration=-1, layer=0, colorful=False, filename=None, save=True, fig=None, ax=None)`: only for `dim = '3D_layer'`. Plots layer `layer` as a 2D plot, easier to visualise
- `animate_layer(self, layer, colorful=False, filename=None)`: creates an animation with `self.plot_layer` history
