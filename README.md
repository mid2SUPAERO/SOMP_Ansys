# SOMP_Ansys

## Dependencies

The code was tested with the following libraries in Python >= 3.7:
- NumPy >= 1.21.5
- SciPy >= 1.6.2
- Matplotlib >= 3.5.1
- jsonpickle >= 3.0.2

Examples may have additional dependencies:
- mpi4py 3.0.1
- niceplots 2.4.0

## Examples

- `SimpleExample.ipynb`: Jupyter notebook with an example of 2D and 3D optimizations
- `global3d.py`: example of a complete 3D optimization. Uses MPI to launch parallel processes with different volume fraction constraints and materials. Results in `results/carbon_glass.out` and `results/natural.out`, visualized in `plots.py`
- `NaturalFibres.ipynb`: Jupyter notebook comparing different materials
- `bridge.py`: example of a complete 2D optimization. Uses MPI to launch parallel processes with different initial orientations
- `Bracket.ipynb`: Jupyter notebook with a more complex 3D optimization

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

- `TopOpt(inputfiles, Ex, Ey, nuxy, nuyz, Gxy, volfrac, r_rho, r_theta, print_direction, initial_angles_type, theta0, alpha0, move, max_iter, tol, dim, jobname, echo)`
  - `inputfiles`: name of the model file (without .db). For multiple load cases, tuple with all model files
  - `Ex`, `Ey`, `nuxy`, `nuyz`, `Gxy`: material properties (considered transverse isotropic, symmetry plane $yz$)
  - `volfrac`: volume fraction constraint for the optimization
  - `r_rho`: radius of the density filter (adjusts minimum feature size)
  - `r_theta`: radius of the orientation filter (adjusts fiber curvature)
  - `print_direction`: defaults to `(0.,0.,1.)`
  - `move`: move limit for variable updating, as a fraction of the allowed range
  - `max_iter`: number of iterations
  - `tol`: stopping criterion, relative change in the objective function. Defaults at 0, i.e., will run until max_iter
  - `initial_angles_type`: method for setting the initial orientations. Defaults to `'fix'`
    - `'fix'`: initial orientations are given
    - `'noise'`: gaussian distribution around the giuven values
    - `'random'`: random orientation for each element
    - `'principal'`: initial orientations follow the principal stress directions considering the whole domain with an isotropic material
  - `theta0`: initial orientation (around z) of the fibers, in degrees. Default: `0.0`
  - `alpha0`: initial orientation (around x) of the fibers, in degrees. Default: `0.0`
  - `dim`: optimization type
    - `'SIMP'`: SIMP method, 2D or 3D optimization
    - `'2D'`: 2D optimization
    - `'3D_layer'`: 3D optimization, structure will be printed in layers and fibers can rotate in the plane $xy$ (defined by the normal vector `print_direction`)
    - `'3D_free'`: 3D optimization, fiber orientations defined by rotations around $z$ and $x$ respectively
  - `jobname`: optional. Subfolder of `TopOpt.res_dir` to store results for this optim. Default: no subfolder, stores results directly on `TopOpt.res_dir`
  - `echo`: boolean. Print compliance at each iteration?

- `set_solid_elem(self, solid_elem)`: list of elements whose densities will be fixed on 1. Indexing starting at 0

- `optim(self)`: runs the optimization, saves all results within the `TopOpt` object

- `save(self, filename)`: saves object into a JSON file

- `load(filename)`: returns `TopOpt` object from JSON file

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
- `plot(self, iteration=-1, colorful=True, elev=None, azim=None, filename=None, save=True, fig=None, ax=None, zoom=None)`: plots the configuration (densities and orientations). A zoom to a specific area can be added by passing a dict with fileds `xmin`, `xmax`, `ymin`, `ymax` representing the area of interest (in length units), `xpos`, `ypos`, `width`, `height` defining the position of the zoomed figure (fraction of the original axes), and `color` defining the color of the outline boxes
- `animate(self, filename=None, colorful=True, elev=None, azim=None)`: creates an animation with `self.plot` history
- `plot_layer(self, iteration=-1, layer=0, colorful=False, filename=None, save=True, fig=None, ax=None, zoom=None)`: only for `dim = '3D_layer'`. Plots layer `layer` as a 2D plot, easier to visualise
- `animate_layer(self, layer, colorful=False, filename=None)`: creates an animation with `self.plot_layer` history
