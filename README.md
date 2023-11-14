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

## Model files

- For 2D optimization: use 4-node 2D quad elements (PLANE182), with KEYOPT(3) = 3 (plane stress with thk)
- For 3D optimization: use 8-node 3D hex elements (SOLID185)

## Usage

Minimal example of an optimization of the model `models/mbb3d.db` made of bamboo and cellulose with a volume fraction of 0.3. Results are saved in the folder `results/`

The next sections present all available functions

```
from optim import TopOpt, Post2D, Post3D

# -------------------- Configure paths --------------------
ANSYS_path = Path('mapdl')
res_dir    = Path('results/')
mod_dir    = Path('models/')
TopOpt.set_paths(ANSYS_path, res_dir, mod_dir)

# -------------------- Define materials -------------------
# {t/mm^3, MPa, -, kgCO2/kg}
bamboo     = {'rho': 700e-12, 'E': 17.5e3, 'v': 0.04, 'CO2': 1.0565}
cellulose  = {'rho': 990e-12, 'E': 3.25e3, 'v': 0.355, 'CO2': 3.8}
Vfiber  = 0.5

Ex, Ey, nuxy, nuyz, Gxy, rho, CO2mat = TopOpt.rule_mixtures(fiber=bamboo, matrix=cellulose, Vfiber=Vfiber)

# --------------------- Define problem --------------------
solver = TopOpt(inputfiles='mbb3d', dim='3D_layer', jobname='3d')
solver.set_material(Ex=Ex, Ey=Ey, nuxy=nuxy, nuyz=nuxy, Gxy=Gxy)
solver.set_volfrac(0.3)
solver.set_filters(r_rho=8, r_theta=20)
solver.set_initial_conditions('random')
solver.create_optimizer()

# ---------------------- Run and save ---------------------
solver.run()
solver.save()

# ---------------------- Visualization --------------------
post = Post3D(solver)
post.plot_convergence()
post.plot(colorful=False)
```

## Class `TopOpt`

### Ansys configuration

- `TopOpt.load_paths(ANSYS_path, res_dir, mod_dir)`

  Configures the paths to be used by all `TopOpt` objects, passed as `pathlib.Path` objects

  - `ANSYS_path`: MAPDL executable path
  - `res_dir`: folder to store results. Individual job results can be stored in subfolders of it
  - `mod_dir`: folder with the .db file (geometry, mesh, constraints, loads)

### Material properties

- `TopOpt.rule_mixtures(fiber, matrix, Vfiber)`

  Returns `Ex`, `Ey`, `nuxy`, `nuyz`, `Gxy`, `rho`, `CO2mat` by the rule of mixtures for a material with fiber volume fraction `Vfiber`. Components `fiber` and `matrix` are given as dictionaries with fields `'rho'`, `'E'`, `'v'`, `'CO2'`

### Problem definition

- `TopOpt(inputfiles, dim='3D_layer', jobname=None, echo=True)`
  - `inputfiles`: name of the model file (without .db). For multiple load cases, tuple with all model files
  - `dim`: optimization type
    - `'SIMP'`: SIMP method, 2D or 3D optimization
    - `'2D'`: 2D optimization
    - `'3D_layer'`: 3D optimization, structure will be printed in layers and fibers can rotate in the plane $x'y'$ (defined by the normal vector `print_direction`)
    - `'3D_free'`: 3D optimization, fiber orientations defined by rotations around $z$ and $x$ respectively
  - `jobname`: subfolder of `TopOpt.res_dir` to store results for this optim. Default: no subfolder, stores results directly on `TopOpt.res_dir`
  - `echo`: print compliance at each iteration?

All setter functions are optional, the optimization can be performed with the default parameters

- `set_material(Ex=1, Ey=1, nuxy=0.3, nuyz=0.3, Gxy=1/(2*(1+0.3)))`
  
  Defines material properties, considered transverse isotropic with symmetry plane $yz$

- `set_volfrac(volfrac=0.3)`

  Defines the volume fraction constraint for the optimization

- `set_filters(r_rho=0, r_theta=0)`

  Defines the filters applied on each iteration, defaults to no filtering

  - `r_rho`: radius of the density filter (adjusts minimum feature size)
  - `r_theta`: radius of the orientation filter (adjusts fiber curvature)
  
- `set_solid_elem(solid_elem=[])`

  List of elements whose densities will be fixed on 1. Indexing starting at 0

- `set_print_direction(print_direction=(0.,0.,1.), overhang_angle=45, overhang_constraint=False)`

  Defines the printing direction, to verify if the structure is printable without adding supports. Depends on `r_rho`, so should be called after `set_filters()`

  - `print_direction`: vector pointing upwards in the printing coordinate system. Does not need to be unitary
  - `overhang_angle`: minimum self-supporting angle. Minimum angle with respect to the horizontal at which features may be created
  - `overhang_constraint`: add overhang constraint to the optimization? Results in self-supported structures

- `set_initial_conditions(initial_angles_type='random', theta0=0, alpha0=0)`

  - `initial_angles_type`: method for setting the initial orientations
    - `'fix'`: initial orientations are given
    - `'noise'`: gaussian distribution around the given values
    - `'random'`: random orientation for each element
    - `'principal'`: initial orientations follow the principal stress directions considering the whole domain with an isotropic material

  - `theta0`: initial orientation (around $z'$) of the fibers, in degrees. Only used with `'fix'` and `'noise'`
  - `alpha0`: initial orientation (around $x'$) of the fibers, in degrees. Only used with `'fix'` and `'noise'`

- `set_optim_options(max_iter=200, tol=0, continuation=False, move=0.2, max_grey=0.3, void_thr=0.1, filled_thr=0.9)`

  - `max_iter`: maximum number of iterations
  - `tol`: stopping criterion, relative change in the objective function. Defaults at 0, i.e., will run until max_iter
  - `continuation`: use a continuation method in the penalization factor?
  - `move`: move limit for variables updating, given as fraction of the allowable range
  - `max_grey`: stopping criterion for continuation, maximum fraction of grey elements, i.e. nor void nor filled
  - `void_thr`: elements are considered void if they have density less or equal `void_thr`
  - `filled_thr`: elements are considered void if they have density greater or equal `filled_thr`

### Run the optimization

- `create_optimizer()`

  Gathers all settings and creates the variable updating scheme. Has to be called before running the optimization

- `run()`

  Runs the optimization, saves all results within the `TopOpt` object

- `print_timing()`

  Prints the optimization timing:
  - total elapsed time
  - time spent solving the finite element model
  - time spent calculating sensitivities
  - time spent updating variables

- `clear_files()`

  Deletes temporary Ansys files related to this optimization. It is automatically called inside `run()`

- `save(filename=None)`

  Saves object into a JSON file

### Design evaluation

- `TopOpt.load(filename)`

  Returns `TopOpt` object from JSON file created by `save()`

- `get_printability()`

  Returns the printability score of the final design (fraction of self-supported elements) and a boolean list indicating whether each element is self-supported

- `get_greyness()`

  Returns the greyness of the final design (fraction of grey elements)

- `get_mass(rho)`

  Returns the mass of the final design
  
- `get_max_disp(load_case=1)`

  Returns the maximum nodal displacement for the `load_case`-th load case, indexing starting on 1

- `get_CO2_footprint(rho, CO2mat, CO2veh)`

  Returns the $CO_2$ footprint of the final design
  - `rho`: density
  - `CO2mat`: mass $CO_2$ emmited per mass material (material production)
  - `CO2veh`: mass $CO_2$ emitted per mass material during life (use in a vehicle) = mass fuel per mass transported per lifetime * service life * mass $CO_2$ emmited per mass fuel

- `fea(x)`

  Returns the aggregated compliance obtained for a specific set of design variables `x`. Useful to evaluate a design obtained from other optimization, but using the current load cases (given the meshes and design variables are the same)

## Class `PostProcessor`

For 2D optimizations: `Post2D(solver)`

For 3D optimizations: `Post3D(solver)`

- `plot_convergence(start_iter=0, penal=False, filename=None, save=True)`

  Simple plot of the convergence history

  - `start_iter`: first iteration shown, may be used to eliminate first values that break the compliance scale
  - `penal`: show subplot with penalization factor evolution?

- `plot(iteration=-1, colorful=True, printability=False, elev=None, azim=None, domain_stl=None, filename=None, save=True, fig=None, ax=None, zoom=None)`

  Plots the configuration at iteration `iteration`

  - `colorful`: color arrows based on their orientation? RGB values based on the vector components:
    - $x$: blue
    - $y$: green
    - $z$: red
  - `zoom`: only for `Post2D`. Inner plot with zoom to a specific area can be added by passing a dict with fields
    - `'xmin'`, `'xmax'`, `'ymin'`, `'ymax'`: area of interest (in model length units)
    - `'xpos'`, `'ypos'`, `'width'`, `'height'`: position and size of the zoomed plot (fraction of the original axes)
    - `'color'`: color of the outline boxes
  - `printability`: only for `Post3D`. Color arrows according to their printability? Self-supported elements are shown in black, non-self-supported elements are shown in red. If `True`, ignores `colorful`
  - `elev`, `azim`: only for `Post3D`. Angles defining the point of view
  - `domain_stl`: only for `Post3D`. Filename of the .stl file with the original domain. If defined, plots an outline of the domain

- `animate(filename=None, colorful=True, printability=False, elev=None, azim=None)`

  Creates an animation where each frame is the `plot()` for one iteration

### Functions specific to `Plot3D`

- `plot_layer(iteration=-1, layer=0, colorful=False, printability=False, filename=None, save=True, fig=None, ax=None, zoom=None)`

  Plots the `layer`-th layer, analogous to `plot()` for `Plot2D`

- `animate_layer(layer=0, colorful=False, filename=None)`

  Creates an animation where each frame is the `plot_layer(layer)` for one iteration

- `animate_print(colorful=False, printability=False, filename=None)`

  Creates an animation where each frame is the `plot_layer(layer)` for one layer, in the final configuration

- `plot_fill(iteration=-1, threshold=0.8, filename=None, save=True, fig=None, ax=None)`

  Plots each eleemnt with density greater than `threshold` as a box. Not efficient, makes a plot of every face of every element
