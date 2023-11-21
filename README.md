# SOMP_Ansys

This code implements a finite element-based topology optimization algorithm for 3D printed structures. It is written in Python and uses Ansys as finite element solver.

### Problem formulation
For each element, a density value $\rho_e$ within the range from 0 (void element) to 1 (filled element) is assigned. To enforce the convergence to a predominantly 0/1 configuration (the only physically feasible values, there is no intermediate material), the material properties for each element are obtained from a power-law interpolation, i.e. dependent on $\rho_e^p$, where $p$ is a penalization factor.

The fiber orientation within each element is defined by two angles: in-plane orientation $\theta_e$ and out-of-plane orientation $\alpha_e$.

The problem formulation, for which the three sets of variables are simultaneously optimized, is (SCHMIDT _et al_, 2020):
$$
\begin{aligned}
    \min_{\boldsymbol\rho,\boldsymbol\theta,\boldsymbol\alpha} c(\boldsymbol\rho,\boldsymbol\theta, \boldsymbol\alpha) & = \sum_e \rho_e^p \, \boldsymbol{u_e^T} \, \boldsymbol{k_0}(\theta_e,\alpha_e) \, \boldsymbol{u_e} \\
    \textrm{s.t. } & \begin{cases}
        \frac{V(\boldsymbol\rho)}{V_0} \leq f \\
        \boldsymbol{KU} = \boldsymbol{F} \\
        0 < \rho_{min} \leq \boldsymbol\rho \leq 1 \\
        -\frac{\pi}{2} \leq \boldsymbol\theta \leq \frac{\pi}{2} \\
        -\frac{\pi}{2} \leq \boldsymbol\alpha \leq \frac{\pi}{2}
    \end{cases}
\end{aligned}
$$
where $\boldsymbol U$ and $\boldsymbol F$ are the global displacement and force vectors, respectively, $\boldsymbol K$ is the global stiffness matrix, $\boldsymbol{u_e}$ and $\boldsymbol{k_e} = \rho_e^p \boldsymbol{k_0}$ are the element displacement vector and stiffness matrix, respectively, $\boldsymbol\rho$ is the vector of design variables, $\rho_{min}$ is the minimum relative density (non-zero to avoid singularity), $V(\boldsymbol\rho)$ and $V_0$ are the material volume and design domain volume, and $f$ is the prescribed volume fraction.

### Filtering
To avoid the appearance of checkerboard patterns and to ensure the mesh-independence of the result, the element sensitivities are filtered by a linear decaying convolution filter (SIGMUND, 2001):
$$
    \rho_e \,\widetilde{\frac{\partial c}{\partial\rho_e}} = \frac{1}{\sum_i H^\rho_{ei}} \, \sum_i H^\rho_{ei} \,\rho_i \,\frac{\partial c}{\partial\rho_i}
$$
$$
    H^\rho_{ei} = \max(0, r_\rho - \Delta(e,i))
$$
where $r_\rho$ is a fixed filter radius and the $\Delta(e,i)$ operator is the distance between the centers of elements $e$ and $i$.

For orientation smoothing, a similar convolution filter was applied directly to the angles at each iteration, adjusted to reduce the weight of void elements on the average:
$$
    \begin{pmatrix}
        \tilde\theta_e \\ \tilde\alpha_e
    \end{pmatrix} = \frac{1}{\sum_i H^\theta_{ei} \, \rho_i} \sum_i H^\theta_{ei} \, \rho_i \begin{pmatrix}
        \theta_i \\ \alpha_i
    \end{pmatrix}
$$
$$
    H^\theta_{ei} = \max(0, r_\theta - \Delta(e,i))
$$
where $r_\theta$ is independent from $r_\rho$ and is related to the desired minimum fiber curvature.

### Finite element formulation
For 2D optimizations, the implementation assumes a 4-node quadrilateral element (Ansys element type PLANE182), whose form functions $N_i$ and strain-displacement matrix $\boldsymbol{B_e}$ are
$$
    \begin{Bmatrix}
        N_1 \\ N_2 \\ N_3  \\ N_4
    \end{Bmatrix} (r,s) = \frac{1}{4} \begin{Bmatrix}
        (1-r) (1-s) \\
        (1+r) (1-s) \\
        (1+r) (1+s) \\
        (1-r) (1+s) \\
    \end{Bmatrix}
$$
$$
    \boldsymbol{B_e} = \begin{bmatrix}
        \frac{\partial N_1}{\partial r} & 0 & \cdots & \frac{\partial N_4}{\partial r} & 0 \\
        0 & \frac{\partial N_1}{\partial s} & \cdots & 0 & \frac{\partial N_4}{\partial s} \\
        \frac{\partial N_1}{\partial s} & \frac{\partial N_1}{\partial r} & \cdots & \frac{\partial N_4}{\partial s} & \frac{\partial N_4}{\partial r} \\
    \end{bmatrix}
$$

For 3D optimizations, the implementation assumes an 8-node brick element (SOLID185):
$$
    \begin{Bmatrix}
        N_1 \\ N_2 \\ N_3 \\ N_4 \\ N_5 \\ N_6 \\ N_7 \\ N_8 \\
    \end{Bmatrix}(r,s,t) = \frac{1}{8} \begin{Bmatrix}
        (1-r) (1-s) (1-t) \\
        (1+r) (1-s) (1-t) \\
        (1+r) (1+s) (1-t) \\
        (1-r) (1+s) (1-t) \\
        (1-r) (1-s) (1+t) \\
        (1+r) (1-s) (1+t) \\
        (1+r) (1+s) (1+t) \\
        (1-r) (1+s) (1+t) \\
    \end{Bmatrix}
$$
$$
    \boldsymbol{B_e} = \begin{bmatrix}
        \frac{\partial N_1}{\partial r} & 0 & 0 & \cdots & \frac{\partial N_8}{\partial r} & 0 & 0 \\
        0 & \frac{\partial N_1}{\partial s} & 0 & \cdots & 0 & \frac{\partial N_8}{\partial s} & 0 \\
        0 & 0 & \frac{\partial N_1}{\partial t} & \cdots & 0 & 0 & \frac{\partial N_8}{\partial t} \\
        0 & \frac{\partial N_1}{\partial t} & \frac{\partial N_1}{\partial s} & \cdots & 0 & \frac{\partial N_8}{\partial t} & \frac{\partial N_8}{\partial s} \\
        \frac{\partial N_1}{\partial t} & 0 & \frac{\partial N_1}{\partial r} & \cdots & \frac{\partial N_8}{\partial t} & 0 & \frac{\partial N_8}{\partial r} \\
        \frac{\partial N_1}{\partial s} & \frac{\partial N_1}{\partial r} & 0 & \cdots & \frac{\partial N_8}{\partial s}  & \frac{\partial N_8}{\partial r} & 0 \\
    \end{bmatrix}
$$

### Material modeling
The materials are modeled as transverse isotropic, suitable for matrices reinforced by unidirectional fibers. The fibers were considered to be aligned with the local $x$ axis, which can be characterised by five independent elastic constants: longitudinal Young modulus $E_x$, transversal Young modulus $E_y$, in-plane Poisson's ratio $\nu_{xy}$, out-of-plane Poisson's ratio $\nu_{yz}$, and in-plane shear modulus $G_{xy}$. From the rule of mixtures for a fiber volume fraction of $V_f$:
$$ E_x = E_f V_f + E_m (1-V_f) $$
$$ E_y = \frac{E_f E_m}{E_f(1-V_f) + E_m V_f} $$
$$ \nu_{xy} = \nu_f V_f + \nu_m (1-V_f) $$
$$ G_{xy} = \frac{G_f G_m}{G_f(1-V_f) + G_m V_f} $$
where $E_f$, $\nu_f$, $G_f$ are the fiber properties and $E_m$, $\nu_m$, $G_m$ are the matrix properties. Finally, $\nu_{yz}$ is defined by symmetries in 3D elasticity (CHRISTENSEN, 1988)
$$ \nu_{yz} = \nu_{xy} \, \frac{1 - \nu_{xy} \frac{E_y}{E_x}}{1 - \nu_{xy}} $$

The constitutive matrix $\boldsymbol{C}$ for transverse isotropic materials and the matrices corresponding to the $\theta_e$ and $\alpha_e$ rotations are
$$
    \boldsymbol{C} = \begin{bmatrix}
        \frac{1}{E_x} & -\frac{\nu_{xy}}{E_x} & -\frac{\nu_{xy}}{E_x} & 0 & 0 & 0 \\
        -\frac{\nu_{xy}}{E_x} & \frac{1}{E_y} & -\frac{\nu_{yz}}{E_y} & 0 & 0 & 0 \\
        -\frac{\nu_{xy}}{E_x} & -\frac{\nu_{yz}}{E_y} & \frac{1}{E_y} & 0 & 0 & 0 \\
        0 & 0 & 0 & \frac{2(1+\nu_{yz})}{E_y} & 0 & 0 \\
        0 & 0 & 0 & 0 & \frac{1}{G_{xy}} & 0 \\
        0 & 0 & 0 & 0 & 0 & \frac{1}{G_{xy}} \\
    \end{bmatrix}^{-1}
$$
$$
    \boldsymbol{T_\theta}(\theta_e) = \begin{bmatrix}
        c_\theta^2 & s_\theta^2 & 0 & 0 & 0 & -2s_\theta c_\theta \\
        s_\theta^2 & c_\theta^2 & 0 & 0 & 0 & 2 s_\theta c_\theta \\
        0 & 0 & 1 & 0 & 0 & 0 \\
        0 & 0 & 0 & c_\theta & s_\theta & 0 \\
        0 & 0 & 0 & -s_\theta & c_\theta & 0 \\
        c_\theta s_\theta & - c_\theta s_\theta & 0 & 0 & 0 & c_\theta^2-s_\theta^2
    \end{bmatrix}
$$
$$
    \boldsymbol{T_\alpha}(\alpha_e) = \begin{bmatrix}
        1 & 0 & 0 & 0 & 0 & 0 \\
        0 & c_\alpha^2 & s_\alpha^2 & -2c_\alpha s_\alpha & 0 & 0 \\
        0 & s_\alpha^2 & c_\alpha^2 & 2 c_\alpha s_\alpha & 0 & 0 \\
        0 & c_\alpha s_\alpha & - c_\alpha s_\alpha & c_\alpha^2-s_\alpha^2 & 0 & 0 \\
        0 & 0 & 0 & 0 & c_\alpha & s_\alpha \\
        0 & 0 & 0 & 0 & -s_\alpha & c_\alpha \\
    \end{bmatrix}
$$
where $c_\theta = \cos\theta_e$, $s_\theta = \sin\theta_e$, $c_\alpha = \cos\alpha_e$, $s_\alpha = \sin\alpha_e$.

### Sensitivity analysis
The variable updating scheme is the Method of Moving Asymptotes - MMA (SVANBERG, 1987). It is then necessary to calculate at each iteration the sensitivities of the compliance with respect to the design variables:
$$
\frac{\partial c}{\partial \rho_e} = -p \, \rho_e^{p-1} \boldsymbol{u_e^T} \, \boldsymbol{k_0} \, \boldsymbol{u_e} = -\frac{p}{\rho_e} \, \underbrace{\rho_e^{p} \boldsymbol{u_e^T} \, \boldsymbol{k_0} \, \boldsymbol{u_e}}_{\substack{\text{element} \\ {\text{strain energy}}}}
$$
$$
\frac{\partial c}{\partial \theta_e} = -\rho_e^p \, \boldsymbol{u_e^T} \, \frac{\partial \boldsymbol{k_0}}{\partial \theta_e} \, \boldsymbol{u_e^T}
$$
$$
\frac{\partial c}{\partial \alpha_e} = -\rho_e^p \, \boldsymbol{u_e^T} \, \frac{\partial \boldsymbol{k_0}}{\partial \alpha_e} \, \boldsymbol{u_e^T}
$$

The derivatives $\frac{\partial \boldsymbol{k_0}}{\partial \theta_e}$ and $\frac{\partial \boldsymbol{k_0}}{\partial \alpha_e}$ need to be integrated since these matrices are not directly accessible as Ansys results. Each integral was numerically evaluated with 2-point Gaussian quadrature.
$$
\frac{\partial \boldsymbol{k_0}}{\partial \theta_e} (\theta_e, \alpha_e) =
\iiint \boldsymbol{B_e^T} \, \boldsymbol{T_\alpha} \, \left( \frac{\partial \boldsymbol{T_\theta}}{\partial \theta_e} \, \boldsymbol{C} \, \boldsymbol{T_\theta^T} + \boldsymbol{T_\theta} \, \boldsymbol{C} \, \frac{\partial \boldsymbol{T_\theta^T}}{\partial\theta_e} \right) \, \boldsymbol{T_\alpha^T} \, \boldsymbol{B_e} \, d\boldsymbol{\Omega}
$$
$$
\frac{\partial \boldsymbol{k_0}}{\partial \alpha_e} (\theta_e, \alpha_e) =
\iiint \boldsymbol{B_e^T} \left( \frac{\partial \boldsymbol{T_\alpha}}{\partial \alpha_e} \boldsymbol{T_\theta} \boldsymbol{C} \boldsymbol{T_\theta^T} \boldsymbol{T_\alpha^T} + \boldsymbol{T_\alpha}  \boldsymbol{T_\theta} \boldsymbol{C} \boldsymbol{T_\theta^T} \frac{\partial \boldsymbol{T_\alpha^T}}{\partial \alpha_e} \right) \boldsymbol{B_e} \, d\boldsymbol{\Omega}
$$

### Multiple load cases
For multiple load cases, the compliances are aggregated using a $n$-norm, which is differentiable:
$$
\begin{aligned}
        \min_{\boldsymbol\rho,\boldsymbol\theta,\boldsymbol\alpha} C(\boldsymbol\rho,\boldsymbol\theta,\boldsymbol\alpha) & = \left(\sum_{i \in LC} c_i(\boldsymbol\rho,\boldsymbol\theta,\boldsymbol\alpha)^n\right)^\frac{1}{n} \\
        & = \left(\sum_{i \in LC} \left(\sum_e \rho_e^p \, \boldsymbol{u_{e,i}^T} \, \boldsymbol{k_0}(\theta_e,\alpha_e) \, \boldsymbol{u_{e,i}}\right)^n\,\right)^\frac{1}{n} \\[10pt]
        \textrm{s.t. } & \begin{cases}
            \frac{V(\boldsymbol\rho)}{V_0} \leq f \\
            \boldsymbol{KU} = \boldsymbol{F} \\
            0 < \rho_{min} \leq \boldsymbol\rho \leq 1 \\
            -\frac{\pi}{2} \leq \boldsymbol\theta \leq \frac{\pi}{2} \\
            -\frac{\pi}{2} \leq \boldsymbol\alpha \leq \frac{\pi}{2} \\
        \end{cases}
    \end{aligned}
$$

The new sensitivities are:
$$
\frac{\partial C}{\partial \, \cdot} = \sum_{i \in LC} c_i^{n-1} \, C^{1-n} \, \frac{\partial c_i}{\partial \, \cdot}
$$

### Continuation method
To facilitate the convergence and avoid local minima, a continuation method in is applied on the penalization factor $p$ as in Castro Almeida (2023). Instead of having a fixed value throughout the whole optimization, it starts at $p = 1$ and is increased each time a convergence in compliance is achieved. The stopping criterion for the continuation is the greyness level of the design, i.e., when the proportion of elements that are neither void nor filled is below a certain level.

### $CO_2$ footprint assessment
The environmental impact of the structure is measured in terms of the mass of $CO_2$ emitted during material production and during its use in a long distance aircraft (DURIEZ _et al_, 2022).

Firstly, the material density $\rho$ was calculated from the fiber and matrix densities $\rho_f$ and $\rho_m$:
$$ \rho = \rho_f V_f + \rho_m (1-V_f) $$

The impact of the material production $CO_{2,mat}$ depends on the total mass $M$ and the $CO_2$ intensity of the material $CO_{2,mat}^i$ (mass of $CO_2$ emitted per mass of material):
$$ CO_{2,mat} = M \cdot CO_{2,mat}^i $$
where $CO_{2,mat}^i$ depends on the $CO_2$ intensities of the fiber and matrix, $CO_{2,f}^i$ and $CO_{2,m}^i$:
$$
    CO_{2,mat}^i = \frac{\rho_f \, V_f \, CO_{2,f}^i + \rho_m \, (1-V_f) \, CO_{2,m}^i}{\rho}
$$

The impact of the use phase $CO_{2,use}$ is calculated as the amount of emissions that would be saved if the component was lighter. Reducing the mass by 1 kg in a long distance aircraft leads to a reduction of 98.8 $tCO_2$ during its lifetime:
$$ CO_{2,use} = M \cdot 98.8\;\mathrm{tCO_2/kg} $$

The value used to compare different designs is the total footprint $CO_{2,tot}$:
$$ CO_{2,tot} = CO_{2,mat} + CO_{2,use} $$

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
- `global3d.py`: example of a complete 3D optimization. Launches multiple optimizations with different materials. Results in `examples/global3d.out`, visualized with `global3d_plots.py`
- `NaturalFibers.ipynb`: Jupyter notebook comparing different materials
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

- `set_initial_conditions(angle_type='random', theta0=0, alpha0=0)`

  - `angle_type`: method for setting the initial orientations
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

- `run()`

  Runs the optimization, saves all results within the `TopOpt` object

- `print_timing()`

  Prints the optimization timing:
  - total elapsed time
  - time spent solving the finite element model
  - time spent calculating sensitivities
  - time spent updating variables

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

## References
- A. Castro Almeida, E. Duriez, F. Lachaud, J. Morlier. New topology optimization for low CO2 footprint AFP composite structures, Poster session, WCSMO 2023.
- R. M. Christensen. Tensor transformations and failure criteria for the analysis of fiber composite materials, Journal of Composite Materials 22.9, 874-897, 1988.
- E. Duriez, J. Morlier, C. Azzaro-Pantel, M. Charlotte. Ecodesign with topology optimization, Procedia CIRP, Elsevier, 454-459, 2022.
- M. Schmidt, L. Couret, C. Gout, C. Pedersen. Structural topology optimization with smoothly varying fiber orientations, Structural and Multidisciplinary Optimization, Springer, 3105-3126, 2020.
- O. Sigmund. A 99 line topology optimization code written in Matlab, Structural and multidisciplinary optimization, Springer, 120-127, 2001.
- K. Svanberg. The method of moving asymptotes—a new method for structural optimization, International journal for numerical methods in engineering, Wiley Online Library, 359-373, 1987.