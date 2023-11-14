import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.spatial import ConvexHull

import niceplots
plt.style.use(niceplots.get_style())

fibers = ['Bamboo', 'Flax', 'Hemp', 'HM carbon', 'LM carbon', 'S-glass', 'E-glass']
resins = ['Cellulose', 'PLA', 'PETG', 'Epoxy', 'Polyester']
names = []

volfrac = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55]
comp = [[] for _ in volfrac]
co2 = [[] for _ in volfrac]

with open('results/natural.out') as f:
    lines = f.readlines()
    for i in range(len(lines)//11): # 11 lines per material
        names.append(lines[11*i].strip())
        for j in range(8):
            comp[j].append(float(lines[11*i+j+2].split()[1]))
            co2[j].append(float(lines[11*i+j+2].split()[3]))

with open('results/carbon_glass.out') as f:
    lines = f.readlines()
    for i in range(len(lines)//11): # 11 lines per material
        names.append(lines[11*i].strip())
        for j in range(8):
            comp[j].append(float(lines[11*i+j+2].split()[1]))
            co2[j].append(float(lines[11*i+j+2].split()[3]))

comp = np.array(comp)
co2 = np.array(co2)

## volfrac = 0.3, index=2
idx = 2
plt.figure()
colors = niceplots.get_colors_list()
labelpos = [(2900,3500), (3300,1600), (3700,750), (5000,3900), (3800,4600), (5900,3200), (6200,600)]
for i in range(len(fibers)):
    x = co2[idx,10*i:10*i+10]
    y = comp[idx,10*i:10*i+10]
    plt.scatter(x, y, c=colors[i])

    # get the convex hull
    points = np.array([[x,y] for x,y in zip(x,y)])
    hull = ConvexHull(points)
    x_hull = np.append(points[hull.vertices,0],
                       points[hull.vertices,0][0])
    y_hull = np.append(points[hull.vertices,1],
                       points[hull.vertices,1][0])
    
    # interpolate
    dist = np.sqrt((x_hull[:-1] - x_hull[1:])**2 + (y_hull[:-1] - y_hull[1:])**2)
    dist_along = np.concatenate(([0], dist.cumsum()))
    spline, u = interpolate.splprep([x_hull, y_hull], 
                                    u=dist_along, s=0, per=1)
    interp_d = np.linspace(dist_along[0], dist_along[-1], 50)
    interp_x, interp_y = interpolate.splev(interp_d, spline)
    # plot shape
    plt.fill(interp_x, interp_y, '--', c=colors[i], alpha=0.2)

    plt.annotate(fibers[i], xy=labelpos[i], color=colors[i])

plt.ylabel('Compliance (N.mm)')
plt.xlabel(r'$CO_{2,tot}$ (kg CO$_2$)')
plt.ylim((500,5300))
plt.xlim((2500,6500))
plt.show()

## natural fibres + matrix, volfrac = 0.3
idx = 2
plt.figure()
colors = niceplots.get_colors_list()
labelpos = [(2850,3250), (4100,2700), (4600,2150)]
for i in range(3): # bamboo, flax, hemp
    x = co2[idx,10*i:10*i+10]
    y = comp[idx,10*i:10*i+10]
    plt.scatter(x[0:2], y[0:2], c=colors[i], marker='o', s=100)
    plt.scatter(x[2:4], y[2:4], c=colors[i], marker='^', s=100)
    plt.scatter(x[4:6], y[4:6], c=colors[i], marker='*', s=100)
    plt.scatter(x[6:8], y[6:8], c=colors[i], marker='s', s=100)
    plt.scatter(x[8:10], y[8:10], c=colors[i], marker='d', s=100)

    # get the convex hull
    points = np.array([[x,y] for x,y in zip(x,y)])
    hull = ConvexHull(points)
    x_hull = np.append(points[hull.vertices,0],
                       points[hull.vertices,0][0])
    y_hull = np.append(points[hull.vertices,1],
                       points[hull.vertices,1][0])
    
    # interpolate
    dist = np.sqrt((x_hull[:-1] - x_hull[1:])**2 + (y_hull[:-1] - y_hull[1:])**2)
    dist_along = np.concatenate(([0], dist.cumsum()))
    spline, u = interpolate.splprep([x_hull, y_hull], 
                                    u=dist_along, s=0, per=1)
    interp_d = np.linspace(dist_along[0], dist_along[-1], 50)
    interp_x, interp_y = interpolate.splev(interp_d, spline)
    # plot shape
    plt.fill(interp_x, interp_y, '--', c=colors[i], alpha=0.2)

    plt.annotate(fibers[i], xy=labelpos[i], color=colors[i])

for i in [1]:
    plt.annotate(names[i], xy=(co2[idx,i],comp[idx,i]), xytext=(co2[idx,i]-200,comp[idx,i]-350), color='k', arrowprops=dict(arrowstyle='->', facecolor='black'))

for i in [10,11]:
    plt.annotate(names[i], xy=(co2[idx,i],comp[idx,i]), xytext=(co2[idx,i]-600,comp[idx,i]-100), color='k', arrowprops=dict(arrowstyle='->', facecolor='black'), va='center')

for i in [20,21]:
    plt.annotate(names[i], xy=(co2[idx,i],comp[idx,i]), xytext=(co2[idx,i]-200,comp[idx,i]-200), color='k', arrowprops=dict(arrowstyle='->', facecolor='black'))

plt.scatter([], [], c='k', marker='o', label=resins[0], s=100)
plt.scatter([], [], c='k', marker='^', label=resins[1], s=100)
plt.scatter([], [], c='k', marker='*', label=resins[2], s=100)
plt.scatter([], [], c='k', marker='s', label=resins[3], s=100)
plt.scatter([], [], c='k', marker='d', label=resins[4], s=100)

plt.ylabel('Compliance (N.mm)')
plt.xlabel(r'$CO_{2,tot}$ (kg CO$_2$)')
plt.ylim((650,3550))
plt.xlim((2450,4800))
plt.legend()
plt.show()

### pareto of all points
# def is_pareto_efficient(costs):
#     """
#     Find the pareto-efficient points
#     :param costs: An (n_points, n_costs) array
#     :return: A (n_points, ) boolean array, indicating whether each point is Pareto efficient
#     """
#     is_efficient = np.ones(costs.shape[0], dtype = bool)
#     for i, c in enumerate(costs):
#         is_efficient[i] = np.all(np.any(costs[:i]>c, axis=1)) and np.all(np.any(costs[i+1:]>c, axis=1))
#     return is_efficient

# plt.figure()
# colors = niceplots.get_colors_list()
# for i in range(7):
#     is_pareto = is_pareto_efficient(np.array([[x,y] for x,y in zip(co2[i,:],comp[i,:])]))
#     plt.scatter(co2[i,is_pareto], comp[i,is_pareto], c=colors[i], label='volfrac = {:.2f}'.format(volfrac[i]))
#     plt.scatter(co2[i,np.logical_not(is_pareto)], comp[i,np.logical_not(is_pareto)], alpha=0.1, c=colors[i])

# plt.ylabel('Compliance (N.mm)')
# plt.xlabel(r'$CO_{2,tot}$ (kg CO$_2$)')
# plt.ylim((0,5200))
# plt.xlim((1500,10500))
# plt.legend()
# plt.show()

# plt.figure()
# colors = niceplots.get_colors_list()
# colors[7] = colors[8]
# pareto = is_pareto_efficient(np.array([[x,y] for x,y in zip(co2.reshape(-1),comp.reshape(-1))])).reshape((8,-1))
# for i in range(8):
#     selected = [range(10),range(10,20),range(20,30),range(30,40),range(40,50),range(50,60),range(60,69)]
#     marker = ['o','^','*','s','d','x','+']
#     for selected, marker, fiber in zip(selected,marker,fibers):
#         is_pareto = pareto[i,:][selected]
#         if i == 0: plt.scatter([],[],c='black',marker=marker,label=fiber)
#         plt.scatter(co2[:,selected][i,is_pareto], comp[:,selected][i,is_pareto], c=colors[i], marker=marker)
#         plt.scatter(co2[:,selected][i,np.logical_not(is_pareto)], comp[:,selected][i,np.logical_not(is_pareto)], alpha=0.05, c=colors[i], marker=marker)

# plt.annotate('f = 0.20', xy=(2400,2300), xytext=(3000,3000), color=colors[0], arrowprops=dict(arrowstyle='-',edgecolor=colors[0]))
# plt.annotate('0.25', xy=(3500,950), xytext=(1900,950), color=colors[1], arrowprops=dict(arrowstyle='-',edgecolor=colors[1]))
# plt.annotate('0.30', xy=(4300,700), xytext=(2000,230), color=colors[2], arrowprops=dict(arrowstyle='-',edgecolor=colors[2]))
# plt.annotate('0.35', xy=(5000,570), xytext=(3750,150), color=colors[3], arrowprops=dict(arrowstyle='-',edgecolor=colors[3]))
# plt.annotate('0.40', xy=(5700,500), xytext=(5000,100), color=colors[4], arrowprops=dict(arrowstyle='-',edgecolor=colors[4]))
# plt.annotate('0.45', xy=(6700,430), xytext=(7000,100), color=colors[5], arrowprops=dict(arrowstyle='-',edgecolor=colors[5]))
# plt.annotate('0.50', xy=(9750,300), xytext=(8500,60), color=colors[6], arrowprops=dict(arrowstyle='-',edgecolor=colors[6]))

# plt.ylabel('Compliance (N.mm)')
# plt.xlabel(r'$CO_{2,tot}$ (kg CO$_2$)')
# plt.ylim((0,4000))
# plt.xlim((1500,10500))
# plt.legend()
# plt.show()