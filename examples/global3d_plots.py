import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.spatial import ConvexHull

import niceplots
plt.style.use(niceplots.get_style())

fibers = ['Bamboo', 'Flax', 'Hemp', 'HM carbon', 'LM carbon', 'S-glass', 'E-glass']
resins = ['Cellulose', 'PLA', 'PETG', 'Epoxy', 'Polyester']

comp = []
co2 = []

with open('examples/global3d.out') as f:
    lines = f.readlines()[3:-1]
    for line in lines:
        data = [data.strip() for data in line.split('|')[1:-2]]
        comp.append(float(data[2]))
        co2.append(float(data[4]))

comp = np.array(comp).reshape((len(fibers),len(resins)))
co2 = np.array(co2).reshape((len(fibers),len(resins)))

plt.figure()
colors = niceplots.get_colors_list()
markers = ['o','^','*','s','d']
labelpos = [(3000,2050), (3400,1100), (3550,600), (5750,100), (3400,200), (4900,1000), (5200,1400)]
for i in range(len(fibers)):
    x = co2[i,:]
    y = comp[i,:]
    
    for j in range(len(resins)):
        plt.scatter(x[j], y[j], c=colors[i], marker=markers[j], s=50)

    # points = []
    # r = 100
    # for x, y in zip(x,y):
    #     points += [[x + np.cos(2*np.pi/6*s)*r, y+ np.sin(2*np.pi/6*s)*r] for s in range(7)]
    # points = np.array(points)

    # hull = ConvexHull(points)
    # x_hull = np.append(points[hull.vertices,0],
    #                    points[hull.vertices,0][0])
    # y_hull = np.append(points[hull.vertices,1],
    #                    points[hull.vertices,1][0])

    # dist = np.sqrt((x_hull[:-1] - x_hull[1:])**2 + (y_hull[:-1] - y_hull[1:])**2)
    # dist_along = np.concatenate(([0], dist.cumsum()))
    # spline, u = interpolate.splprep([x_hull, y_hull], 
    #                                 u=dist_along, s=0, per=1)
    # interp_d = np.linspace(dist_along[0], dist_along[-1], 50)
    # interp_x, interp_y = interpolate.splev(interp_d, spline)

    # plt.fill(interp_x, interp_y, '--', c=colors[i], alpha=0.2)
    plt.annotate(fibers[i], xy=labelpos[i], color=colors[i])

for j in range(len(resins)):
    plt.scatter([], [], c='k', marker=markers[j], label=resins[j], s=25)

plt.ylabel('Compliance (N.mm)')
plt.xlabel(r'$CO_{2,tot}$ (kg CO$_2$)')
plt.ylim((0,3200))
plt.xlim((2500,6500))
plt.legend(prop={"size":14})
plt.show()