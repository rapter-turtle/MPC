
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches

def plotTrackProj(simX):
    # load track

    v=simX[:,2]

    x = simX[:,3]
    y = simX[:,4]
    # plot racetrack map

    # #Setup plot
    # plt.figure()
    # plt.ylim(bottom=-20,top=20)
    # plt.xlim(left=0,right=40)
    # plt.ylabel('y[m]')
    # plt.xlabel('x[m]')

    # # Draw driven trajectory
    # heatmap = plt.scatter(x,y, c=v, cmap=cm.rainbow, edgecolor='none', marker='o',linewidth=1)
    # cbar = plt.colorbar(heatmap, fraction=0.035)
    # cbar.set_label("velocity in [m/s]")
    # ax = plt.gca()
    # ax.set_aspect('equal', 'box')

    # circle = patches.Circle((obs[0], obs[1]), obs[2], edgecolor='black', facecolor='black', linewidth=0.5, fill=True)
    # ax.add_patch(circle)


def plotRes(simX,simU,t):
    # plot results
    plt.figure()
    plt.subplot(4, 1, 1)
    plt.step(t, simU[:,0], color='r')
    # plt.step(t, simU[:,1], color='g')
    # plt.step(t, simU[:,2], color='b')
    plt.title('closed-loop simulation')
    plt.legend(['Tau_x_dot'])
    plt.ylabel('radian')
    plt.xlabel('t')
    plt.grid(True)
    plt.subplot(4, 1, 2)
    plt.plot(t, simX[:,6])
    plt.ylabel('Tau')
    plt.xlabel('t')
    plt.legend(['Tau'])
    plt.grid(True)
    plt.subplot(4, 1, 3)
    plt.plot(t, simX[:,0])
    plt.ylabel('u')
    plt.xlabel('t')
    plt.legend(['u'])
    plt.grid(True)
    plt.subplot(4, 1, 4)
    plt.plot(t, simX[:,3])
    plt.ylabel('x')
    plt.xlabel('t')
    plt.legend(['x'])
    plt.grid(True)