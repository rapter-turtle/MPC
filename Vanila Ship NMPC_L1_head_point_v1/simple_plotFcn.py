
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import math

def plotTrackProj(simX):
    # load track

    v=simX[:,0]

    x = simX[:,3]
    y = simX[:,4]
    # plot racetrack map

    # #Setup plot
    plt.figure()
    plt.ylim(bottom=-20,top=20)
    plt.xlim(left=-10,right=400)
    plt.ylabel('y[m]')
    plt.xlabel('x[m]')

    # Draw driven trajectory
    heatmap = plt.scatter(x,y, c=v, cmap=cm.rainbow, edgecolor='none', marker='o',linewidth=1)
    cbar = plt.colorbar(heatmap, fraction=0.035)
    cbar.set_label("velocity in [m/s]")
    ax = plt.gca()
    # ax.set_aspect('equal', 'box')
    ax.grid(True)
    # circle = patches.Circle((obs[0], obs[1]), obs[2], edgecolor='black', facecolor='black', linewidth=0.5, fill=True)
    # ax.add_patch(circle)


def plotRes(simX,simU,t, sim_l1_con, sim_param, sim_x_estim, real):
    # plot results
    N = 5
    M = 2
    plt.figure()
    plt.subplot(N, M, 1)
    plt.step(t,sim_x_estim[:,3] - real[:,3] , color='r')
    plt.ylabel('estim error Vy')
    plt.xlabel('t')
    plt.grid(True)
    plt.subplot(N, M, 2)
    plt.plot(t, sim_x_estim[:,1] - real[:,1])
    plt.ylabel('estim error y')
    plt.xlabel('t')
    plt.grid(True)
    plt.subplot(N, M, 3)
    plt.plot(t, real[:,1])
    plt.plot(t, sim_x_estim[:,1])
    plt.ylabel('y distance ')
    plt.xlabel('t')
    plt.grid(True)
    plt.subplot(N, M, 4)
    plt.plot(t, real[:,1])
    plt.ylabel('estimated x error ')
    plt.xlabel('t')
    plt.grid(True)
    plt.subplot(N, M, 5)
    plt.plot(t, sim_param[:,0] )
    plt.ylabel('param x ')
    plt.xlabel('t')
    plt.grid(True)
    plt.subplot(N, M, 6)
    plt.plot(t, sim_param[:,1])
    plt.plot(t, simX[:,9])
    plt.ylabel('param y ')
    plt.xlabel('t')
    plt.grid(True)    
    plt.subplot(N, M, 7)
    plt.plot(t, sim_l1_con[:,0])
    plt.ylabel('l1 x ')
    plt.xlabel('t')
    plt.grid(True)
    plt.subplot(N, M, 8)
    plt.plot(t, sim_l1_con[:,1])
    plt.ylabel('l1 y ')
    plt.xlabel('t')
    plt.grid(True)
    plt.subplot(N, M, 9)
    plt.plot(t, simX[:,6] )
    plt.ylabel('Tau x mpc')
    plt.xlabel('t')
    plt.grid(True)
    plt.subplot(N, M, 10)
    plt.plot(t, simX[:,7] )
    plt.ylabel('Tau y mpc')
    plt.xlabel('t')
    plt.grid(True)       



def current_plot(x, ax, t, state):
    l = 3.5
    # Extract current state
    # current_x = x[0][3] - l*math.cos(state[5]) + state[8]
    # current_y = x[0][4] - l*math.sin(state[5]) + state[9]
    # current_psi = x[0][5] + state[10]
    current_x = state[3]
    current_y = state[4]
    current_psi = state[5]

    # Plot ship's shape polygon according to the current state
    ship_length = 7  # Example length of the ship
    ship_width = 0.1  # Example width of the ship

    # Define ship shape vertices
    ship_vertices = np.array([[current_x - 0.5 * ship_length * np.cos(current_psi) - 0.5 * ship_width * np.sin(current_psi),
                                current_y - 0.5 * ship_length * np.sin(current_psi) + 0.5 * ship_width * np.cos(current_psi)],
                               [current_x + 0.5 * ship_length * np.cos(current_psi) - 0.5 * ship_width * np.sin(current_psi),
                                current_y + 0.5 * ship_length * np.sin(current_psi) + 0.5 * ship_width * np.cos(current_psi)],
                               [current_x + 0.8 * ship_length * np.cos(current_psi),
                                current_y + 0.8 * ship_length * np.sin(current_psi)], 
                               [current_x + 0.5 * ship_length * np.cos(current_psi) + 0.5 * ship_width * np.sin(current_psi),
                                current_y + 0.5 * ship_length * np.sin(current_psi) - 0.5 * ship_width * np.cos(current_psi)],
                               [current_x - 0.5 * ship_length * np.cos(current_psi) + 0.5 * ship_width * np.sin(current_psi),
                                current_y - 0.5 * ship_length * np.sin(current_psi) - 0.5 * ship_width * np.cos(current_psi)]])

    # Plot ship's shape polygon
    ship_polygon = Polygon(ship_vertices, closed=True, edgecolor='b', facecolor='b')
    ax.add_patch(ship_polygon)

    # Plot the trajectory of (x, y) for the prediction horizon
    predicted_horizon_x = [sub_list[3] for sub_list in x]
    predicted_horizon_y = [sub_list[4] for sub_list in x]
    ax.plot(predicted_horizon_x -np.cos(state[5]) + state[8], predicted_horizon_y -np.sin(state[5])  + state[9], 'r-', label='Predicted Horizon')
    # Add labels and legend
    ax.grid(True)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.legend()
    # ax.axis('equal')
    ax.text(0.5, 1.02, f'Time: {t}', fontsize=12, ha='center', va='bottom', transform=ax.transAxes)
    plt.ylim(bottom=-20,top=20)
    plt.xlim(left=-10,right=200)
    plt.draw() 
    plt.pause(0.001)

    
    ax.clear() 
    

def current_estim_plot(current,t):
    # plot results
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(t, current[:,0])
    plt.ylabel('current X')
    plt.xlabel('t')
    plt.grid(True)
    plt.subplot(2, 1, 2)
    plt.plot(t, current[:,1])
    plt.ylabel('current Y')
    plt.xlabel('t')
    plt.grid(True)    