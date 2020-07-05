#!/usr/bin/env python
# coding: utf-8

# # Modeling: The Clogging Problem

# ## Problem setup

# Imports

# In[1]:

import math
import random
import numpy as np
import numpy.linalg as linalg
from scipy import interpolate
from scipy.integrate import ode

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.animation as animation
from matplotlib.patches import Circle

import pickle
import os
import argparse


# Constants

# In[2]:

#===Pipe system===
# len_m = 300 * 10 **(-6) #radius of pipe at mouth (m) microns
# len_c = 30 * 10 ** (-6) #radius of pipe at constriction (m)
# length = 300 * 10 ** (-6) #length of pipe
# scalef = 2 * 10**8 #scaling factor between the actual radius and the graphed radius
len_m = 300
len_c = 30
length = 800
scalef = 1/10
slope = (len_m-len_c)/length

#===Particles===
mass = 10**(-6) # mass of particle in kg
R_graphic = 5
R_actual = 10 * 10**(-6)

#===Physical Constants===
E = 10 ** 6  #start with super soft spheres
poisson = 0.3
alpha = 5/2
dyn_vis = 8.9 * 10 ** (-4) #dynamic viscosity (8.90 × 10−4 Pa*s for water)

#===Nondimentionalization Constants===
beta = 6 * math.pi * dyn_vis * R_actual
x0 = mass/beta
t0 = mass/beta

#===File OD consts===
PATH = "/home/aw/Documents/Clogging/clogging-research/outputs/"
TRAJ_PATH = "_trajectory.txt"
OVERVIEW_PATH = "_overview.txt"
TIME_PATH = "_time.txt"
ENERGY_PATH = "_energy.txt"


# ### Define the Streamfunction

# Create the coefficient matrix to solve for the stream function

# In[3]:

#Parameters
#n - size of the matrix
#x - step in x dir
#y - step in y dir
#returns - nxn coefficient matrix
def streamfuncCoeffsMatrix(n, x, y):
    coeffs = np.zeros((n**2,n**2))

    for j in range(n):
            coeffs[j][j] = 1
            coeffs[(n-1)**2+j][(n-1)**2+j] = 1

    #calculate the slope of the walls
    slope = (len_m - len_c)/length

    for i in range(n):
        for j in range(n):
            if (j==0 or j==n-1):
                coeffs[i*n+j][i*n+j] = 1
            elif (i<n/2 and (j<=slope*i or j>=(len_m*scalef-slope*i))):
                coeffs[i*n+j][i*n+j] = 1
            elif (i>=n/2 and (j>=(slope*i+len_c*scalef) or j<=(len_m*scalef-len_c*scalef-slope*i))):
                coeffs[i*n+j][i*n+j] = 1
            else:
                coeffs[i*n+j][i*n + (j-1)] = x**2/2/(x**2 + y**2)
                coeffs[i*n+j][i*n + (j+1)] = x**2/2/(x**2 + y**2)
                coeffs[i*n+j][((i-1)%n)*n + j] = y**2/2/(x**2 + y**2)
                coeffs[i*n+j][((i+1)%n)*n + j] = y**2/2/(x**2 + y**2)
                coeffs[i*n+j][i*n + j] = -1

    return coeffs


# calculate the boundary conditions for the streamfunction

#Parameters
#n - size of the matrix
#returns - 1d array with boundary conditions for the stream function
def getStreamFuncVals(n):

    #calculate the slope of the walls
    slope = (len_m - len_c)/length

    #create matrix with values the stream fn should equal
    vals = np.zeros((n**2))
    for j in range(n):
        vals[j*n+n-1] = 1

    for i in range(n):
        for j in range(n):
            if (j>=(slope*i+ (len_c)*scalef) and j>=(len_m*scalef-slope*i)):
                vals[i*n+j] = 1

    return vals


# Plot the boundary conditions of the stream function
def plotStreamFunVals(n):
    # n = int(len_m*scalef)
    vals = getStreamFuncVals(n)

    #convert vals to a square matrix for visualization
    vals_sq = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            vals_sq[i][j] = vals[j*n+i]

    plt.pcolor(vals_sq)
    plt.colorbar()
    plt.show()

# Calculate the streamfunction
def calcStreamFun(n):
    # n = int(len_m*scalef)

    #calculate streamfunction
    coeffs = streamfuncCoeffsMatrix(n,1,1)
    vals = getStreamFuncVals(n)

    func = linalg.solve(coeffs, vals)

    stream = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            stream[i][j] = func[i*n+j]

    return stream

def plotStreamFun(streamfun, n) :
    streamfun_graph = np.zeros((n,n))

    for i in range(n):
        for j in range(n):
            streamfun_graph[i][j] = streamfun[j][i]

    # plt.pcolor(stream, vmin = -1, vmax = 1)
    plt.pcolor(streamfun_graph)
    plt.title("Streamfunction")
    plt.colorbar()
    plt.show()



n = int(length*scalef)
streamfun = calcStreamFun(n)

plotStreamFunVals(n)
plotStreamFun(streamfun, n)

# Define the x and y derivatives of the streamfunction

#Params
#psi - streamfunction
#dx - change in x
#i - x value
#j - y value
#returns- approximation of d(psi)/dx
def dPsi_dx(psi, dx, i, j):
    return (psi[i+1][j]-psi[i-1][j])/2/dx

#Params
#psi - streamfunction
#dy - change in y
#i - x value
#j - y value
#returns- approximation of d(psi)/dy
def dPsi_dy(psi, dy, i, j):
    return (psi[i][j+1]-psi[i][j-1])/2/dy


# Plot the velocities in the x and y directions based on the streamfunction. vel_x = dPsi/dy, vel_y = - dPsi/dx

def getFluidVel(streamfun, nx, ny):
    u = np.zeros((nx,ny))
    v = np.zeros((nx,ny))

    for i in range(1,nx-1):
        for j in range(1,ny-1):
            u[i][j] = dPsi_dy(streamfun, .1, i, j)/2
            v[i][j] = -dPsi_dx(streamfun, .1, i, j)/2

    return u, v

def getFluidVelGraphic(streamfun, nx, ny):
    #pcolor graphs seem to plot the x values on the vertical axis so I manualy flipped these for visualization purposes
    u_graph = np.zeros((ny,nx))
    v_graph = np.zeros((ny,nx))

    for i in range(1,nx-1):
        for j in range(1,ny-1):
            u_graph[j][i] = dPsi_dy(streamfun, .1, i, j)
            v_graph[j][i] = -dPsi_dx(streamfun, .1, i, j)

    return u_graph, v_graph

def plotFluidVel(streamfun, nx, ny):

    u, v = getFluidVel(streamfun, nx, ny)
    u_graph, v_graph = getFluidVelGraphic(streamfun, nx, ny)

    plt.pcolor(v_graph)
    plt.colorbar()
    plt.title("Velocity in y direction")
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

    plt.pcolor(u_graph)
    plt.colorbar()
    plt.title("Velocity in x direction")
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

    for j in range(5):
        vels = [v[10+j*10][i] for i in range(ny)]
        label = (10+j*10)
        plt.plot(vels, label=label)

    plt.legend()
    plt.title("Velocity of fluid flow in y dir at different x values")
    plt.show()


    for j in range(6):
        vels = [u[i][25+j*1] for i in range(nx)]
        label = (25+j*1)
        plt.plot(vels, label=label)

    plt.legend()
    plt.title("Velocity of fluid flow in x dir at different y values")
    plt.show()

# plotFluidVel(streamfun, 80,30)

#
# # Calculate the velocity for any point in the field by averaging values from the velocity grid

#Parameters
# x: x position
# y: y position
# u: vel field in x dir
# v: vel field in y dir
#dx: step in x dir
#dy: step in y dir
#returns: vel in x and y dirs at (x,y)
def getVelGrid(x, y, u, v, dx, dy):
    grid = np.zeros((len(u), len(u),2))
    for i in range(len(u)):
        for j in range(len(u)):
            grid[i][j] = [i*dx, j*dy]

    #find indicies in the grid on each side of the given (x,y) position
    left = math.floor(x/dx)
    right = math.ceil(x/dx)
    top = math.ceil(y/dy)
    bottom = math.floor(y/dy)

    #TODO
    if (left == right):
        right = (right + 1)
    if (bottom == top):
        top = (top + 1)

    #use a weighted average of neighboring grid points to solve for the velocity at (x,y)
    xvel = (u[left][bottom] * (dx-abs(x-grid[left][bottom][0]))/dy + u[left][top] * (dx-abs(x-grid[left][top][0]))/dy
            + u[right][bottom] * (dx-abs(x-grid[right][bottom][0]))/dy + u[right][top] * (dx-abs(x-grid[right][top][0]))/dy)/4

    yvel = (v[left][bottom] * (dy-abs(y-grid[left][bottom][1]))/dy + v[left][top] * (dy-abs(y-grid[left][top][1]))/dy
            + v[right][bottom] * (dy-abs(y-grid[right][bottom][1]))/dy + v[right][top] * (dy-abs(y-grid[right][top][1]))/dy)/4

    return (xvel, yvel)


# # This can actually be done with scipy's interpolation method https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.interpolate.interp2d.html
#
#Parameters
# u: vel field in x dir
# v: vel field in y dir
#dx: step in x dir
#dy: step in y dir
#returns: vel functions in x and y dirs
def interpolateVelFn(u, v, dx, dy, nx, ny):
    x = np.arange(0, nx, 1)
    y = np.arange(0, ny, 1)

    velX = interpolate.interp2d(x, y, u.flatten(), kind='cubic')
    velY = interpolate.interp2d(x, y, v.flatten(), kind='cubic')

    return velX, velY

def plotVelFun(u, v):
    velX, velY = interpolateVelFn(u, v, 1, 1, length*scalef, len_m*scalef)

    #calculate the y velocity at a range of points
    yvels_20 = [velY(20, i/3) for i in range(n*3)]
    yvals = [i/3 for i in range(n*3)]

    plt.plot(yvals, yvels_20, label="interpolated values")
    plt.scatter(np.arange(0,n,1), v[20,:], label="grid values", color='red')
    plt.title("Velocity in the Y direction at x=20")
    plt.xlabel("Y position")
    plt.ylabel("Velocity in the Y direction")
    plt.legend()
    plt.show()

    #calculate the x velocity at a range of points
    xvels_20 = [velX(20, i/3) for i in range(n*3)]
    xvals = [i/3 for i in range(n*3)]

    plt.plot(xvals, xvels_20, label="interpolated values")
    plt.scatter(np.arange(0,n,1), u[20,:], label="grid values", color='red')
    plt.title("Velocity in the X direction at x=20")
    plt.xlabel("X position")
    plt.ylabel("Velocity in the X direction")
    plt.legend()
    plt.show()


# Graph the velocity field

def plotVelocityField(u, v, nx, ny):
    X = np.zeros((nx, ny))
    Y = np.zeros((nx, ny))

    for i in range(nx):
        for j in range(ny):
            X[i][j] = i
            Y[i][j] = j

    plt.quiver(X, Y, u, v, headaxislength=6)
    plt.title("Velocity Field for Pipe with Constriction")
    plt.gca().set_aspect('equal', adjustable='box')

    # plt.plot((0, 30, 60), (60, 35, 60), c="blue")
    # plt.plot((0, 30, 60), (0, 25, 0), c="blue")
    # plt.colorbar()
    plt.show()

# u, v = getFluidVel(streamfun, 80, 30)
# plotVelocityField(u, v, 80, 30)

#Run Simulation

# Calculate the forces on a particle

#Calculate the forces from a collision between 2 particles
#R - radius of the particle (in proportion to the system)
#xi, yi - position of particle 1
#vxi, vyi - velocity of particle 1
#xj, yj - position of particle 2
#vxj, vyj - velocity of particle 2
#returns - the force on particle i in the x and y directions and the potential
def calcCollision(R, xi, yi, vxi, vyi, xj, yj, vxj, vyj):

    distance = math.sqrt((xi-xj)**2+(yi-yj)**2)
    rij = (xi-xj, yi-yj)
    unit = unitVec(rij)

    #calculate potential
    Vij = 4*E/(3*(1-poisson)**2) * math.sqrt(R_actual/2) * ((1-distance/(2*R))**alpha)
    dVdr = 4*E/(3*(1-poisson)**2) * math.sqrt(R_actual/2) * (-alpha/2/R) * (1-distance/(2*R))**(alpha-1)
    Fx = - dVdr * unit[0]
    Fy = - dVdr * unit[1]

    return Fx, Fy, Vij

#calculate the force on a particle from the fluid
#x, y - position of particle
#vx, vy - velocity of particle
#u, v - the functions for calculating the velocity an the x and y directions
#returns - force of the fluid in x and y directions
def calcFluidForce(x, y, vx, vy, u, v):
    #force due to fluid flow
    uvel = u(x, y)
    vvel = v(x, y)

    Fx_fluid = beta * (uvel - vx)
    Fy_fluid = beta * (vvel - vy)

    return Fx_fluid, Fy_fluid

#Same as above but nondimentinalized, includes nondim constants tau and eta
def calcFluidForceNonDim(x, y, vx, vy, u, v):
    #force due to fluid flow
    uvel = u(x, y)
    vvel = v(x, y)

    Fx_fluid = beta * (uvel * t0**2 /x0 /mass - vx * t0 / mass)
    Fy_fluid = beta * (vvel * t0**2 /x0 /mass - vy * t0 / mass)

    return Fx_fluid, Fy_fluid

#Calculate the potential as the particle approaches a wall
#x, y - position of particle
#slope - slope of the wall
#tau, eta - nondimentionalization constants
#returns - forces in the x and y directions
def calcPotentialWall(x, y, slope):
    a = 50
    V = (math.e ** (-a*(y-x*slope)) + math.e **(a*(y-(len_m*scalef-x*slope))))
    Fx = -a*slope*V
    Fy=  a*(math.e ** (-a*(y-x*slope)) - math.e **(a*(y-(len_m*scalef-x*slope))))
    return Fx, Fy

# Misc geometry helper functions

#returns a unit vector in the same dir as some 2d vector v
def unitVec(v):
    v_mag = math.sqrt(v[0]**2 + v[1]**2)
    if (v_mag > 0):
        v_unit = (v[0]/v_mag, v[1]/v_mag)
    else:
        v_unit = (0, 0)
    return v_unit


# Plot the force of the wall as a function of y, calculated using the wall potential function

def plotWallForce():
    x = np.linspace(10,50,100)
    fx, fy = calcPotentialWall(20, x, slope)

    plt.plot(x, fx, label="x direction")
    plt.plot(x, fy, label = "y direction")
    plt.legend()
    plt.ylim(-100,100)
    plt.title("Wall force at x=20 accross y values")
    plt.xlabel("y")
    plt.ylabel("Force")


# Step the simulation: Calculate the derivatives at a given point

#Parameters
#t - timestep
#pos - array of the form (x0, y0, vx0, vy0, x1, ...)
#num_parts - total # particles in simulation
#R - radius of particle
#energy - 1d array to document the energy values at each step
#forces - array to document forces at each step
#         of the form: forces[i] = [[Fx_fluid, Fy_fluid], [Fx_wall, Fy_wall], [Fx_col, Fy_col]]
#vVel, yVel - the functions for calculating the velocity at a certain position
#
#Returns: derivatives of each value of the position array
#         [x0', y0', x0'', y0'', x1'...]
def stepODE(t, pos, num_parts, R, energy, forces, times, derivs, xVel, yVel):

    ddt = []
    V_col = 0
    times.append(t)

    for i in range(num_parts):
        x = pos[i*4]
        y = pos[i*4+1]
        vx = pos[i*4+2]
        vy = pos[i*4+3]

        #force due to fluid flow
        #TODO: testing w/o nondim
        Fx_fluid, Fy_fluid = calcFluidForceNonDim(x, y, vx, vy, xVel, yVel)

        #force due to collisions
        Fx_col = 0
        Fy_col = 0
        for j in range(num_parts):
            if j != i:
                distance = math.sqrt((x-pos[j*4])**2 + (y-pos[j*4+1])**2)

                #if the particles overlap
                if (distance < 2*R):
                    xj = pos[j*4]
                    yj = pos[j*4+1]
                    vxj = pos[j*4+2]
                    vyj = pos[j*4+3]

                    Fx, Fy, V = calcCollision(R, x, y, vx, vy, xj, yj, vxj, vyj)
                    Fx_col += Fx
                    Fy_col += Fy
                    V_col += V

        #force from wall potential
        wallX = 0
        wallY = 0
        if (x <= length*scalef/2):

            #calculate the point on the edge of the particle which is closest to the wall
            #the edge of the particle is what matters, not the center
            #this is a vector parpendicular to the wall
            if (y <= len_m/2*scalef):
                direction = unitVec((-1, 1/slope))
            else:
                direction = unitVec((-1, -1/slope))
            wallX, wallY = calcPotentialWall(x - direction[0]*R, y - direction[1]*R, slope)

        #document forces
        if i == 0:
            forces.append([[Fx_fluid, Fy_fluid], [wallX, wallY],[Fx_col, Fy_col]])

        Fx_net = Fx_fluid + wallX + Fx_col
        Fy_net = Fy_fluid + wallY + Fy_col
        ddt = ddt + [vx, vy, Fx_net, Fy_net] #TODO should the acceleration be F/m??

    derivs.append(ddt)

    energy.append(V_col)
    return ddt


# Run a simulation. Sets up and moniters the solver

# In[25]:


#Params
#num_parts - number of particles
#r - radius of particle
#dt - timestep
#tf - end time
#pos0 - inital positions/velocities
#returns - y - postiions over time
#          energy - energy at each timestep
#          forces - forces at each timestep
#          times - time at each iteration
#          derives - derivatives at each timestep
def runSim(num_parts, r, dt, tf, pos0, u, v):

    print("starting sim....")
    energy = []
    forces = []
    times = []
    derivs = []
    xvel, yvel = interpolateVelFn(u, v, 1, 1, length*scalef, len_m*scalef)

    solver = ode(stepODE).set_integrator('lsoda')
    solver.set_initial_value(pos0, 0).set_f_params(num_parts, r, energy, forces, times, derivs, xvel, yvel)
    y, t = [pos0], []
    while solver.successful() and solver.t < tf:
        t.append(solver.t)
        out = solver.integrate(solver.t+dt)
        y = np.concatenate((y, [out]), axis=0)

#     print(solver.get_return_code())

    print("finished sim...")
    return y, energy, forces, times, derivs


# Animate the trajectories of the particles

# get_ipython().run_line_magic('matplotlib', 'inline')

def generateAnim(y, r, n):
    xmax = length*scalef
    ymax = len_m*scalef
    X = np.linspace(0, xmax, int(length*scalef))
    Y = np.linspace(0, ymax, int(len_m*scalef))


    streamfun = calcStreamFun(n)
    u, v = getFluidVelGraphic(streamfun, int(length*scalef), int(len_m*scalef))
    #initialize figure and create a scatterplot
    fig, ax = plt.subplots()
    plt.xlim(0,xmax)
    plt.ylim(0,ymax)
    plt.pcolor(X, Y, u)
    plt.colorbar()
    plt.gca().set_aspect('equal', adjustable='box')

    plt.plot((0, xmax/2 +1, xmax), (ymax, scalef*(len_m+len_c)/2 + 1, ymax), c="blue")
    plt.plot((0, xmax/2+1, xmax), (0, scalef*(len_m-len_c)/2, 0), c="blue")

    circles = []

    def updateParticles_2(timestep):

        positions = []
        curr_num_parts = int(len(y[:][int(timestep*5)])/4)
        for i in range(curr_num_parts):
            positions.append((y[:][int(timestep)*5][0+i*4], y[:][int(timestep)*5][1+i*4]))

            if (i >= len(circles)):
                circles.append(Circle((0,0), r, color="white", fill=False))
                ax.add_artist(circles[i])

            circles[i].center = positions[-1]

        #hide circles that have exited the system
        for i in range(len(circles) -curr_num_parts):
            circles[curr_num_parts+i].center = (-5,-5)

        return circles,

    #create the animation
    ani = animation.FuncAnimation(fig, updateParticles_2, frames=int((len(y[:])-1)/5), interval=100)

    return ani


## Visualize the Results

# Run a simulation

# get_ipython().run_line_magic('matplotlib', 'inline')
#
# n = int(len_m*scalef)
# streamfun = calcStreamFun(80)
# u, v = getFluidVel(streamfun, 80, 30)
# #format: x_i, y_i, vx_i, vy_i, x_i+1...
# pos0 = []
# num_parts = 3
# for j in range(num_parts):
#     if j == 1:
#         x = 23.5
#     else:
#         x = 24
#     pos0 = pos0 + [x, 10 + j*5.05, 0, 0]
#
# # pos0 = pos0 + [18, 23, 0, 0]
# # pos0 = pos0 + [18, 27, 0, 0]
# # pos0 = pos0 + [15, 31, 0, 0]
# # pos0 = pos0 + [18, 35, 0, 0]
# # pos0 = pos0 + [18, 39, 0, 0]
#
# r = .7
# trajectory, energy, forces, t, der = runSim(num_parts, r, 0.1, 45, pos0, u, v)
#
# ani = generateAnim(trajectory, num_parts, r, streamfun, n)
# # ani.show()
# plt.show()
# ani.save('clog.06242020_temporary.gif', writer='imagemagick')

# HTML(ani.to_jshtml())
#
#
# # Plot the forces from differnt sources (fluid flow, the wall, collisions) as a function of time
#
# # In[62]:
#
#
# fcx = []
# fwx = []
# ffx = []
#
# for i in range(len(forces)):
#     fcx.append(forces[i][2][0])
#     fwx.append(forces[i][1][0])
#     ffx.append(forces[i][0][0])
#
# plt.plot(t, fwx, label="wall")
# plt.plot(t, fcx, label="collision net")
# # plt.plot(t)
# plt.plot(t, ffx, label="fluid")
# plt.legend()
# plt.title("Forces over time - x direction")
# plt.ylim(-2,2)
# # plt.savefig("clog_temp_06242020_forces.png")
#
#
# # Plot the total energy of the system over time
#
# # In[63]:
#
#
# plt.plot(t, energy)
# plt.title("Energy over time")
# plt.ylim(0,.10)
# plt.show()
# # plt.savefig("clog_temp_06242020_energy.png")
#
# plt.plot(t)
# plt.title("Time at each step")
# plt.show()
#
#
# # Single particle behavior
#
# # In[507]:
#
#
# get_ipython().run_line_magic('matplotlib', 'inline')
#
# #format: x_i, y_i, vx_i, vy_i, x_i+1...
# pos0 = []
# num_parts = 1
# for i in range(num_parts):
#     if i == 1:
#         x = 18
#     else:
#         x = 25
#     pos0 = pos0 + [x, 27+ 5*i, -4, 4]
#
# num_particles = 1
# r = 1.5
# trajectory, energy, forces, t, der = runSim(num_particles, r, 0.1, 30, pos0)
#
# ani = generateAnim(trajectory, num_particles, r)
# HTML(ani.to_jshtml())
#
#
# # In[508]:
#
#
# fcx = []
# fwx = []
# ffx = []
#
# for i in range(len(forces)):
#     fcx.append(forces[i][2][0])
#     fwx.append(forces[i][1][0])
#     ffx.append(forces[i][0][0])
#
# plt.plot(t, fwx, label="wall")
# plt.plot(t, fcx, label="collision net")
# plt.plot(t, ffx, label="fluid")
# plt.legend()
# plt.title("Forces over time")
# # plt.ylim(-1,1)
#
#
# # Plot streamlines of particles in the fluid flow
#
# # ### Testing: trajectories of individual particles at various points
#
# # In[509]:
#
#
# R = 1
# trajectory1, energy1, forces1, times1, d1 = runSim(1, R, 0.1, 50, [22,25,1e-8,1e-8])
# trajectory2, energy2, forces2, times2, d2 = runSim(1, R, 0.1, 50, [22,30,0,0])
# trajectory3, energy3, forces3, times3, d3 = runSim(1, R, 0.1, 50, [22,35,1e-8,1e-8])
#
# #initialize figure and create a scatterplot
# fig, ax = plt.subplots()
# plt.pcolor(Y, X, u_graph)
# plt.colorbar()
#
# plt.plot(trajectory1[:,0], trajectory1[:,1], color="white")
# plt.plot(trajectory2[:,0], trajectory2[:,1], color="white")
# plt.plot(trajectory3[:,0], trajectory3[:,1], color="white")
#
# plt.title("Particle trajectories over time")
# plt.show()
#
#
# # In[510]:
#
#
# plt.plot(times1, label="sucess")
# plt.plot(times2, label="error")
# plt.legend()
# plt.xlabel("iteration number")
# plt.ylabel("total time elapsed")
# plt.title("Time vs Step Number")
#
#
# # In[511]:
#
#
# plt.plot([row[0] for row in d1], label="x'")
# plt.plot([row[1] for row in d1], label="y'")
# plt.plot([row[2] for row in d1], label="x''")
# plt.plot([row[3] for row in d1], label="y''")
# plt.legend()
# plt.xlabel("iteration number")
# plt.ylabel("derivative")
# plt.title("Derivatives over Time: Particle off centerline")
# plt.show()
#
# plt.plot([row[0] for row in d2], label="x'")
# plt.plot([row[1] for row in d2], label="y'")
# plt.plot([row[2] for row in d2], label="x''")
# plt.plot([row[3] for row in d2], label="y''")
# plt.legend()
# plt.xlabel("iteration number")
# plt.ylabel("derivative")
# plt.title("Derivatives over Time: Particle on centerline")
# plt.show()
#
#
# # These graphs show that particles on and off the centerline now both have successful models. Previously, the particle on the centerline would throw an error, but this bug has been resolved and both now have reasonable timesteps and derivatives
#
# # In[512]:
#
#
# fcx = []
# fwx = []
# ffx = []
#
# for i in range(len(forces1)):
#     fcx.append(forces1[i][2][0])
#     fwx.append(forces1[i][1][0])
#     ffx.append(forces1[i][0][0])
#
# plt.plot(fwx, label="wall")
# plt.plot(fcx, label="collision net")
# plt.plot(ffx, label="fluid")
# plt.legend()
# plt.title("Forces over time")
# # plt.ylim(-1,1)
#
#### Testing: adding/removing particles

#Randomly introduce a new particle that doesn't overlap with other existing particles
# r: radius
# existing_particles: list of positions of other particles (x1, y1, vx1, vy1, x2 ...)
# return pos of new particle, start at 0,0 if couldn't find a pos in a reasonable # attempts

def randStartingPt(r, existing_particles, num_particles):

    x = 5
    no_col = False
    count = 0

    while (no_col == False and count < 30):

        y = r + slope*x + 1 + random.random()*(len_m*scalef - 2*slope*x - 2*r -2)

        no_col = True

        for i in range(num_particles) :
            xi = existing_particles[i*4]
            yi = existing_particles[i*4+1]
            distance = math.sqrt((x-xi)**2+(y-yi)**2)
            if (distance < 2*r): no_col = False

        count += 1

    if (no_col):
        print([x,y])
        return [x, y, 0, 0]

    return []


#Params
#num_parts - number of particles
#r - radius of particle
#dt - timestep
#tf - end time
#returns - y - postiions over time
#          energy - energy at each timestep
#          forces - forces at each timestep
#          times - time at each iteration
#          derives - derivatives at each timestep
def runSimAdditive(name, num_parts, r, dt, tf):

    energy = []
    forces = []
    times = []
    derivs = []

    streamfun = calcStreamFun(80)
    u, v = getFluidVel(streamfun, int(length*scalef), int(len_m*scalef))
    xvel, yvel = interpolateVelFn(u, v, 1, 1, int(length*scalef), int(len_m*scalef))

    current_num_parts = 0

    #format: x_i, y_i, vx_i, vy_i, x_i+1...
    pos0 = []
    pos0 = pos0 + randStartingPt(r, [], current_num_parts)
    current_num_parts += 1

    solver = ode(stepODE).set_integrator('lsoda')
    solver.set_initial_value(pos0, 0).set_f_params(current_num_parts, r, energy, forces, times, derivs, xvel, yvel)
    y, t = [pos0], []
    for i in range(num_parts):

        while solver.successful() and (solver.t < (tf * ((i+1)/num_parts))):
            t.append(solver.t)
            out = solver.integrate(solver.t+dt)
            y = y+[out.tolist()]

        out = out.tolist()
        #rm any particles that have exited the system
        # for j in range(current_num_parts):
        #     if (out[j] >= length*scalef -1):
        #         out = out[:j] + out[j+4:]
        #         current_num_parts -= 1


        curr_t = solver.t
        new_part = randStartingPt(r, out, current_num_parts)

        pos = out + new_part
        if (new_part != []) : current_num_parts += 1

        solver = ode(stepODE).set_integrator('lsoda')
        solver.set_initial_value(pos, curr_t).set_f_params(current_num_parts, r, energy, forces, times, derivs, xvel, yvel)

    writeData(name, y, times, energy, False, False, 0, r)

    return y, energy, forces, times, derivs


def writeData(name, traj, t, energy, clog, metastable, clog_t, r):

    path = PATH + name + "/"

    try:
        os.mkdir(path)
    except OSError:
        print ("failed to create directory %s" % path)
    else:
        print ("created directory %s " % path)

    with open(path + name + TRAJ_PATH, 'wb') as f:
        pickle.dump(traj, f)

    with open(path + name + TIME_PATH, 'wb') as f:
        pickle.dump(t, f)

    with open(path + name + ENERGY_PATH, 'wb') as f:
        pickle.dump(energy, f)

    plt.plot(t, energy)
    plt.savefig(path+name+"_energyFig.png")

    file = open(path+name+OVERVIEW_PATH, "w+")
    file.write(str(r)) #radius, needs to be easily readable
    file.write("\nTotal Time: %f" % t[-1])
    file.write("\nClog: %s" % clog)
    file.write("\nMetastable: %s" % metastable)
    file.write("\nClog occured at: %f" % clog_t)
    file.close()

def makeAnimFromFile(name):

    path = PATH + name + "/"

    n = int(length*scalef)

    with open(path + name + TRAJ_PATH, 'rb') as f:
        traj = pickle.load(f)

    file = open(path+name+OVERVIEW_PATH, "r")
    r = file.readline()
    file.close
    r = float(r.splitlines()[0])

    ani = generateAnim(traj, r, n)
    plt.show()


parser = argparse.ArgumentParser()
parser.add_argument("name",
                    help="name to store output under/fetch output from")
parser.add_argument("-r", "--radius", type=float, default = 1,
                    help="particle radius")
parser.add_argument("-t", "--time", type=float, default=1000,
                    help="total time")
parser.add_argument("-s", "--step", type=float, default=0.1,
                    help="timestep")
parser.add_argument("-run", "--run-sim", dest="run_sim", action="store_true",
                    help="run a simulation")
parser.add_argument("-nrun", "--no-sim", dest="run_sim", action="store_false",
                    help="dont run a simulation")
parser.set_defaults(run_sim=True)
parser.add_argument("-a", "--animate", dest="animate", action="store_true",
                    help="create animation from stored output")
parser.add_argument("-na", "--no-animate", dest="animate", action="store_false",
                    help="dont create animation from stored output")
parser.set_defaults(animate=False)

#TODO expand
def runExtendedSim(name, radius, timestep, timef):
    runSimAdditive(name, int(timef/5), radius, timestep, timef)

args = parser.parse_args()

if (args.run_sim):
    runExtendedSim(name, args.radius, args.step, args.time)
elif (args.animate):
    makeAnimFromFile(args.name)
else:
    print("No task assigned.\nEnding program")
