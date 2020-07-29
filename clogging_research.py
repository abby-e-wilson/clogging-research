#Modeling: The Clogging Problem
#Abby Wilson
#Summer 2020

# Imports
import math
import random
import numpy as np
import numpy.linalg as linalg
from scipy import interpolate
from scipy.integrate import ode

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm
import matplotlib.animation as animation
from matplotlib.patches import Circle

import os
import pickle
import argparse

# Constants

#===Pipe system===
# len_m = 300 * 10 **(-6) #radius of pipe at mouth (m) microns
# len_c = 30 * 10 ** (-6) #radius of pipe at constriction (m)
# length = 300 * 10 ** (-6) #length of pipe
# scalef = 2 * 10**8 #scaling factor between the actual radius and the graphed radius
len_m = 600
len_c = 150
length = 600
scalef = 1/10
slope = (len_m-len_c)/length

#===Particles===
mass = 10**(-6) # mass of particle in kg
R_actual = 10 * 10**(-6)
inertia = 2/5*mass*R_actual**2

#===Physical Constants===
E = 10 ** 6  #start with super soft spheres
poisson = 0.3
alpha = 5/2
dyn_vis = 8.9 * 10 ** (-4) #dynamic viscosity (8.90 × 10−4 Pa*s for water)
density = 997 #kg/m^3, for water
maxV = .2 #max fluid velocity

#===Nondimentionalization Constants===
beta = 6 * math.pi * dyn_vis * R_actual
x0 = mass/beta
t0 = mass/beta

#===File OD consts===
PATH = "/home/aw/Documents/Clogging/clogging-research/outputs/"
PATH_IN = "/home/aw/Documents/Clogging/clogging-research/inputs/"
TRAJ_PATH = "_trajectory.txt"
OVERVIEW_PATH = "_overview.txt"
TIME_PATH = "_time.txt"
ENERGY_PATH = "_energy.txt"
STREAM = "_stream.txt"

#=== Define the Streamfunction ===

# Create the coefficient matrix to solve for the stream function

#Parameters
#n - size of the matrix
#x - step in x dir
#y - step in y dir
#returns - nxn coefficient matrix
def streamfuncCoeffsMatrixBiharmonic(n, x, y):
    coeffs = np.zeros((n**2,n**2))

    for i in range(n):
        for j in range(n):

            #BD Conditions (set velocity values)
            if (j==0 or j==n-1):
                coeffs[i*n+j][i*n+j] = 1
            elif (i<n/2 and (j<=slope*i or j>=(len_m*scalef-slope*i))):
                coeffs[i*n+j][i*n+j] = 1
            elif (i>=n/2 and (j>=(slope*i+len_c*scalef) or j<=(len_m*scalef-len_c*scalef-slope*i))):
                coeffs[i*n+j][i*n+j] = 1

            #Body Conditions
            else:
                #Finite difference method for biharmonic equation

                if (j+2 < n):
                    coeffs[i*n+j][i*n + (j+2)] = 1/y**4
                    coeffs[i*n+j][i*n + (j+1)] = -2/x**2/y**2 - 4/y**4
                else:
                    #if off the edge, compensate by swapping j+2 with j+1, as the bd should have constant value
                    coeffs[i*n+j][i*n + (j+1)] = -2/x**2/y**2 - 4/y**4 + 1/y**4
                if (j-2 >= 0 ):
                    coeffs[i*n+j][i*n + (j-2)] = 1/y**4
                    coeffs[i*n+j][i*n + (j-1)] = -2/x**2/y**2 - 4/y**4
                else:
                    #if off the edge, compensate by swapping j+2 with j+1, as the bd should have constant value
                    coeffs[i*n+j][i*n + (j-1)] = -2/x**2/y**2 - 4/y**4 + 1/y**4

                coeffs[i*n+j][((i+2)%n)*n + j] = 1/x**4
                coeffs[i*n+j][((i-2)%n)*n + j] = 1/x**4

                coeffs[i*n+j][((i+1)%n)*n + (j+1)] = 1/x**2/y**2
                coeffs[i*n+j][((i-1)%n)*n + (j+1)] = 1/x**2/y**2
                coeffs[i*n+j][((i+1)%n)*n + (j-1)] = 1/x**2/y**2
                coeffs[i*n+j][((i-1)%n)*n + (j-1)] = 1/x**2/y**2

                coeffs[i*n+j][((i+1)%n)*n + j] = -2/x**2/y**2 - 4/x**4
                coeffs[i*n+j][((i-1)%n)*n + j] = -2/x**2/y**2 - 4/x**4

                coeffs[i*n+j][i*n + j] = 4/x**2/y**2 + 6/x**4 + 6/y**4

    return coeffs

#Calculate the boundary conditions for the streamfunction

#Parameters
#n - size of the matrix
#returns - 1d array with boundary conditions for the stream function
def getStreamFuncVals(n):

    #create matrix with values the stream fn should equal
    vals = np.zeros((n**2))

    for i in range(n):
        for j in range(n):
            if (j == n-1 or j>=(slope*i+ (len_c)*scalef) and j>=(len_m*scalef-slope*i)):
                #upper bd
                vals[i*n+j] = 1
            elif (j<=(slope*i) and j<=(len_m*scalef-len_c*scalef-slope*i)):
                #lower bd
                vals[i*n+j] = 0
            elif (j!= 0 and j!= n-1):
                #body
                vals[i*n+j] = 0

    return vals


# Plot the boundary conditions of the stream function
def plotStreamFunVals(n):
    # n = int(len_m*scalef)
    vals = getStreamFuncVals(n)

    #convert vals to a square matrix for visualization
    vals_sq = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            print(i,j)
            vals_sq[i][j] = vals[j*n+i]

    plt.pcolor(vals_sq)
    plt.colorbar()
    plt.show()


# Calculate the streamfunction
def calcStreamFun(n):
    # n = int(len_m*scalef)

    #calculate streamfunction
    coeffs = streamfuncCoeffsMatrixBiharmonic(n,1,1)
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

    plt.pcolor(streamfun_graph)

    xmax = length* scalef
    ymax = len_m*scalef
    plt.plot((0, xmax/2, xmax), (ymax, scalef*(len_m+len_c)/2, ymax), c="blue")
    plt.plot((0, xmax/2, xmax), (0, scalef*(len_m-len_c)/2, 0), c="blue")
    plt.grid(b=True, which='minor', color='#666666', linestyle='-')
    plt.title("Streamfunction")
    plt.colorbar()
    plt.xlabel("<--- length --->")
    plt.ylabel("<--- Pipe Inlet --->")
    # plt.savefig("streamfun_corr.png")
    plt.show()

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
    maxval = 0

    for i in range(1,nx-1):
        for j in range(1,ny-1):
            u[i][j] = dPsi_dy(streamfun, .1, i, j)
            v[i][j] = -dPsi_dx(streamfun, .1, i, j)
            mag = math.sqrt(u[i][j]**2+v[i][j]**2)
            if mag > maxval : maxval = mag
    u = u/maxval * maxV
    v = v/maxval * maxV
    print(maxval, maxV)
    return u, v

def vorticity(u, v,n ):
    w = np.zeros((n,n))
    for i in range(1,n-1):
        for j in range(1,n-1):
            w[i][j] = dPsi_dx(v,1,i,j) - dPsi_dy(u,1,i,j)

    plt.pcolor(w)
    plt.colorbar()
    plt.show()

    return w

# streamfun = calcStreamFun(60)
# u, v = getFluidVel(streamfun, 60,60)
# vorticity(u,v,60)

def writeVelocity(streamfun):

    try:
        os.mkdir(PATH_IN)
    except OSError:
        print ("failed to create directory %s" % PATH_IN)
    else:
        print ("created directory %s " % PATH_IN)

    with open(PATH_IN + STREAM, 'wb') as f:
        pickle.dump(streamfun, f)

def readVelocity():

    with open(PATH_IN  + STREAM, 'rb') as f:
        streamfun = pickle.load(f)

    return streamfun


def plotStreamFunWProf(streamfun, n) :
    streamfun_graph = np.zeros((n,n))
    X = np.zeros((n,n))
    Y = np.zeros((n,n))

    for i in range(n):
        for j in range(n):
            X[i][j] = i
            Y[i][j] = j
            streamfun_graph[i][j] = streamfun[j][i]

    bounds = np.linspace(np.amin(streamfun), np.amax(streamfun), 10)
    norm = colors.BoundaryNorm(boundaries=bounds, ncolors=256)
    # plt.pcolormesh(streamfun_graph, norm=norm)
    bd = np.linspace(0,1,10)
    plt.contour(X, Y, streamfun, levels=bd[1:-1])
    plt.gca().set_aspect('equal', adjustable='box')


    xmax = length* scalef
    ymax = len_m*scalef
    plt.plot((0, xmax/2, xmax), (ymax, scalef*(len_m+len_c)/2, ymax), c="blue")
    plt.plot((0, xmax/2, xmax), (0, scalef*(len_m-len_c)/2, 0), c="blue")
    plt.grid(b=True, which='minor', color='#666666', linestyle='-')
    plt.title("Streamlines of the Streamfunction")
    plt.xlabel("<--- length --->")
    plt.ylabel("<--- Pipe Inlet --->")
    # plt.colorbar()
    # plt.plot(vels, x, color="white")

    plt.savefig("streamfun_corr_lines.png")
    plt.show()

# n = int(length*scalef)
# # #
# plotStreamFunVals(60)
# streamfun = calcStreamFun(60)
# plotStreamFunWProf(streamfun, 60)
# writeVelocity(streamfun)

def getFluidVelGraphic(streamfun, nx, ny):
    #pcolor graphs seem to plot the x values on the vertical axis so I manualy flipped these for visualization purposes
    u_graph = np.zeros((ny,nx))
    v_graph = np.zeros((ny,nx))
    maxval = 0

    for i in range(1,nx-1):
        for j in range(1,ny-1):
            u_graph[j][i] = dPsi_dy(streamfun, .1, i, j)
            v_graph[j][i] = -dPsi_dx(streamfun, .1, i, j)
            mag = math.sqrt(u_graph[j][i]**2+v_graph[j][i]**2)
            if mag > maxval : maxval = mag
    u_graph = u_graph/maxval * maxV
    v_graph = v_graph/maxval * maxV

    return u_graph, v_graph


def plotFluidVel(streamfun, nx, ny):

    u, v = getFluidVel(streamfun, nx, ny)
    u_graph, v_graph = getFluidVelGraphic(streamfun, nx, ny)

    mag = np.zeros((nx,ny))
    for i in range(nx):
        for j in range(ny):
            mag[i][j] = math.sqrt(u_graph[i][j]**2 + v_graph[i][j]**2)

    plt.pcolor(v_graph)
    plt.colorbar()
    plt.title("Velocity in y direction")
    plt.gca().set_aspect('equal', adjustable='box')
    # plt.grid(b=True, which='major', color='#666666', linestyle='-')

    plt.xlabel("<--- length --->")
    plt.ylabel("<--- Pipe Inlet --->")
    # plt.savefig("streamfun_corr_y.png")
    plt.show()

    plt.pcolor(u_graph)
    plt.colorbar()
    plt.title("Velocity in x direction")
    plt.gca().set_aspect('equal', adjustable='box')
    # plt.grid(b=True, which='major', color='#666666', linestyle='-')
    plt.xlabel("<--- length --->")
    plt.ylabel("<--- Pipe Inlet --->")
    # plt.savefig("streamfun_corr_x.png")
    plt.show()

    plt.pcolor(mag)
    plt.colorbar()
    plt.title("Velocity magnitude")
    plt.gca().set_aspect('equal', adjustable='box')
    # plt.grid(b=True, which='major', color='#666666', linestyle='-')
    plt.show()

    for j in range(5):
        vels = [v[10+j*10][i] for i in range(ny)]
        label = (10+j*10)
        plt.plot(vels, label=label)

    plt.legend()
    plt.title("Velocity of fluid flow in y dir at different x values")
    plt.show()


    for j in range(12):
        vels = [u[10+j*3][i] for i in range(nx)]
        label = "x=" + str(10+j*3)
        plt.plot(vels, label=label)

    print("average: " + str(np.average(vels)))
    plt.legend()
    plt.title("Velocity of fluid flow in x dir at different x values")
    plt.xlabel("Velocity profile along pipe cross section")
    plt.ylabel("Velocity (m/s)")
    plt.savefig("streamfun_corr_velprof.png")
    plt.show()

streamfun = calcStreamFun(60)
# plotFluidVel(streamfun, 60,60)

# streamfun = readVelocity()
# plotFluidVel(streamfun, int(length*scalef), int(len_m*scalef))
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
    velX, velY = interpolateVelFn(u, v, .1, .1, length*scalef, len_m*scalef)

    #calculate the y velocity at a range of points
    yvels_20 = [velY(40, i/3) for i in range(n*3)]
    yvals = [i/3 for i in range(n*3)]

    plt.plot(yvals, yvels_20, label="interpolated values")
    plt.scatter(np.arange(0,30,1), v[40,:], label="grid values", color='red')
    plt.title("Velocity in the Y direction at x=20")
    plt.xlabel("Y position")
    plt.ylabel("Velocity in the Y direction")
    plt.legend()
    plt.show()

    #calculate the x velocity at a range of points
    xvels_20 = [velX(40, i/3) for i in range(n*3)]
    xvals = [i/3 for i in range(n*3)]

    plt.plot(xvals, xvels_20, label="interpolated values")
    plt.scatter(np.arange(0,30,1), u[40,:], label="grid values", color='red')
    plt.title("Velocity in the X direction at x=20")
    plt.xlabel("X position")
    plt.ylabel("Velocity in the X direction")
    plt.legend()
    plt.show()

# u, v = getFluidVel(streamfun, 80, 30)
# plotVelFun(u,v)

# Graph the velocity field

def plotVelocityField(u, v, nx, ny):
    X = np.zeros((nx, ny))
    Y = np.zeros((nx, ny))
    U = np.zeros((nx, ny))
    V = np.zeros((nx, ny))
    mag = np.zeros((nx, ny))

    #
    for i in range(30):
        for j in range(30):
            # if (i!=0 or j!=0):
            X[i*2][j*2] = i*2
            Y[i*2][j*2] = j*2
            U[i*2][j*2] = u[i*2][j*2]
            V[i*2][j*2] = v[i*2][j*2]
            X[i*2+1][j*2+1]=-10
            # u[i*2][j*2] = math.sqrt(u[i*2][j*2]**2+v[i*2][j*2]**2)
    #
    n=60
    for i in range(60):
        for j in range(60):

            if (j == n-1 or j>=(slope*i+ (len_c)*scalef) and j>=(len_m*scalef-slope*i)):
                #upper bd
                X[i][j] = -10
                Y[i][j] = 0
            elif (j<=(slope*i) and j<=(len_m*scalef-len_c*scalef-slope*i)):
                X[i][j] = -10
                Y[i][j] = 0
            elif (i%2==0 and j%2==0):
                X[i][j] = i
                Y[i][j] = j
                U[i][j] = u[i][j]
                V[i][j] = v[i][j]
                mag[i][j] = math.sqrt(u[i][j]**2+v[i][j]**2)
            else:
                X[i][j]=-10

    # plt.pcolor(U)
    # plt.colorbar()
    plt.quiver(X, Y, u, v, headaxislength=6,color='black')
    plt.xlim(0,60)
    plt.title("Velocity Field for Pipe with Constriction")
    plt.gca().set_aspect('equal', adjustable='box')

    xmax = length* scalef
    ymax = len_m*scalef
    plt.plot((0, xmax/2, xmax), (ymax, scalef*(len_m+len_c)/2, ymax), c="blue")
    plt.plot((0, xmax/2, xmax), (0, scalef*(len_m-len_c)/2, 0), c="blue")
    # plt.colorbar()
    # plt.savefig("corr_velfield.png")
    plt.show()

# u, v = getFluidVel(streamfun, 60, 60)
# plotVelocityField(u, v, 60, 60)

#Run Simulation

# Calculate the forces on a particle


#returns a unit vector in the same dir as some 2d vector v
def unitVec(v):
    v_mag = math.sqrt(v[0]**2 + v[1]**2)
    if (v_mag > 0):
        v_unit = (v[0]/v_mag, v[1]/v_mag)
    else:
        v_unit = (0, 0)
    return v_unit

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


#Calculate the forces from a collision between 2 particles
#R - radius of the particle (in proportion to the system)
#xi, yi - position of particle 1
#vxi, vyi - velocity of particle 1
#xj, yj - position of particle 2
#vxj, vyj - velocity of particle 2
#returns - the force on particle i in the x and y directions and the potential
def calcCollisionAd(R, xi, yi, xj, yj):

    distance = math.sqrt((xi-xj)**2+(yi-yj)**2)
    rij = (xi-xj, yi-yj)
    unit = unitVec(rij)

    # a0 = (9*math.pi*R**2/E)**(1/9)
    # a0=0.01*R
    # d = (2*R - distance)/a0
    #
    # #calculate potential
    # Vij = 4*E/(3*(1-poisson)**2) * math.sqrt(R_actual/2) * ((1-distance/(2*R))**alpha)
    # dVdr = 4*E/(3*(1-poisson)**2) * math.sqrt(R_actual/2) * alpha/2/R * (-d**3+d**(alpha-1))
    # Fx = - dVdr * unit[0]
    # Fy = - dVdr * unit[1]
    #
    # if (yj > yi):
    #     print(d, Fy)
    #

    gamma = 1 #30*10**(-3)
    Eeff = E/2/(1-poisson**2)
    Fc = 3*math.pi*R*gamma
    a0 = (9*math.pi*R**2*gamma/E)**(1/3)
    Dc = a0**2/2/(6**(1/3))/R

    Reff = R/2 # 1/Reff = 1/R1 + 1/R2 but R1=R2 here
    deformation = (2*R - distance)/2
    print(deformation, a0, deformation)
    a = math.sqrt(Reff * deformation)

    Fn = 4*Fc *((a/a0)**3-(a/a0)**(3/2))

    Fx = Fn * unit[0]/1000
    Fy = Fn * unit[1]/1000
    Vij = 1
    return Fx, Fy, Vij


def plotColForceAd():
    R = 2
    xi=0
    yi=0
    xj= np.linspace(4, 2,100)
    yj=0

    overlap = 2- xj

    f = []
    d = []
    for i in range(100):
        fx, fy, v = calcCollisionAd(R, xi, yi,xj[i], yj)
        f.append(fx)
        d.append(v)

    plt.plot(overlap, f)
    # plt.ylim(-100,100)
    plt.show()

# plotColForceAd()

def calcAdhesiveForce(R, xi, yi, xj, yj):
    distance = math.sqrt((xi-xj)**2+(yi-yj)**2)
    rij = (xi-xj, yi-yj)
    unit = unitVec(rij)

    f = 0.5
    #calculate potential
    Fx = - f * unit[0]
    Fy = - f * unit[1]

    return Fx, Fy


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
    return Fx, Fy, V

# Misc geometry helper functions


# Plot the force of the wall as a function of y, calculated using the wall potential function

def plotWallForce():
    x = np.linspace(10,20,100)
    fx, fy, vwall = calcPotentialWall(40, x, slope)

    plt.plot(x, fx, label="x direction")
    plt.plot(x, fy, label = "y direction")
    plt.legend()
    plt.ylim(-100,100)
    plt.title("Wall force at x=20 accross y values")
    plt.xlabel("y")
    plt.ylabel("Force")
    plt.show()


def interpolateVort(w, dx, dy, nx, ny):
    x = np.arange(0, nx, 1)
    y = np.arange(0, ny, 1)

    vorticity = interpolate.interp2d(x, y, w.flatten(), kind='cubic')

    return vorticity

def calcHydroTorque(w, x, y, R):
    #TODO should r be cubed?
    t = 8 * math.pi * dyn_vis * R_actual**3 * w(x,y)
    # print(w(x,y), x, y)
    return t

streamfun = calcStreamFun(60)
u,v = getFluidVel(streamfun, 60,60)
# todo update R
w = vorticity(u,v,60)
wfn = interpolateVort(w,1,1,60,60)
vortx = []
for i in range(1,2000):
    # print(wfn(15,60/100/i))
    vortx.append(calcHydroTorque(wfn, 30,60*(2000-i)/2000, 1))

plt.plot(vortx)
plt.show()

# plotWallForce()
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
def stepODE(t, pos, num_parts, R, energy, forces, times, derivs, xVel, yVel, vortx):

    ddt = []
    V_col = 0
    times.append(t)

    for i in range(num_parts):
        x = pos[i*6]
        y = pos[i*6+1]
        vx = pos[i*6+2]
        vy = pos[i*6+3]
        theta = pos[i*6+4]
        w = pos[i*6+5]

        #force due to fluid flow
        #TODO: testing w/o nondim
        Fx_fluid, Fy_fluid = calcFluidForceNonDim(x, y, vx, vy, xVel, yVel)

        #force due to collisions
        Fx_col = 0
        Fy_col = 0
        for j in range(num_parts):
            if j != i:
                distance = math.sqrt((x-pos[j*6])**2 + (y-pos[j*6+1])**2)

                #if the particles overlap
                if (distance < 2*R):
                    xj = pos[j*6]
                    yj = pos[j*6+1]
                    vxj = pos[j*6+2]
                    vyj = pos[j*6+3]

                    Fx, Fy, V = calcCollision(R, x, y, vx,  vy, xj, yj, vxj, vyj)
                    Fx_col += Fx
                    Fy_col += Fy
                    V_col += V
                    # Fa_x, Fa_y = calcAdhesiveForce(R, x, y, xj, yj)

                    # Fx_col += Fa_x
                    # Fy_col += Fa_y

        #force from wall potential
        wallX = 0
        wallY = 0
        V_wall = 0
        if (x <= length*scalef/2):
            #calculate the point on the edge of the particle which is closest to the wall
            #the edge of the particle is what matters, not the center
            #this is a vector parpendicular to the wall
            if (y <= len_m/2*scalef):
                direction = unitVec((-1, 1/slope))
            else:
                direction = unitVec((-1, -1/slope))
            wallX, wallY, V_wall= calcPotentialWall(x - direction[0]*R, y - direction[1]*R, slope)
            # if (abs(wallX) > 0.001) :
            #     wallX += -0.5*direction[0]
            #     wallY += -0.5*direction[1]

        #document forces
        if i == 0:
            forces.append([[Fx_fluid, Fy_fluid], [wallX, wallY],[Fx_col, Fy_col]])

        Fx_net = Fx_fluid + wallX + Fx_col
        Fy_net = Fy_fluid + wallY + Fy_col

        Tnet = calcHydroTorque(vortx, x,y,R)

        ddt = ddt + [vx, vy, Fx_net, Fy_net, w, Tnet/inertia] #TODO should the acceleration be F/m??

    derivs.append(ddt)

    energy.append(V_col+V_wall)
    return ddt

#Parameters
#t - timestep
#pos - array of the form (x0, y0, vx0, vy0, x1, ...)
#num_parts - total # particles in simulation
#R - radius of particle
#energy - 1d array to document the energy values at each step
#vVel, yVel - the functions for calculating the velocity at a certain position
#Returns: derivatives of each value of the position array
#         [x0', y0', x0'', y0'', x1'...]
def stepODELong(t, pos, num_parts, R, energy, times, xVel, yVel):

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

                    # Fa_x, Fa_y = calcAdhesiveForce(R, x, y, xj, yj)
                    #
                    # Fx += Fa_x
                    # Fy += Fa_y

        #force from wall potential
        wallX = 0
        wallY = 0
        V_wall = 0
        if (x <= length*scalef/2):

            #calculate the point on the edge of the particle which is closest to the wall
            #the edge of the particle is what matters, not the center
            #this is a vector parpendicular to the wall
            if (y <= len_m/2*scalef):
                direction = unitVec((-1, 1/slope))
            else:
                direction = unitVec((-1, -1/slope))
            wallX, wallY, V = calcPotentialWall(x - direction[0]*R, y - direction[1]*R, slope)

        Fx_net = Fx_fluid + wallX + Fx_col
        Fy_net = Fy_fluid + wallY + Fy_col
        ddt = ddt + [vx, vy, Fx_net, Fy_net] #TODO should the acceleration be F/m??

    energy.append(V_col+V_wall)
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
    w = vorticity(u,v,60)
    vortx = interpolateVort(w, 1,1,60,60)

    solver = ode(stepODE).set_integrator('lsoda')
    solver.set_initial_value(pos0, 0).set_f_params(num_parts, r, energy, forces, times, derivs, xvel, yvel, vortx)
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

    # streamfun = calcStreamFun(80)
    streamfun = readVelocity()
    u, v = getFluidVelGraphic(streamfun, int(length*scalef), int(len_m*scalef))
    #initialize figure and create a scatterplot
    fig, ax = plt.subplots()
    plt.xlim(0,xmax)
    plt.ylim(0,ymax)
    # plt.pcolor(X, Y, u)
    # plt.colorbar()
    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid(b=True, which='major', color='#666666', linestyle='-')

    plt.plot((0, xmax/2, xmax), (ymax, scalef*(len_m+len_c)/2, ymax), c="blue")
    plt.plot((0, xmax/2, xmax), (0, scalef*(len_m-len_c)/2, 0), c="blue")

    circles = []

    def updateParticles_2(timestep):

        positions = []
        curr_num_parts = int(len(y[:][int(timestep*20)])/6)

        curr_num_parts = int(len(y[:][int(timestep*20)])/6)
        for i in range(curr_num_parts):
            posx = y[:][int(timestep*20)][0+i*6]
            posy = y[:][int(timestep*20)][1+i*6]
            theta = y[:][int(timestep*20)][4+i*6]
            positions.append((posx, posy))

            if (i >= len(circles)):
                circles.append(Circle((0,0), r, color="black", fill=False))
                ax.add_artist(circles[i])

            circles[i].center = positions[-1]

            ax.plot((posx, posy), (posx + R*math.cos(theta), posy + R*math.sin(theta)))

        #hide circles that have exited the system
        for i in range(len(circles) -curr_num_parts):
            circles[curr_num_parts+i].center = (-5,-5)

        return circles,

    #create the animation
    ani = animation.FuncAnimation(fig, updateParticles_2, frames=int(len(y)/20), interval=1)

    return ani


## Visualize the Results

# Run a simulation

# get_ipython().run_line_magic('matplotlib', 'inline')
# # #
n = int(length*scalef)
streamfun = calcStreamFun(60)
# streamfun = readVelocity()
u, v = getFluidVel(streamfun, 60, 60)
# #format: x_i, y_i, vx_i, vy_i, x_i+1...
# pos0 = []
num_parts = 6
# for j in range(num_parts):
#     if j == 1:
#         x = 23.5
#     else:
#         x = 24
#     pos0 = pos0 + [x, 10 + j*5.05, 0, 0]

# pos0 = [24, 24.71211, 0, 0, 22.785, 30.0, 0, 0, 24, 35.28789, 0, 0, 22, 27, 0, 0]
# pos0 = [33,12,0,0,32.523,15,0,0,33,18,0,0]#,32,13.5,0,0]
# pos0 =[30,13,0,0,30,17,0,0,27,12,0,0,27,18,0,0,27,15,0,0,25,14,0,0]
# pos0 = pos0 + [18, 23, 0, 0]
# pos0 = pos0 + [18, 27, 0, 0]
# pos0 = pos0 + [15, 31, 0, 0]
# pos0 = pos0 + [18, 35, 0, 0]
# pos0 = pos0 + [18, 39, 0,
pos0 = [21, 21, 0,0, 0,0]#21,39,0,0, 15,30,0,0]

r = 3
trajectory, energy, forces, t, der = runSim(1, r, 0.1, 200, pos0, u, v)

xmin = 15.0
xmax = 15.1
# for i in range(25):
#     xmid = (xmax + xmin)/2
#     pos0 = [21, 21, 0,0, 21,39,0,0, xmid,30,0,0]
#     print(pos0)
#     trajectory, energy, forces, t, der = runSim(3, r, 0.1, 200, pos0, u, v)
#
#     if (trajectory[-1][2 *4] > trajectory[-1][0]):
#         print("center particle came out in front")
#         xmax = xmid
#     elif (trajectory[-1][0] > length*scalef/2): #clog broke down exists
#         print("outisde particles came out in front")
#         xmin = xmid
#     else :
#         #clog still exists!
#         print("clog stable")
#         break

# pos0 = [21, 21, 0, 0, 21, 39, 0, 0, 15.049290466308596, 30, 0, 0, 13,24,0,0]
# print(pos0)
# trajectory, energy, forces, t, der = runSim(4, r, 0.1, 250, pos0, u, v)
ani = generateAnim(trajectory, r, n)
plt.show()
#
#
# Writer = animation.writers['ffmpeg']
# writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
# ani.save('clog.072320_4part.mp4', writer=writer)

#===Testing: adding/removing particles===


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
#particle_delay - time between eachc particle being introduced into the system
#save_prog - write progress after this time interval
#r - radius of particle
#dt - timestep
#tf - end time
#returns - y - postiions over time
#          energy - energy at each timestep
#          forces - forces at each timestep
#          times - time at each iteration
#          derives - derivatives at each timestep
def runSimAdditive(name, particle_delay, save_prog, r, dt, tf):

    energy = []
    forces = []
    times = []
    derivs = []

    streamfun = calcStreamFun(80)
    u, v = getFluidVel(streamfun, int(length*scalef), int(len_m*scalef))
    xvel, yvel = interpolateVelFn(u, v, 1, 1, int(length*scalef), int(len_m*scalef))

    current_num_parts = 0
    last_wrote = 0
    step_pos_E = 0

    clog = False
    metastable = False
    clogt = 0

    #format: x_i, y_i, vx_i, vy_i, x_i+1...
    pos0 = []
    pos0 = pos0 + randStartingPt(r, [], current_num_parts)
    current_num_parts += 1

    solver = ode(stepODELong).set_integrator('lsoda')
    solver.set_initial_value(pos0, 0).set_f_params(current_num_parts, r, energy, times, xvel, yvel)
    y, t = [pos0], []

    # for i in range(num_parts):
    while solver.successful() and (solver.t < tf):

        ti = solver.t

        #run until its time to add a new particle
        while solver.successful() and (solver.t < (ti + particle_delay)) and solver.t < tf:
            t.append(solver.t)
            out = solver.integrate(solver.t+dt)
            y = y+[out.tolist()]

            if energy[-1]>0 : step_pos_E += (t[-1] -t[-2])
            else: step_pos_E = 0

            if step_pos_E > dt :
                clog = True
                clogt = t[-1]

            if clog == True and energy[-1] == 0:
                metastable = True

        #rm any particles that have exited the system
        out = out.tolist()
        remove = []
        for j in range(current_num_parts):
            print(j*4, len(out))
            if (out[j*4] >= length*scalef -1): remove.append(j)

        for i in remove:
            out = out[:i*4] + out[i*4+4:]
            current_num_parts -= 1

        #add a new particle
        curr_t = solver.t
        add_new = random.random()*3 #add a randomiized number of particles in (0,1,2)
        pos = out
        #add 1 new particle
        if (add_new > 1):
            new_part = randStartingPt(r, pos, current_num_parts)
            pos = pos + new_part
            if (new_part != []) : current_num_parts += 1

        #add another new particle
        if (add_new > 2):
            new_part = randStartingPt(r, pos, current_num_parts)
            pos = pos + new_part
            if (new_part != []) : current_num_parts += 1

        #restart integrator w/ new sys of eqs
        solver = ode(stepODELong).set_integrator('lsoda')
        solver.set_initial_value(pos, curr_t).set_f_params(current_num_parts, r, energy, times, xvel, yvel)

        #periodically save progress
        if (curr_t > last_wrote + save_prog):
            writeData(name, y, times, energy, clog, metastable, clogt, r)

    writeData(name, y, times, energy, clog, metastable, clogt, r)

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
parser.add_argument("-p", "--particle-gap", type=float, default=10,
                    help="timestep between adding particles")
parser.add_argument("-sv", "--save-interval", type=float, default=100,
                    help="save progress after this interval")
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
def runExtendedSim(name, gap, save_prog, radius, timestep, timef):
    runSimAdditive(name, gap, save_prog, radius, timestep, timef)

args = parser.parse_args()

if (args.run_sim):
    runExtendedSim(args.name, args.particle_gap, args.save_interval, args.radius, args.step, args.time)
elif (args.animate):
    makeAnimFromFile(args.name)
else:
    print("No task assigned.\nEnding program")
