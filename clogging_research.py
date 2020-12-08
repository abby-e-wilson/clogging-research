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
len_m = 600 #all units here in micrometers
len_c = 150 #um
length = 600 #um

# len_m = 100
# len_c = 100
# length = 100

scalef = 1/10 #um
scalef_sim = 1
slope = (len_m-len_c)/length

#===Particles===from scipy.integrate import ode

mass = 10**(-6) # mass of particle in kg
R = 30 #micrometers
inertia = 2/5*mass*R**2

#===Physical Constants===
E = (10 ** 3 ) * (10**(-6))  #E in N/um**2 (newtons per micrometer squared)  (start with super soft spheres)
print(E)
poisson = 0.2
alpha = 5/2

#===Fluid Constants===
dyn_vis = 8.9 * 10 ** (-4) * 10**(-6) #dynamic viscosity (8.90 × 10−4 Pa*s for water, units of micrometers)
# dyn_vis = 10 * 10 ** (-4) * 10**(-6) #dynamic viscosity (8.90 × 10−4 Pa*s for water, units of micrometers)
# density = 997 #kg/m^3, for water
maxV = 2 #max fluid velocity
beta = 6 * math.pi * dyn_vis * R

#===Nondimentionalization Constants===
x0 = 1/beta
t0 = 1/beta

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
            u[i][j] = dPsi_dy(streamfun, 10, i, j)
            v[i][j] = -dPsi_dx(streamfun, 10, i, j)
            mag = math.sqrt(u[i][j]**2+v[i][j]**2)
            if mag > maxval : maxval = mag
    u = u/maxval * maxV #/ math.sqrt(2)
    v = v/maxval * maxV#/math.sqrt(2)
    print(maxval, maxV)
    return u, v

def vorticity(u, v,n ):
    w = np.zeros((n,n))
    for i in range(1,n-1):
        for j in range(1,n-1):
            w[i][j] = dPsi_dx(v,10,i,j) - dPsi_dy(u,10,i,j)

    # plt.pcolor(w)
    # plt.colorbar()
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
    plt.contourf(X, Y, streamfun, levels=bd)
    plt.colorbar()
    plt.contour(X, Y, streamfun, levels=bd[1:-1], colors='k')
    plt.gca().set_aspect('equal', adjustable='box')


    xmax = length* scalef
    ymax = len_m*scalef

    plt.fill((0, xmax/2, xmax,0), (ymax, scalef*(len_m+len_c)/2, ymax,ymax), c="white")
    plt.fill((0, xmax/2, xmax,0), (0, scalef*(len_m-len_c)/2, 0,0), c="white")
    plt.grid(b=True, which='minor', color='#666666', linestyle='-')
    plt.title("Streamlines of the Streamfunction")
    plt.xlabel("<---- Pipe length ---->")
    plt.ylabel("<---- Pipe Inlet ---->")
    plt.xlim(0,60)
    plt.ylim(0,60)
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

# streamfun = calcStreamFun(60)
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
    x = np.arange(0, nx*dx, dx)
    y = np.arange(0, ny*dy, dy)

    velX = interpolate.interp2d(x, y, u.flatten(), kind='cubic')
    velY = interpolate.interp2d(x, y, v.flatten(), kind='cubic')

    return velX, velY


# streamfun = calcStreamFun(60)
# u, v = getFluidVel(streamfun, 60,60)
# uvel, vvel = interpolateVelFn(u, v, 10,10, 60,60)
# x = np.linspace(0,600,1000)
# plt.plot(x, uvel(100,x))
# plt.show()

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

            mag[j][i] = math.sqrt(u[i][j]**2+v[i][j]**2)
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
            else:
                X[i][j]=-10

    plt.pcolor(mag)
    plt.colorbar()
    # plt.colorbar()
    plt.quiver(X, Y, u, v, headaxislength=6,color='black')
    plt.xlim(0,60)
    plt.title("Velocity Field for Pipe with Constriction")
    plt.xlabel("<---- Pipe length ---->")
    plt.ylabel("<---- Pipe Inlet ---->")
    plt.gca().set_aspect('equal', adjustable='box')

    xmax = length* scalef
    ymax = len_m*scalef
    plt.fill((0, xmax/2, xmax,0), (ymax, scalef*(len_m+len_c)/2+1, ymax, ymax), c="white")
    plt.fill((0, xmax/2, xmax,0), (0, scalef*(len_m-len_c)/2, 0,0), c="white")
    plt.plot((0, xmax/2, xmax), (ymax, scalef*(len_m+len_c)/2+1, ymax), c="black")
    plt.plot((0, xmax/2, xmax), (0, scalef*(len_m-len_c)/2, 0), c="black")
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
total_disp = []
relvelocity = []
svector = []
unittan = []
friction_options = []
overlapN = []
coltorques = [[0,0]]
def calcCollision(R, xi, yi, xj, yj, vxi, vyi, vxj, vyj, dispi, dispj, dt, wi, wj):

    global coltorques
    # print("collision")
    #Basic vectors
    distance = math.sqrt((xi-xj)**2+(yi-yj)**2)
    #TODO negated nij
    nij = -np.array(unitVec([xj-xi, yj-yi]))
    tij = np.array(unitVec([nij[1], -nij[0]]))

    #effective values across both particles
    r_eff = R/2
    m_eff = mass/2
    E_eff = E/2/(1-poisson**2)
    G_eff = E_eff/4/(2+poisson)/(1-poisson)

    #Relative velocity calculation
    wi_vec = [0,0,wi]
    wj_vec = [0,0,wj]
    ri = [R*nij[0], R*nij[1], 0]
    rj = [-R*nij[0], -R*nij[1], 0]
    angveli = np.cross(wi_vec, ri)
    angvelj = np.cross(wj_vec, rj)
    veli = np.array([vxi + angveli[0], vyi + angveli[1]])
    velj = np.array([vxj + angvelj[0], vyj + angvelj[1]])

    relvel = -1*(velj - veli)
    if (np.linalg.norm(relvel) != 0):
        unit_angvel = relvel/np.linalg.norm(relvel)
    else:
        unit_angvel = [0,0]
    S = np.array(unitVec(np.dot(relvel, tij) * tij)) #Unit vec in tang plane in dir of relative velocity

    #normal overlap
    overlap_n = (1-distance/2/R)*2*R #- distance
    # overlap_n = (2*R - distance)

    #Normal coefficients
    er = 1 # coefficent of resitution
    damping_coeff = -2 * math.sqrt(5/6) * np.log(er)/math.sqrt(np.log(er)**2 + math.pi**2)
    kn = 4/3*E_eff*math.sqrt(r_eff*overlap_n)
    yn = damping_coeff * math.sqrt(2/3*kn*m_eff)

    #Hertz-Mindlin contact Model
    Fn = (kn* overlap_n - yn*np.dot(relvel, nij))
    # print(np.dot(relvel, nij), relvel, nij)
    Fx = Fn * nij[0] #negative because the force is opposite the normal vector,
    Fy = Fn * nij[1] #it points into the particle

    # tangential overlap
    # dispi_curr = np.dot([vxi, vyi], tij) * dt + dispi
    # if (abs(np.dot(relvel, S))<10e-5):
    #     dispi_curr = 0
    # print("collision - ",dispi_curr)
    # overlap_t = abs(dispi_curr+dispj)
    #NEW DISP I: disp = origi point of contact
    if(np.linalg.norm(dispi) == 0):
        print("set pos")
        dispi = [xi,yi]
        dispj = [xj,yj]
    dispi_curr = np.array([dispi[0]-xi, dispi[1]-yi])
    dispj_curr = np.array([dispj[0]-xj, dispj[1]-yj])
    disptot = dispi_curr-dispj_curr #[-xi+dispi[0],-yi+dispi[1]]
    tangential_direction = np.dot(disptot, tij)*tij
    if(np.linalg.norm(tangential_direction) != 0):
        tdnorm = np.array(tangential_direction)/np.linalg.norm(tangential_direction)
    else:
        tdnorm = np.array([0,0])
    overlap_t = np.linalg.norm(tangential_direction)


    #tangential coefficients
    kt = 8 * G_eff * math.sqrt(r_eff*overlap_n)
    yt = damping_coeff *  math.sqrt(kt*m_eff)
    coeff_friction = 0.2

    #tangential force
    magFt = -min(coeff_friction*abs(Fn),abs(kt*overlap_t- yt*np.dot(relvel, tdnorm)))
    Ft = magFt*S
    Ft = -magFt*tdnorm
    # Ft = 0 * S
    # print(Ft)

    #TODO SIGN ERROR HERE
    torque = np.cross([-R*nij[0], -R*nij[1], 0], [Ft[0], Ft[1], 0])[2]

    if (yi<290):
    #     friction_options.append([coeff_friction*abs(Fn), kt*overlap_t, magFt])
    #     relvelocity.append(relvel)
    #     unittan.append(tij)
    #     relative_angle = np.arccos(np.dot(unit_angvel[0:2], tij))
    #     svector.append(relative_angle)
        overlapN.append(overlap_n)

    # TODO update
    Vij = 0

    coeff_rolling = 0.02
    rolling_torque = 0
    # if (wi != 0):
    #     dir_relvel = np.cross([R*nij[0], R*nij[1], 0], [relvel[0], relvel[1], 0])[2]
    #     rolling_torque = -dir_relvel/abs(dir_relvel) * coeff_rolling * abs(Fn) * R#/MODIFIED - ok?

    if (yi>310):
        # print("col")
        # if(walltorques == None):
        #     walltorques = []
        total_disp.append(overlap_t)
        coltorques = np.concatenate((coltorques, np.array([[torque, rolling_torque]])))

    torque += rolling_torque

    # Ft = [0,0]
    # torque = 0
    # dispi = 0
    return Fx, Fy,Ft[0],Ft[1],0,torque, dispi#contact_updated#Ft[0],Ft[1], Vij, torque, contact_updated


#Calculate the forces from a collision between 2 particles
#R - radius of the particle (in proportion to the system)
#xi, yi - position of particle 1
#vxi, vyi - velocity of particle 1
#xj, yj - position of particle 2
#vxj, vyj - velocity of particle 2
#returns - the force on particle i in the x and y directions and the potential
def CollisionAd(R, xi, yi, xj, yj):

    distance = math.sqrt((xi-xj)**2+(yi-yj)**2)
    rij = (xi-xj, yi-yj)
    unit = unitVec(rij)
    print("collision")

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
    # print(deformation, a0, deformation)
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
        fx, fy, v = CollisionAd(R, xi, yi,xj[i], yj)
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

    Fx_fluid = beta * (uvel - vx)
    Fy_fluid = beta * (vvel- vy)
    # Fx_fluid = beta * (uvel * t0**2 /x0 - vx * t0)
    # Fy_fluid = beta * (vvel * t0**2 /x0 - vy * t0)
    # Fx_fluid = beta * (uvel * t0**2 /x0 /mass - vx * t0 / mass)
    # Fy_fluid = beta * (vvel * t0**2 /x0 /mass - vy * t0 / mass)

    return Fx_fluid[0], Fy_fluid[0]
    # print(Fx_fluid, Fy_fluid)
    # return Fx_fluid, Fy_fluid

#Calculate the potential as the particle approaches a wall
#x, y - position of particle
#slope - slope of the wall
#tau, eta - nondimentionalization constants
#returns - forces in the x and y directions
walltorques = np.array([[0,0]])
walldisp = [[0,0],[0,0],[0,0]]
def calcPotentialWall(x, y, slope, Fxex, Fyex, R, tnet, direction, i, w, vx, vy, disp, dt):
    global walltorques

    # a = 50
    # V = (math.e ** (-a*(y-x*slope)) + math.e **(a*(y-(len_m*scalef_sim-x*slope))))
    # Fx = -a*slope*V*10e-6
    # Fy=  a*(math.e ** (-a*(y-x*slope)) - math.e **(a*(y-(len_m*scalef_sim-x*slope))))*10e-6
    # Fn = math.sqrt(Fx**2 + Fy**2)

    m1 = slope
    m2 = -slope
    b1 = 0
    b2 = len_m
    d1 = abs((b1+m1*x-y)/math.sqrt(1+m1**2))
    d2 = abs((b2+m2*x-y)/math.sqrt(1+m2**2))
    distance = min(d1, d2)

    #Basic vectors
    nij = np.array(direction)
    tij = np.array(unitVec([nij[1], -nij[0]]))

    #effective values across both particles
    r_eff = R
    m_eff = mass
    E_eff = E/2/(1-poisson**2)
    G_eff = E_eff/4/(2+poisson)/(1-poisson)

    #Relative velocity calculation
    wi_vec = [0,0,w]
    ri = [R*nij[0], R*nij[1], 0]
    angveli = np.cross(wi_vec, ri)
    # print(w, angveli)
    veli = np.array([vx + angveli[0], vy + angveli[1]])

    relvel = veli
    if (np.linalg.norm(relvel) != 0):
        unit_angvel = relvel/np.linalg.norm(relvel)
    else:
        unit_angvel = [0,0]
    S = np.array(unitVec(np.dot(relvel, tij) * tij)) #Unit vec in tang plane in dir of relative velocity
    # print(S)
    #normal overlap
    overlap_n = (R-distance)
    # overlap_n = (0.5 - distance)
    # overlap_n = (2*R - distance)

    torque = 0
    dispi_curr = 0
    Ft = [0,0]
    Fx = 0
    Fy = 0
    # print(overlap_n, distance)
    if (overlap_n >= 0):
    #     #Normal coefficients
        er = 1 # coefficent of resitution
        damping_coeff = -2 * math.sqrt(5/6) * np.log(er)/math.sqrt(np.log(er)**2 + math.pi**2)
        kn = 4/3*E_eff*math.sqrt(r_eff*overlap_n)
        yn = damping_coeff * math.sqrt(2/3*kn*m_eff)

        #Hertz-Mindlin contact Model
        Fn = (kn* overlap_n - yn*np.dot(relvel, nij)) *2
        Fx += Fn * nij[0] #negative because the force is opposite the normal vector,
        Fy += Fn * nij[1] #it points into the particle
    #
        # tangential overlap
        # if (distance < 1):
        # print(veli, tij)
        # dispi_curr = np.dot([vx, vy], tij) * dt + disp
        # if (abs(np.dot(relvel, S))<10e-5):
        #     dispi_curr = 0
        # # print(dispi_curr)
        # overlap_t = abs(dispi_curr)
        #
        if(np.linalg.norm(disp) == 0):
            disp= [x,y]
        disptot = [-x+disp[0],-y+disp[1]]
        tangential_direction = np.dot(disptot, tij)*tij
        if(np.linalg.norm(tangential_direction) != 0):
            tdnorm = tangential_direction/np.linalg.norm(tangential_direction)
        else:
            tdnorm = np.array([0,0])
        overlap_t = np.linalg.norm(tangential_direction)
        # overlap_t = dispi_curr
        # overlap_t = abs(disp)
        # else:
            # overlap_t = 0

        #tangential coefficients
        kt = 8 * G_eff * math.sqrt(r_eff*overlap_n)
        yt = damping_coeff *  math.sqrt(kt*m_eff)
        coeff_friction = 0.2

        #tangential force
        #TESTING
        magFt = -min(coeff_friction*abs(Fn),abs( kt*overlap_t- yt*np.dot(relvel, tdnorm)))
        # magFt = -min(coeff_friction*abs(Fn), abs(kt*np.linalg.norm(relvel)- yt*np.dot(relvel, S)))
        Ft = magFt*S
        Ft = -magFt*tdnorm
        # print(magFt)

        #TODO SIGN ERROR HERE
        torque = np.cross([-R*nij[0], -R*nij[1], 0], [Ft[0], Ft[1], 0])[2]

        if (y<290):
            # print(S, relvel[0])
            friction_options.append([coeff_friction*abs(Fn), kt*overlap_t, overlap_t])
            relvelocity.append(relvel[0])
            unittan.append(tij)
            # relative_angle = np.arccos(np.dot(unit_angvel[0:2], tij))
            svector.append([kt*overlap_t, yt*np.dot(relvel,S)])
            # overlapN.append(overlap_n)
            # total_disp.append(overlap_t)

        coeff_rolling = 0.02
        rolling_torque = 0
        # if (w != 0):
        #     dir_relvel = np.cross([R*nij[0], R*nij[1], 0], [relvel[0], relvel[1], 0])[2]
        #     rolling_torque = -dir_relvel/abs(dir_relvel) * coeff_rolling * abs(Fn) * R#/MODIFIED - ok?
            # print(w, rolling_torque, torque)
        if (y<290):
            # print("wall")
            # if(walltorques == None):
            #     walltorques = []
            walltorques = np.concatenate((walltorques, np.array([[torque, rolling_torque]])))
        torque += rolling_torque

    else:
        disp = [0,0]

    V = 0
    # Ft = [0,0]
    # torque = 0
    # disp = [0,0]
    # print(Ft)
    return Fx, Fy, Ft[0], Ft[1], V, torque, disp#torque
    # return Fx, Fy, V, 0, dispi_curr

# Misc geometry helper functions

#dimer
def attractiveForce(xi, yi, xj, yj, R):

    distance = math.sqrt((xi-xj)**2+(yi-yj)**2)
    rij = (xi-xj, yi-yj)
    unit = unitVec(rij)

    f = 2*(2*R - distance)

    return f*unit[0], f*unit[1]

def attractivePotential(xi, yi, xj, yj, R):

    distance = math.sqrt((xi-xj)**2+(yi-yj)**2)
    rij = (xi-xj, yi-yj)
    unit = unitVec(rij)

    V = (2*R-distance)**2

    return V

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


def wallFrictionTorque(x, y, Fx, fy, R):

    if (y <= len_m/2*scalef_sim):
        direction = unitVec((1, slope))
    else:
        direction = unitVec((1, -slope))

    direction = np.array(direction)
    force = np.array([Fx, Fy])

    proj_of_force_on_wall_dir = (np.dot(force, direction))*direction

    print("Smaller?", math.sqrt(Fx**2+Fy**2) <= math.sqrt(proj_of_force_on_wall_dir[0]**2+proj_of_force_on_wall_dir[1]**2))

    return proj_of_force_on_wall_dir*(-R)

torques = []
def collisionTorque(R, x, y, xj, yj, Fx, Fy, w, tnet, i):

    rij = (x-xj, y-yj, 0)
    unit = np.array(unitVec(rij))
    torque_dir = np.array([unit[1], -unit[0], 0])

    force = [Fx, Fy, 0]

    torque = 0
    proj_of_force_on_wall_dir = [0,0]
    # if (abs(w) ==0):#< 10e-4):
    #static
    print("static",abs(w))
    # print(force, torque_dir)
    rad = -R*np.array(unit)
    proj_of_force_on_wall_dir = (np.dot(force, torque_dir))*torque_dir
    torque += -tnet#np.cross(proj_of_force_on_wall_dir, rad)[2] - tnet
    # else:
    if (abs(w)>0.0001):
        print("kinetic")
        # normal_force = np.dot(force[0:2], unit)*unit
        torque += -w/abs(w) *math.sqrt(Fx**2+Fy**2)*R #math.sqrt(normal_force[0]**2+normal_force[1]**2)

    # print(torque)
    if y <300 and i==0:
        torques.append(torque)
    #torque, counterforce x and y
    return torque, -proj_of_force_on_wall_dir[0], -proj_of_force_on_wall_dir[1]



def interpolateVort(w, dx, dy, nx, ny):
    x = np.arange(0, nx*dx, dx)
    y = np.arange(0, ny*dy, dy)

    vorticity = interpolate.interp2d(x, y, w.flatten(), kind='cubic')

    return vorticity

def calcHydroTorque(w, x, y, a, R):
    #TODO should r be cubed?
    t = 8 * math.pi * dyn_vis * R**3 * (w(x,y)-a)
    # print(w(x,y), x, y)
    return t

# streamfun = calcStreamFun(60)
# u,v = getFluidVel(streamfun, 60,60)
# # todo update R
# w = vorticity(u,v,60)
# wfn = interpolateVort(w,1,1,60,60)
# vortx = []
# for i in range(1,2000):
#     # print(wfn(15,60/100/i))
#     vortx.append(calcHydroTorque(wfn, 30,60*(2000-i)/2000, 1))
#
# plt.plot(vortx)
# plt.show()

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
omega = []
tdisp = [[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0]]
# tdisp = [[0,0,0],[0,0,0],[0,0,0]]
tlast = 0
collision_friction = [[0,0],[0,0],[0,0]]
def stepODE(t, pos, num_parts, R, energy, forces, times, derivs, xVel, yVel, vortx):

    global tlast
    global tdisp
    # global walldisp
    # print(tlast)
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
        if (i==0):
            omega.append(w)

        #force due to fluid flow
        #TODO: testing w/o nondim
        # Fx_fluid, Fy_fluid = calcFluidForce(x, y, vx, vy, xVel, yVel)
        Fx_fluid, Fy_fluid = calcFluidForceNonDim(x, y, vx, vy, xVel, yVel)
        # Fx_fluid, Fy_fluid = calcFluidForce(x, y, vx, vy, xVel, yVel)

        #force due to collisions
        Fx_col = 0
        Fy_col = 0
        fricx_tot = 0
        fricy_tot = 0
        Tnet = 0
        for j in range(num_parts):

            if j != i:
                distance = math.sqrt((x-pos[j*6])**2 + (y-pos[j*6+1])**2)

                #if the particles overlap
                if (distance < 2*R):
                    xj = pos[j*6]
                    yj = pos[j*6+1]
                    vxj = pos[j*6+2]
                    vyj = pos[j*6+3]
                    wj = pos[j*6+5]

                    # if (tdisp[i][j*2] == 0):
                    #     r = [xj - x, yj-y]
                    #     tdisp[i][j*2] = x + r[0]/2
                    #     tdisp[i][j*2] = y + r[1]/2

                    Fx, Fy, fricx, fricy, V, col_torque, newdisp = calcCollision(R, x, y, xj, yj, vx, vy, vxj, vyj, tdisp[i][j], tdisp[j][i], t-tlast, w, wj)
                    Fx_col += Fx
                    Fy_col += Fy
                    fricx_tot += fricx
                    fricy_tot += fricy
                    V_col += V
                    Tnet += col_torque
                    tdisp[i][j] = newdisp
                    # tdisp[i][j*2+1] = newdisp[1]
                    # print(Fx, Fx/mass)
                    # Fa_x, Fa_y = calcAdhesiveForce(R, x, y, xj, yj)

                    # Fx_col += Fa_x
                    # Fy_col += Fa_y

                    # if j==0 and i==3 or j==3 and i==0:
                    #     Fx_atr, Fy_atr = attractiveForce(x, y, xj, yj, R)
                    #     Fx_col += Fx_atr
                    #     Fy_col += Fy_atr

                #else not in contact
                else:
                    # print("no contact")
                    #reset displacement
                    # print("reset")
                    tdisp[i][j] = [0,0]
                    # tdisp[i][j*2+1] = 0

                # if (i==0 and j ==2):
                #     print("displacecment", tdisp[i][j], tdisp[j][i])
                #     total_disp.append(abs(tdisp[i][j]+tdisp[j][i]))


        Tnet += calcHydroTorque(vortx, x,y,w, R) #+ Twall
        # print(Tnet/inertia, Twall/inertia)

        #force from wall potential
        wallX = 0
        wallY = 0
        wfx = 0
        wfy = 0
        V_wall = 0
        Twall = 0
        if (x <= length*scalef_sim/2):
        # if (x <= length/2):

            #calculate the point on the edge of the particle which is closest to the wall
            #the edge of the particle is what matters, not the center
            #this is a vector parpendicular to the wall
            # if (y <= len_m/2):
            if (y <= len_m/2*scalef_sim):
                direction = unitVec((-1, 1/slope))
            else:
                direction = unitVec((-1, -1/slope))
            # wallX, wallY, wfx, wfy, V_wall, Twall, newwalldisp = calcPotentialWall(x - direction[0]*R, y - direction[1]*R, slope, Fx_col+Fx_fluid, Fy_col + Fy_fluid, R, Tnet, direction, i, w, vx, vy, walldisp[i], t-tlast)
            #actual x, y - not nearest point
            wallX, wallY, wfx, wfy, V_wall, Twall, newwalldisp = calcPotentialWall(x, y, slope, Fx_col+Fx_fluid, Fy_col + Fy_fluid, R, Tnet, direction, i, w, vx, vy, walldisp[i], t-tlast)

            walldisp[i] = newwalldisp
            Tnet += Twall

        collision_friction[i] = [fricx_tot, fricy_tot]
        #document forces
        if i == 0:
            forces.append([[Fx_fluid, Fy_fluid], [wallX, wallY],[Fx_col, Fy_col],[fricx_tot, fricy_tot], [wfx, wfy]])
            # forces[-1] = [[Fx_fluid, Fy_fluid], [wallX, wallY],[Fx_col, Fy_col],[fricx_tot, fricy_tot], [wfx, wfy]]

        Fx_net = Fx_fluid + wallX + Fx_col + fricx_tot + wfx
        Fy_net = Fy_fluid + wallY + Fy_col + fricy_tot + wfy

        T_col = 0
        # for j in range(num_parts):
        #     if j != i:
        #         distance = math.sqrt((x-pos[j*6])**2 + (y-pos[j*6+1])**2)
        #
        #         #if the particles overlap
        #         if (distance < 2*R):
        #             xj = pos[j*6]
        #             yj = pos[j*6+1]
        #             vxj = pos[j*6+2]
        #             vyj = pos[j*6+3]
        #
        #             #TODO this should be force between these two surfaces, not colition total for 3+ systems
        #             T_col_curr, Fricx, Fricy = collisionTorque(R, x, y, xj, yj, Fx_col, Fy_col, w, Tnet, i)
        #             T_col += T_col_curr
        #             Fx_net += Fricx
        #             Fy_net += Fricy
                    # print(Tnet, Twall, T_col)

        # err = 10e-6
        # if (abs(vx) <err and abs(vy)<err and abs(T_col)>0 and (T_col+Tnet)==0):
        #     print("stuck")
        #     Fx_net = 0
        #     Fy_net = 0
            # T_col =0
            # Twall = 0
            # Tnet=0

        # if (Fx_col != 0):
        #     Tnet = 0

        # if (T_col + Tnet==0):
            # ddt = ddt + [vx, vy, Fx_net/mass, Fy_net/mass, 0,0 ]#+ Twall +T_col] #TODO should the acceleration be F/m??
        # else:
            # print(( T_col+Tnet)/inertia, w)

        # if (w==0):
        #     print("w=0: torque is ", (T_col+Tnet+Twall))
        if (i==0):
            torques.append(Tnet)

        ddt = ddt + [vx, vy, Fx_net/mass, Fy_net/mass, w, Tnet/inertia]# (T_col+Tnet+Twall)/inertia ]#+ Twall +T_col] #TODO should the acceleration be F/m??

    tlast = t

    derivs.append(ddt)

    energy.append(V_col+V_wall)
    return ddt


def stickPotential(x, y, xp, yp):
    dist = math.sqrt((x-xp)**2+(y-yp)**2)
    dir = np.array(unitVec([xp-x, yp-y]))
    F = dist * dir
    return F


# omega = np.array((0))
# tdisp = [[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0]]
# tdisp = [[0,0,0],[0,0,0],[0,0,0]]
# tdisp = [[[0,0],[0,0]],[[0,0],[0,0]]]
tlast = 0
collision_friction = [[0,0],[0,0],[0,0]]
# walldisp = [0]
def stepODEFricTest(t, pos, num_parts, R, energy, forces, times, derivs):

    global tlast
    global tdisp
    global omega
    # global walldisp
    # print(tlast)
    ddt = []
    V_col = 0
    times.append(t)

    #stuck particle i=0

    i = 1
    f_stuck = stickPotential(pos[i*6], pos[i*6+1], 50,75)

    ddt = [pos[i*6+2],pos[i*6+3],f_stuck[0]/mass, f_stuck[1]/mass, 0,0]

    i = 0
    j = 1

    x = pos[i*6]
    y = pos[i*6+1]
    vx = pos[i*6+2]
    vy = pos[i*6+3]
    theta = pos[i*6+4]
    w = pos[i*6+5]
    omega = np.append(omega, [w])

    #force due to fluid flow
    # Fx_fluid, Fy_fluid = calcFluidForceNonDim(x, y, vx, vy, xVel, yVel)

    #force due to collisions
    Fx_col = 0
    Fy_col = 0
    fricx_tot = 0
    fricy_tot = 0
    Tnet = 0


    distance = math.sqrt((x-pos[j*6])**2 + (y-pos[j*6+1])**2)

    #if the particles overlap
    if (distance < 2*R):
        xj = pos[j*6]
        yj = pos[j*6+1]
        vxj = pos[j*6+2]
        vyj = pos[j*6+3]
        wj = pos[j*6+5]

        Fx, Fy, fricx, fricy, V, col_torque, newdisp = calcCollision(R, x, y, xj, yj, vx, vy, vxj, vyj, tdisp[i][j], tdisp[j][i], t-tlast, w, wj)
        Fx_col += Fx
        Fy_col += Fy
        fricx_tot += fricx
        fricy_tot += fricy
        V_col += V
        Tnet += col_torque
        tdisp[i][j] = newdisp

    #else not in contact
    else:
        tdisp[i][j] = [0,0]


    streamfun = calcStreamFun(10)
    u, v = getFluidVel(streamfun, 10,10)
    xvel, yvel = interpolateVelFn(u, v, 10, 10, 10, 10)

    # print(vx, vy, x, y)
    Fx_fluid, Fy_fluid = calcFluidForceNonDim(x, y, vx, vy, xvel, yvel)
    # Fx_fluid = 10e-5
    # Fy_fluid = 0
    # Fx_fluid = 0
    # Fy_fluid = 0
    # Tnet += calcHydroTorque(vortx, x,y,w, R) #+ Twall
    # print(Tnet/inertia, Twall/inertia)

    #force from wall potential
    wallX = 0
    wallY = 0
    wfx = 0
    wfy = 0
    V_wall = 0
    Twall = 0

    #calculate the point on the edge of the particle which is closest to the wall
    #the edge of the particle is what matters, not the center
    #this is a vector parpendicular to the wall
    # if (y <= len_m/2):
    if (y <= 50):
        # direction = unitVec((-1, 1/slope))
        direction = [0,1]
    else:
        # direction = unitVec((-1, -1/slope))
        direction = [0,-1]
    # print(walldisp[i])
    wallX, wallY, wfx, wfy, V_wall, Twall, newwalldisp = calcPotentialWall(x, y, slope, Fx_col, Fy_col, R, Tnet, direction, i, w, vx, vy, walldisp[i], t-tlast)

    walldisp[i] = newwalldisp
    # print(walldisp[i])
    Tnet += Twall

    # collision_friction[i] = [fricx_tot, fricy_tot]
    #document forces
    forces.append([[Fx_fluid, Fy_fluid], [wallX, wallY],[Fx_col, Fy_col],[fricx_tot, fricy_tot], [wfx, wfy]])

    # print("wall fric", wfx)
    Fx_net = Fx_fluid + wallX + Fx_col + fricx_tot + wfx
    Fy_net = Fy_fluid + wallY + Fy_col + fricy_tot + wfy

    ddt = [vx, vy, Fx_net/mass, Fy_net/mass, w, Tnet/inertia] + ddt

    tlast = t
    print("time", tlast)

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
# def stepODELong(t, pos, num_parts, R, energy, times, xVel, yVel):
#
#     ddt = []
#     V_col = 0
#     times.append(t)
#
#     for i in range(num_parts):
#         x = pos[i*4]
#         y = pos[i*4+1]
#         vx = pos[i*4+2]
#         vy = pos[i*4+3]
#
#         #force due to fluid flow
#         #TODO: testing w/o nondim
#         Fx_fluid, Fy_fluid = calcFluidForceNonDim(x, y, vx, vy, xVel, yVel)
#
#         #force due to collisions
#         Fx_col = 0
#         Fy_col = 0
#         for j in range(num_parts):
#             if j != i:
#                 distance = math.sqrt((x-pos[j*4])**2 + (y-pos[j*4+1])**2)
#
#                 #if the particles overlap
#                 if (distance < 2*R):
#                     xj = pos[j*4]
#                     yj = pos[j*4+1]
#                     vxj = pos[j*4+2]
#                     vyj = pos[j*4+3]
#
#                     Fx, Fy, V = calcCollision(R, x, y, vx, vy, xj, yj, vxj, vyj)
#                     Fx_col += Fx
#                     Fy_col += Fy
#                     V_col += V
#
#                     # Fa_x, Fa_y = calcAdhesiveForce(R, x, y, xj, yj)
#                     #
#                     # Fx += Fa_x
#                     # Fy += Fa_y
#
#         #force from wall potential
#         wallX = 0
#         wallY = 0
#         V_wall = 0
#         if (x <= length*scalef/2):
#
#             #calculate the point on the edge of the particle which is closest to the wall
#             #the edge of the particle is what matters, not the center
#             #this is a vector parpendicular to the wall
#             if (y <= len_m/2*scalef):
#                 direction = unitVec((-1, 1/slope))
#             else:
#                 direction = unitVec((-1, -1/slope))
#             wallX, wallY, V = calcPotentialWall(x - direction[0]*R, y - direction[1]*R, slope)
#
#         Fx_net = Fx_fluid + wallX + Fx_col
#         Fy_net = Fy_fluid + wallY + Fy_col
#         ddt = ddt + [vx, vy, Fx_net, Fy_net] #TODO should the acceleration be F/m??
#
#     energy.append(V_col+V_wall)
#     return ddt

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

velocities = []
graphic_fric = []
def runSim(num_parts, r, dt, tf, pos0, u, v):

    print("starting sim....")
    energy = []
    forces = []
    times = []
    derivs = []
    xvel, yvel = interpolateVelFn(u, v, 10, 10, 60, 60)
    w = vorticity(u,v,60)
    vortx = interpolateVort(w, 10,10,60,60)

    solver = ode(stepODE).set_integrator('lsoda')
    solver.set_initial_value(pos0, 0).set_f_params(num_parts, r, energy, forces, times, derivs, xvel, yvel, vortx)
    # #
    # solver = ode(stepODEFricTest).set_integrator('lsoda')
    # solver.set_initial_value(pos0, 0).set_f_params(num_parts, r, energy, forces, times, derivs)

    y, t = [pos0], []
    while solver.successful() and solver.t < tf:
        t.append(solver.t)
        print(solver.t)
        # forces.append([])
        out = solver.integrate(solver.t+dt)
        y = np.concatenate((y, [out]), axis=0)
        vels = []
        frics = []
        for i in range(num_parts):
            vels.append([math.sqrt(out[i*6+2]**2+ out[i*6+3]**2)])
            # print(collision_friction)
            # print(collision_friction[i])
            frics.append(collision_friction[i])
        velocities.append(vels)
        graphic_fric.append(frics)
        # print(y[-1])

#     print(solver.get_return_code())

    print("finished sim...")
    # return y, energy, forces, times, derivs
    return y, energy, forces, t, derivs





# Animate the trajectories of the particles

# get_ipython().run_line_magic('matplotlib', 'inline')

def generateAnim(y, r, friction):
    xmax = length*scalef_sim
    ymax = len_m*scalef_sim
    X = np.linspace(0, xmax, int(length*scalef_sim))
    Y = np.linspace(0, ymax, int(len_m*scalef_sim))

    # streamfun = calcStreamFun(80)
    # streamfun = readVelocity()
    # u, v = getFluidVelGraphic(streamfun, int(length*scalef), int(len_m*scalef))
    #initialize figure and create a scatterplot
    fig, ax = plt.subplots()
    plt.xlim(0,xmax)
    plt.ylim(0,ymax)
    # plt.pcolor(X, Y, u)
    # plt.colorbar()
    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid(b=True, which='major', color='#666666', linestyle='-')

    plt.plot((0, xmax/2, xmax), (ymax, (len_m*scalef_sim+len_c*scalef_sim)/2, ymax), c="blue")
    plt.plot((0, xmax/2, xmax), (0, (len_m*scalef_sim-len_c*scalef_sim)/2, 0), c="blue")
    # plt.plot((0, xmax/2, xmax), (ymax, scalef*(len_m+len_c)/2, ymax), c="blue")
    # plt.plot((0, xmax/2, xmax), (0, scalef*(len_m-len_c)/2, 0), c="blue")

    scatter = ax.scatter([], [], c='red')
    circles = []
    # arrow = plt.arrow(0,0,0,0)
    # patch = ax.add_patch(arrow)

    def updateParticles_2(timestep):

        # ax.clear()

        # global patch
        positions = []
        dots = []
        curr_num_parts = int(len(y[:][int(timestep*5)]))

        curr_num_parts = int(len(y[:][int(timestep*5)])/6)

        # for patch in patches:
        # patch.remove()
        # while (len(ax.patches) >0):
        #     ax.patches.pop().remove()

        for i in range(curr_num_parts):
            posx = y[:][int(timestep*5)][0+i*6]
            posy = y[:][int(timestep*5)][1+i*6]
            theta = y[:][int(timestep*5)][4+i*6]
            positions.append((posx, posy))

            if (i >= len(circles)):
                circles.append(Circle((0,0), r, color="black", fill=False))
                ax.add_artist(circles[i])

            circles[i].center = positions[-1]

            dots.append([posx + r*math.cos(theta), posy + r*math.sin(theta)])

            # if (i==0):
            #     newpatch = plt.arrow(posx, posy,friction[int(timestep*5)][i][0]*10e6,friction[int(timestep*5)][i][1]*10e6)
            #     patch = ax.add_patch(newpatch)

        #hide circles that have exited the system
        for i in range(len(circles) -curr_num_parts):
            circles[curr_num_parts+i].center = (-5,-5)

        scatter.set_offsets(dots)
        return circles, scatter#, patch

    #create the animation
    ani = animation.FuncAnimation(fig, updateParticles_2, frames=int(len(y)/5), interval=1)

    return ani


## Visualize the Results

# Run a simulation

# get_ipython().run_line_magic('matplotlib', 'inline')
# # #
# n = int(length*scalef)
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
pos0 = [19, 23, 0,0, 19,37,0,0]#, 15,30,0,0]

r = 30
# trajectory, energy, forces, t, der = runSim(3, r, 0.1, 200, pos0, u, v)

# xmin = 15.0
# xmax = 15.1
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

pos0 = [210, 210, 0, 0, 0,0, 210.2, 390, 0, 0,0,0, 149.5, 300, 0, 0,0,0]#, 13,24,0,0]
# pos0 = [210, 210, 0, 0, 0,0, 210, 390, 0, 0,0,0, 150, 300, 0, 0,0,0]#, 13,24,0,0]
# pos0 = [210, 390, 0, 0,0,0,149.5, 300, 0, 0,0,0]
# pos0 = [50,25,0,0,0,0,50,75,0,0,0,0]
# pos0 = [210, 210, 0, 0, 0,0, 210, 390, 0, 0,0,0, 155, 300, 0, 0,0,0]#, 13,24,0,0]

#with friciton
# pos0 = [210, 210, 0, 0, 0,0, 210, 390, 0, 0,0,0, 150, 300, 0, 0,0,0]#, 13,24,0,0]

# pos0 = [210, 210, 0, 0, 210, 390, 0, 0, 151, 300, 0, 0]#, 13,24,0,0]
# print(pos0)
# pos0 = [21, 21, 0, 0, 21, 39, 0, 0, 15.049290466308596, 30, 0, 0, 13,24,0,0]
# pos0 = [20.5, 21, 0, 0, 21.5, 39, 0, 0, 15.1, 31, 0, 0, 14.5,24.5,0,0]
# pos0 = [21, 21, 0, 0, 0, 0, 21, 39, 0, 0, 0, 0, 15.049290466308596, 30, 0, 0,0,0]#, 13,24,0,0]
# print(pos0)
# pos0 = [210, 240, 0,0,0,0,210,360,0,0,0,0]
trajectory, energy, forces, t, der = runSim(3, r, 0.1, 85, pos0, u, v)
# r = 25.0001
# trajectory, energy, forces, t, der = runSim(2, r, 0.01, 0.8, pos0, u, v)
ani = generateAnim(trajectory, r, np.array(graphic_fric))
plt.show()

# Writer = animation.writers['ffmpeg']
# writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
# ani.save('clog.120620_fric_stable.mp4', writer=writer)

# x1 = [trajectory[:][i][i] for i in range(len(t))]
# x2 = [trajectory[:][i][i] for i in range(len(trajectory[:]))]
# x3 = [trajectory[:][i][i] for i in range(len(trajectory[:]))]
# plt.plot(t, x1)
# plt.plot(x2)
# plt.plot(x3)
# plt.title("x pos")
# plt.show()
#
#
# print(trajectory[40])
# print(len(trajectory), len(t))
plt.plot(trajectory[:,6])
# plt.plot(t, trajectory[:][:][3])
# # plt.plot(t, trajectory[:][:][5])
plt.title("y pos")
plt.show()

# graphic_fric = np.array(graphic_fric)
# print(graphic_fric[:,0])
# plt.title("Friction")
# plt.plot(graphic_fric[:,0][:,0])
# plt.plot(graphic_fric[:,0][:,1])
# plt.show()
# plt.title("Torque")
# plt.plot(torques, label="Collision")
# # plt.plot(walltorques, label="Wall")
# plt.show()
# # #
# plt.title("friction options")
# plt.plot(np.array(friction_options)[:,0], label="kinetic")
# plt.plot(np.array(friction_options)[:,1], label="static")
# plt.plot(np.array(friction_options)[:,2], label="mag")
# max = max(np.array(friction_options)[:,0])
# plt.ylim(-max, max)
# plt.legend()
# plt.show()
plt.title("angular velocity")
plt.plot(omega)
plt.show()
plt.title("relative velocity")
plt.plot(relvelocity)
plt.show()
# plt.plot(unittan)
# plt.title("Unit tangent")
# plt.show()
# plt.title("Friction 2nd term")
# plt.plot(np.array(svector)[:,0])
# plt.plot(np.array(svector)[:,1])
# plt.show()
plt.plot(overlapN)
plt.title("Overlap N")
plt.show()
plt.plot(total_disp)
plt.title("Total displacement")
plt.show()
#
#
plt.show()
print(walltorques)
plt.plot(np.array(walltorques)[:,0],label="torque")
plt.plot(np.array(walltorques)[:,1],label="rolling torque")
plt.title("Wall torque")
plt.legend()
plt.show()


print(coltorques)
plt.plot(np.array(coltorques)[:,0],label="torque")
plt.plot(np.array(coltorques)[:,1],label="rolling torque")
plt.title("Collision torque")
plt.legend()
plt.show()
#====================================================
# plot system

# xmax = length*scalef
# ymax = len_m*scalef
# X = np.linspace(0, xmax, int(length*scalef))
# Y = np.linspace(0, ymax, int(len_m*scalef))
# plt.xlim(0,xmax)
# plt.ylim(0,ymax)
#
# plt.gca().set_aspect('equal', adjustable='box')
# plt.grid(b=True, which='major', color='#666666', linestyle='-')
#
# plt.plot((0, xmax/2, xmax), (ymax, scalef*(len_m+len_c)/2, ymax), c="blue")
# plt.plot((0, xmax/2, xmax), (0, scalef*(len_m-len_c)/2, 0), c="blue")
#
# plt.show()


v1 = []
v2 = []
v3 = []
# v4 = []
for i in velocities:
    # print(i)
    v1.append(i[0])
    v2.append(i[1])
    v3.append(i[2])
#     v4.append(i[3])
# #
plt.plot(t, v1, label='particle 1')
plt.plot(t, v2, label='particle 2')
plt.plot(t, v3, label='particle 3')
# plt.plot(t, v4, label='particle 4')
plt.legend()
# plt.plot(v4)
plt.xlabel('time')
plt.ylabel('velocity')
plt.title('Velocity of particles - 4 particle bridge')
# plt.xlim(75,85)
# plt.ylim(0,0.1)
# plt.xlim(70,104)
plt.show()
fcx = []
fwx = []
ffx = []
fricx = []
wf = []

usedtimes = []
print(forces)
print(len(forces))
for i in range(len(forces)):
    if(len(forces[i]) >0 ):
        # print(forces[i])
        # usedtimes.append(t[i])
        fricx.append(forces[i][3][0]/mass)
        fcx.append(forces[i][2][0]/mass)
        fwx.append(forces[i][1][0]/mass)
        ffx.append(forces[i][0][0]/mass)
        wf.append(forces[i][4][0]/mass)

plt.plot(fwx, label="wall")
plt.plot(fcx, label="collision")
plt.plot(ffx, label="fluid")
plt.plot(fricx, label="friction collision")
plt.plot(wf, label="friction wall")
# plt.plot(usedtimes, fwx, label="wall")
# plt.plot(usedtimes, fcx, label="collision")
# plt.plot(usedtimes, ffx, label="fluid")
# plt.plot(usedtimes, fricx, label="friction collision")
# plt.plot(usedtimes, wf, label="friction wall")
plt.title("Forces over time on one particle - X dir")
plt.ylabel("Force in x direction")
# plt.ylim(-2.5,2.5)
# plt.xlim(77,80)

plt.xlabel("time")
plt.legend()
plt.show()


# for i in range(len(forces)-2):
#     print(forces[i])
#     wf.append(forces[i][4][0]/mass)
#     fricx.append(forces[i][3][0]/mass)
#     fcx.append(forces[i][2][0]/mass)
#     fwx.append(forces[i][1][0]/mass)
#     ffx.append(forces[i][0][0]/mass)
#
# plt.plot(t, fwx, label="wall")
# plt.plot(t, fcx, label="collision")
# plt.plot(t, ffx, label="fluid")
# plt.plot(t, fricx, label="friction collision")
# plt.plot(t, wf, label="friction wall")
# plt.title("Forces over time on one particle - X dir")
# plt.ylabel("Force in x direction")
# # plt.ylim(-10,10)
# # plt.xlim(77,80)
#
# plt.xlabel("time")
# plt.legend()
# plt.show()


fcx = []
fwx = []
ffx = []
fricx = []
wf = []

for i in range(len(forces)):
    if (len(forces[i])>0):
        wf.append(forces[i][4][1]/mass)
        fricx.append(forces[i][3][1]/mass)
        fcx.append(forces[i][2][1]/mass)
        fwx.append(forces[i][1][1]/mass)
        ffx.append(forces[i][0][1]/mass)

plt.plot( fwx, label="wall")
plt.plot( fcx, label="collision")
plt.plot( ffx, label="fluid")
plt.plot( fricx, label="friction collision")
plt.plot( wf, label="friction wall")
plt.title("Forces over time on one particle - Y dir")
plt.ylabel("Force in y direction")
# plt.ylim(-2.5,2.5)
# plt.xlim(77,80)

plt.xlabel("time")
plt.legend()
plt.show()


print(trajectory[-1])
#====================================================
#Hessian

n = 4
R = 3

#Get the stable point at a certain timestep of a simulation
# fig, ax = plt.subplots()
# index = 0
#
# pos_next = []
# pos_stable = []
# for i in range(len(t)):
#     # print(t[i])
#     if (t[i] >=80):
#         index = i
#         for i in range(n):
#             x = (trajectory[:][index][0+i*4])
#             y = (trajectory[:][index][1+i*4])
#
#             pos_stable.append(x)
#             pos_stable.append(y)
# #         print(index, t[index])
# #
#         break
# for i in range(len(t)):
#     # print(t[i])
#     if (t[i] >=95):
#         index = i
#         for i in range(n):
#             x = (trajectory[:][index][0+i*4])
#             y = (trajectory[:][index][1+i*4])
#
#             pos_next.append(x)
#             pos_next.append(y)
# #         print(index, t[index])
# #
#         break
#
# print(pos_stable)
# print(pos_next)


#4 particle clog -std
# pos_next = [27.47321495344648, 24.45205662041595, 26.633047026833307, 36.17141872581759, 25.920913733889176, 30.230098526852228, 21.679938849893798, 25.988175746283222]
# pos_stable = [27.179643917021988, 24.231840291244293, 27.148660288125736, 35.7926523311934, 25.617269092372563, 30.00880906042497, 21.39192584378055, 25.747005392800148]

#2 particle clog - symmetric
# dx = 30 - 21.866253691763898
# dy = 30 - 24.016367818392556
# pos_stable = [30-dx, 30-dy, 30-dx, 30+dy]

#2 particle clog
# pos_stable = [21.866253691763898, 24.016367818392556, 21.866264412250516, 35.983624168376295]
# pos_next = [21.814000924460974, 23.977323175392222, 21.919122835932694, 35.94412037686787]

#3 particle clog
# pos_stable = [27.18370348898932, 24.24083953453755, 27.183709391670448, 35.75915605971447, 25.556907179942257, 29.999998633412062]
# pos_next = [27.196120630131837, 24.250168559785475, 27.19615425637689, 35.74980634385685, 25.536484471599348, 29.999992320234476]


#4 part w/ adhesion
# pos_stable = [27.25614994727328, 24.28977665111323, 27.074186906158857, 35.84296840920806, 25.614383390989165, 30.041497642439364, 21.43940419613143, 25.73656411539582]

# [27.439105778236783, 24.43161444676956, 27.141569069818853, 35.79363693789873, 25.405844621435975, 30.058928711420137, 21.477554661059088, 25.09892793462067]
# [28.1953402393687, 25.002000780955065, 27.24759917623977, 35.711602940273984, 25.098481720595018, 30.126007155968857, 22.19264795117, 24.884631859295677]

# pos_stable = [27.638578337803644, 24.58332373833967, 26.884545975494998, 35.97845434964382, 25.468053167593858, 30.159830394613646, 21.70765779855862, 25.48810404421516]
# pos_next = [28.35037978397013, 25.113614500653625, 26.530646174789098, 36.24741176600586, 25.467077286318215, 30.358075769903003, 22.35314199813014, 25.235808298669305]

# pos_stable  = [27.740314471711553, 24.65219266455878, 26.846259361905695, 36.01725410392413, 25.48303406117755, 30.19247290462759, 21.798480354963758, 25.463085946698897]
# pos_next = [28.75825875380239, 25.42099414057658, 26.391144947255693, 36.34972144072959, 25.485593677342997, 30.43291937515091, 22.766333822803787, 25.09068797113054]

#FRICTION
# pos_stable = [ 2.73009897e+02,  2.42257410e+02, 2.73005198e+02,  3.57746114e+02, 2.56706688e+02, 2.99999988e+02]


# print(pos_stable)

def Energy(pos, n, R):

    E = 0
    for i in range(n):
        x = pos[i*2+0]
        y = pos[i*2+1]

        for j in range(n):
            if j != i:
                distance = math.sqrt((x-pos[j*2])**2 + (y-pos[j*2+1])**2)

                #if the particles overlap
                if (distance < 2*R):
                    xj = pos[j*2]
                    yj = pos[j*2+1]

                    Fx, Fy, V = calcCollision(R, x, y, xj, yj)
                    E += V
                    # print(V)

                #Add attractive forces between particles
                # if (i==2 and j==0 or i==0 and j==2):[j*2], pos[j*2+1], R)
                # if (i==2 and j==3 or i==3 and j==2):
                #     E += attractivePotential(x, y, pos[j*2], pos[j*2+1], R)
                # if (i==3 and j==0 or i==0 and j==3):
                #     E += attractivePotential(x, y, pos[j*2], pos[j*2+1], R)

        #calculate the point on the edge of the particle which is closest to the wall
        #the edge of the particle is what matters, not the center
        #this is a vector parpendicular to the wall
        if (y <= len_m/2*scalef):
            direction = unitVec((-1, 1/slope))
        else:
            direction = unitVec((-1, -1/slope))
        wallX, wallY, V_wall= calcPotentialWall(x - direction[0]*R, y - direction[1]*R, slope)
        E += V_wall
        # print(V_wall)

    # print("E:  ", E)
    return E


def second_deriv_E(pos, n, R, i, j, di, dj):

    pos_1_1 = pos.copy()
    pos_1_1[i] += di
    pos_1_1[j] += dj
    # print(pos[i], pos_1_1[i], di)

    pos_1_neg1 = pos.copy()
    pos_1_neg1[i] += di
    pos_1_neg1[j] -= dj

    pos_neg1_1 = pos.copy()
    pos_neg1_1[i] -= di
    pos_neg1_1[j] += dj

    pos_neg1_neg1 = pos.copy()
    pos_neg1_neg1[i] -= di
    pos_neg1_neg1[j] -= dj

    #calculate the second derivative with respec to i and j
    derivE = (Energy(pos_1_1, n, R) - Energy(pos_1_neg1, n, R) - Energy(pos_neg1_1, n, R) + Energy(pos_neg1_neg1, n, R))/4/di/dj

    return derivE

#sample 2nd der method, uneeded
def second_deriv_one_var(pos, n, R, i, di):
    pos_plus = pos.copy()
    pos_plus[i] += di

    pos_neg = pos.copy()
    pos_neg[i] -= di

    derivE = (Energy(pos_plus, n, R) - 2 * Energy(pos, n, R) + Energy(pos_neg, n, R))/di**2

    return derivE

Hessian = np.zeros((n*2, n*2))
for i in range(n*2):
    for j in range(n*2):
        Hessian[i][j] = second_deriv_E(pos_stable, n, R, i, j, 10e-4, 10e-4)
# np.savetxt("hessian_diff.txt", Hessian)

# BD Hessian
xvel, yvel = interpolateVelFn(u, v, 1, 1, length*scalef, len_m*scalef)
bdHessian = np.zeros((n*2+1, n*2+1))
for i in range(n*2+1):
    for j in range(n*2+1):
        if i==0 and j%2==1:
            Fx, Fy = calcFluidForceNonDim(pos_stable[j-1], pos_stable[j], 0, 0, xvel, yvel)
            bdHessian[i][j] = Fx
            bdHessian[i][j+1] = Fy
        if j==0 and i%2==1 :
            Fx, Fy = calcFluidForceNonDim(pos_stable[i-1], pos_stable[i], 0, 0, xvel, yvel)
            bdHessian[i][j] = Fx
            bdHessian[i+1][j] = Fy
        # if i == 1 and j==6:
        #     Fx, Fy = attractiveForce(pos_stable[4], pos_stable[5], pos_stable[6], pos_stable[7], R)
        #     bdHessian[i][j] = Fx
        #     bdHessian[i][j+1] = Fy
        #     bdHessian[j][i] = Fx
        #     bdHessian[j+1][i] = Fy
        #     Fx, Fy = attractiveForce(pos_stable[6], pos_stable[7], pos_stable[4], pos_stable[5], R)
        #     bdHessian[i][j+2] = Fx
        #     bdHessian[i][j+3] = Fy
        #     bdHessian[j+2][i] = Fx
        #     bdHessian[j+3][i] = Fy
        elif i>0 and j>0:
            bdHessian[i][j] = second_deriv_E(pos_stable, n, R, i-1, j-1, 10e-4, 10e-4)
            # print(i,j,bdHessian[i][j])

#BD Hessian - separate constraints
# xvel, yvel = interpolateVelFn(u, v, 1, 1, length*scalef, len_m*scalef)
# bdHessian = np.zeros((n*3, n*3))
# for i in range(n*3):
#     for j in range(n*3):
#         if i < n and j-n==i*2:
#             Fx, Fy = calcFluidForceNonDim(pos_stable[j-n], pos_stable[j-n-1], 0, 0, xvel, yvel)
#             bdHessian[i][j] = Fx
#             bdHessian[i][j+1] = Fy
#             print("did force")
#         if j<n and i-n==j*2:
#             Fx, Fy = calcFluidForceNonDim(pos_stable[i-n], pos_stable[i-n-1], 0, 0, xvel, yvel)
#             bdHessian[i][j] = Fx
#             bdHessian[i+1][j] = Fy
#         elif i>=n and j>=n:
#             bdHessian[i][j] = second_deriv_E(pos_stable, n, R, i-n, j-n, 10e-4, 10e-4)

#Get Eigen values/vectors
w, v = linalg.eig(bdHessian)

print("Hessian:\n")
print(bdHessian)
# np.savetxt("bdhess_4part_dimer.csv", bdHessian, delimiter=',')

print("\n\nEigenvalues\n")
print(w)
print("\n\nEigenvectors\n")
print(v)

energy_stable = Energy(pos_stable, n, R)
print("Total Energy: "+str(energy_stable))

# # constraints in the bd Hessian - 1 row of fluids
constraints = 1

for i in range(n*2+constraints):
    eigvec = v[:,i][constraints:]
    eigvalue = w[i]

    new_pos = np.add(eigvec*10e-4, pos_stable)
    energy_new = Energy(new_pos, n, R)

    print("Does the energy decrease when system is shifted in direction of eigenvector?")
    print("New Energy #"+str(i)+" "+str(energy_new)+" " + str(energy_new<energy_stable)+ " "+str(eigvalue))

# for i in range(n*2+constraints):
#     vec = v[:,i][constraints:]
#     eigvalue = w[i]
#     steps = np.linspace(-10e-2, 10e-2, 1000)
#
#     energies = []
#     for j in steps:
#         new_pos = np.add(vec*j,pos_stable)
#         energies.append(Energy(new_pos, n, R)-energy_stable) #get CHANGE in energy
#
#     plt.plot(steps, energies)
#     plt.title("Change in energy around eigenvalue: " + str(w[i]))
#     plt.plot(steps, np.zeros((1000)))#plot a zero line
#     plt.show()

#Test eigenvalues are correctly matched with vectors
for i in range(n*2+constraints):
    print("Eigenvector corresponding to "+str(w[i]))
    print(v[:,i]*w[i])
    print(np.dot(bdHessian, v[:,i])) #these should be equal

#Plot all eigenvectors
# for i in range(n*2+constraints):
#     fig, ax = plt.subplots()
#
#     #plot particles
#     for j in range(n):
#
#         x = pos_stable[0 +j*2]
#         y = pos_stable[1 +j*2]
#
#         circle = Circle((0,0), R, color="black", fill=False)
#         circle.center = [x,y]
#         ax.add_artist(circle)
#
#     #plot vectors
#     ax.arrow(pos_stable[0], pos_stable[1], v[:,i][0+constraints]*3, v[:,i][1+constraints]*3, head_width=1)
#     ax.arrow(pos_stable[2], pos_stable[3], v[:,i][2+constraints]*3, v[:,i][3+constraints]*3, head_width=1)
#     ax.arrow(pos_stable[4], pos_stable[5], v[:,i][4+constraints]*3, v[:,i][5+constraints]*3, head_width=1)
#     ax.arrow(pos_stable[6], pos_stable[7], v[:,i][6+constraints]*3, v[:,i][7+constraints]*3, head_width=1)
#
#     #plot bd lines
#     xmax = length*scalef
#     ymax = len_m*scalef
#     plt.plot((0, xmax/2, xmax), (ymax, scalef*(len_m+len_c)/2, ymax), c="blue")
#     plt.plot((0, xmax/2, xmax), (0, scalef*(len_m-len_c)/2, 0), c="blue")
#
#     plt.gca().set_aspect('equal', adjustable='box')
#     plt.figtext(.5,.97,"Eigenvector with eigenvalue: " + str(w[i]), fontsize=10, ha='center')
#     plt.figtext(.5,.9,str(v[:,i]),fontsize=8,ha='center')
#     plt.ylim(0,60)
#     plt.xlim(0,60)
#     print("saving fig..."+str(i))
#     plt.savefig("bdhess_4part_dimer" + str(i))
#     plt.show()


#Directional Analysis
direction_of_motion = np.array(pos_next)-np.array(pos_stable)
dir_motion = direction_of_motion/np.linalg.norm(direction_of_motion)

print("Dir of motion: "+str(dir_motion))

directional_vectors = v[1:,1:]
print(directional_vectors)


# fig, ax = plt.subplots()

#plot particles
# for j in range(n):
#
#     x = pos_stable[0 +j*2]
#     y = pos_stable[1 +j*2]
#
#     circle = Circle((0,0), R, color="black", fill=False)
#     circle.center = [x,y]
#     ax.add_artist(circle)
#
# #plot vectors
# ax.arrow(pos_stable[0], pos_stable[1], dir_motion[0], dir_motion[1], head_width=1)
# ax.arrow(pos_stable[2], pos_stable[3], dir_motion[2], dir_motion[3], head_width=1)
# ax.arrow(pos_stable[4], pos_stable[5], dir_motion[4], dir_motion[5], head_width=1)
# ax.arrow(pos_stable[6], pos_stable[7], dir_motion[6], dir_motion[7], head_width=1)
#
# #plot bd lines
# xmax = length*scalef
# ymax = len_m*scalef
# plt.plot((0, xmax/2, xmax), (ymax, scalef*(len_m+len_c)/2, ymax), c="blue")
# plt.plot((0, xmax/2, xmax), (0, scalef*(len_m-len_c)/2, 0), c="blue")
#
# plt.gca().set_aspect('equal', adjustable='box')
# plt.title("Direction of Motion in Simulation")
# plt.ylim(0,60)
# plt.xlim(0,60)
# # print("saving fig..."+str(i))
# # plt.savefig("bdhess_4part_updated" + str(i))
# plt.show()

#Linear independance:

# coeffs = np.linalg.solve(v[1:,1:], np.zeros((n*2)))
# # print(v[:,1:])
# print("linear independance: ", coeffs)

# np.savetxt("eigenvec.csv", v)
# np.savetxt("eigenvec_col7.csv", v[:,7])
# np.savetxt("eigenvec_trimmed.csv", v[1:,1:])
# print(v[1:,1:])
# print(v[:,7])

coeffs = np.linalg.solve(directional_vectors, direction_of_motion)
coeffs_norm = coeffs/math.sqrt(np.dot(coeffs, coeffs)) #np.linalg.norm(coeffs)

print("coeffs*vectors", np.dot(directional_vectors, coeffs))
print(direction_of_motion)

# summed_vec = np.add(np.add(np.add(v[:,6][1:]*coeffs[5], v[:,7][1:]*coeffs[6]), v[:,8][1:]*coeffs[7]), v[:,5][1:]*coeffs[4])
# print(coeffs[5])
# print(coeffs[6])
# print(summed_vec)
# plt.plot(-v[:,6][1:], label="6-pos")
# plt.plot(v[:,7][1:], label="7-neg")
# plt.plot(summed_vec, coeffs, label="summed")
# plt.plot(dir_motion, label="actual")
# plt.legend()
# plt.show()


print(coeffs)
print(coeffs_norm)
print(w[1:])
labels = [str(round(i,3)) for i in w[1:]]

x_pos = [i for i, _ in enumerate(coeffs_norm)]
colors = []
for i in range(n*2):
    if (w[i+1] >0):
        colors.append('r')
    else:
        colors.append('b')

plt.bar(x_pos, coeffs_norm, color=colors)
plt.xticks(x_pos, labels)
plt.title("Particle Motion Comparaison with Eigenvectors: "+str(n)+" Particles")
plt.xlabel("Eigenvalue: red is positive, blue negative")
plt.ylabel("Coefficients (normalized) on associated eigenvector")
plt.show()

#=====================================================
#
#


#
#
# stable1 = np.where(v1 == np.amin(v1))
# print(stable1, np.amin(v1))
# stable2 = np.where(v2 == np.amin(v2))
# print(stable2, np.amin(v2))
# stable3 = np.where(v3 == np.amin(v3))
# print(stable3, np.amin(v3))
# stable4 = np.where(v4 == np.amin(v4))
# print(stable4, np.amin(v4))
#===Testing: adding/removing particles===

#0.026550599845902603

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
