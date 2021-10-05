from __future__ import print_function
from fenics import *
from dolfin import *
from BCObject import *
import random
import time
import numpy as np
import glob, os
# import dlib

mu = Constant(1.0)                   # viscosity
alphaunderbar = 2.5 * mu / (100**2)  # parameter for \alpha
alphabar = 2.5 * mu / (0.01**2)      # parameter for \alpha
q = Constant(0.01) # q value that controls difficulty/discrete-valuedness of solution

def alpha(rho):
    """Inverse permeability as a function of rho, equation (40)"""
    return alphabar + (alphaunderbar - alphabar) * rho * (1 + q) / (rho + q)

# Mesh boundaries
lefBound = 0.00
rigBound = 0.42
botBound = 0.21
topBound = 0.63
#N = 63

#mesh = RectangleMesh(Point(lefBound, botBound), Point(rigBound, topBound), 127, 63)
mesh = Mesh("/home/jun/Documents/Magneto/FEniCSSimulations/AutoWriter/mesh/Lshape_new.xml") # Load mesh
coords = mesh.coordinates()
ind_x0 = np.where(coords[:,0] == 0)
coords_inlet = np.sort(coords[ind_x0, 1])
# print(coords_inlet)

A = FunctionSpace(mesh, "CG", 1)        # control function space
U_h = VectorElement("CG", mesh.ufl_cell(), 2)
P_h = FiniteElement("CG", mesh.ufl_cell(), 1)
W = FunctionSpace(mesh, U_h*P_h)          # mixed Taylor-Hood function space

def forward(rho):
    """Solve the forward problem for a given fluid distribution rho(x)."""
    w = Function(W)
    (u, p) = split(w)
    (v, q) = TestFunctions(W)

    nu = Constant(0.01)
    f = Constant((0.0, 0.0))
    
    # Is this equation right?
    F = alpha(rho)*inner(dot(grad(u), u), v)*dx + alpha(rho)*nu*inner(grad(u), grad(v))*dx - alpha(rho)*inner(p, div(v))*dx + alpha(rho)*inner(q, div(u))*dx + inner(f, v)*dx # Navier Stokes
#    F = alpha(rho)*inner(dot(grad(u), u), v)*dx + alpha(rho)*nu*inner(grad(u), grad(v))*dx - alpha(rho)*inner(p, div(v))*dx + alpha(rho)*inner(q, div(u))*dx + inner(f, v)*dx # Navier Stokes
#    F = inner(u, v)*dx + inner(grad(u), grad(v))*dx - inner(grad(p), v)*dx + inner(div(u), q)*dx #+ inner(f, v)*dx # Navier Stokes
    
    # Boundary condition assignment
    bc = [DirichletBC(W.sub(0), inVelVal1, inlet_BC1()),        
 	      DirichletBC(W.sub(0), inVelVal2, inlet_BC2()),
		  DirichletBC(W.sub(0), inVelVal3, inlet_BC3()),
 	      DirichletBC(W.sub(0), inVelVal4, inlet_BC4()),
		  DirichletBC(W.sub(0), inVelVal5, inlet_BC5()),
 	      DirichletBC(W.sub(0), inVelVal6, inlet_BC6()),
		  DirichletBC(W.sub(0), inVelVal7, inlet_BC7()),
 	      DirichletBC(W.sub(0), inVelVal8, inlet_BC8()),
 	      DirichletBC(W.sub(0), inVelVal9, inlet_BC9()),
		  DirichletBC(W.sub(0), inVelVal10, inlet_BC10()),
 	      DirichletBC(W.sub(0), inVelVal11, inlet_BC11()),
		  DirichletBC(W.sub(0), inVelVal12, inlet_BC12()),
 	      DirichletBC(W.sub(0), inVelVal13, inlet_BC13()),
		  DirichletBC(W.sub(0), inVelVal14, inlet_BC14()),
		  DirichletBC(W.sub(0), inVelVal15, inlet_BC15()),
 	      DirichletBC(W.sub(0), inVelVal16, inlet_BC16()),
		  DirichletBC(W.sub(0), inVelVal17, inlet_BC17()),
 	      DirichletBC(W.sub(0), inVelVal18, inlet_BC18()),
		  DirichletBC(W.sub(0), inVelVal19, inlet_BC19()),	
          # DirichletBC(W.sub(0), inVelVal20, inlet_BC20()),
          DirichletBC(W.sub(0), inVelVal1, point_BC1(),method='pointwise'),
          # DirichletBC(W.sub(0), inVelVal2, point_BC2(),method='pointwise'),
          # DirichletBC(W.sub(0), inVelVal3, point_BC3(),method='pointwise'),
          # DirichletBC(W.sub(0), inVelVal4, point_BC4(),method='pointwise'),
          # DirichletBC(W.sub(0), inVelVal5, point_BC5(),method='pointwise'),
          # DirichletBC(W.sub(0), inVelVal6, point_BC6(),method='pointwise'),
          # DirichletBC(W.sub(0), inVelVal7, point_BC7(),method='pointwise'),
          # DirichletBC(W.sub(0), inVelVal8, point_BC8(),method='pointwise'),
          # DirichletBC(W.sub(0), inVelVal9, point_BC9(),method='pointwise'),
          # DirichletBC(W.sub(0), inVelVal10, point_BC10(),method='pointwise'),
          # DirichletBC(W.sub(0), inVelVal11, point_BC11(),method='pointwise'),
          # DirichletBC(W.sub(0), inVelVal12, point_BC12(),method='pointwise'),
          # DirichletBC(W.sub(0), inVelVal13, point_BC13(),method='pointwise'),
          # DirichletBC(W.sub(0), inVelVal14, point_BC14(),method='pointwise'),
          # DirichletBC(W.sub(0), inVelVal15, point_BC15(),method='pointwise'),
          # DirichletBC(W.sub(0), inVelVal16, point_BC16(),method='pointwise'),
          # DirichletBC(W.sub(0), inVelVal17, point_BC17(),method='pointwise'),
          # DirichletBC(W.sub(0), inVelVal18, point_BC18(),method='pointwise'),
          # DirichletBC(W.sub(0), inVelVal19, point_BC19(),method='pointwise'),
          DirichletBC(W.sub(0), inVelVal20, point_BC20(),method='pointwise'),
           # bc = [DirichletBC(W.sub(0), inVelVal, inObj),
          DirichletBC(W.sub(1), outPreVal, outObj),
          DirichletBC(W.sub(0), noSlip.value, notBC(inObj, outObj))]
    solve(F == 0, w, bcs=bc)

    return w

#def inflowVelExpression(k):
#    if not isinstance(k, int):
#        print("Possible error: K is not an integer.")
#        
#    velExp = None
#    
#    if k == 0:
#        velExp = '1'
#    else:
#        velExp = '(1 - pow(6*(x[1] - 0.315), ' + str(k) + '))'
#        
#    return velExp

# Inlet boundary region
class Lshape_in(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and ((0.21 < x[1] < 0.42 and near(x[0], lefBound)))

# Outlet boundary region
class Lshape_out(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and ((0.21 < x[0] < 0.42 and near(x[1], topBound)))
    
# Point-wise location of inlet of pipes (21 points at inlet, totally 64 points on boundary based on the resolution)		
class inlet_BC1(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and ((coords_inlet[0,1] <= x[1] <= coords_inlet[0,2] and near(x[0], lefBound)))
class inlet_BC2(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and ((coords_inlet[0,2] <= x[1] <= coords_inlet[0,3] and near(x[0], lefBound)))
class inlet_BC3(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and ((coords_inlet[0,3] <= x[1] <= coords_inlet[0,4] and near(x[0], lefBound)))
class inlet_BC4(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and ((coords_inlet[0,4] <= x[1] <= coords_inlet[0,5] and near(x[0], lefBound)))
class inlet_BC5(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and ((coords_inlet[0,5] <= x[1] <= coords_inlet[0,6] and near(x[0], lefBound)))
class inlet_BC6(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and ((coords_inlet[0,6] <= x[1] <= coords_inlet[0,7] and near(x[0], lefBound)))
class inlet_BC7(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and ((coords_inlet[0,7] <= x[1] <= coords_inlet[0,8] and near(x[0], lefBound)))
class inlet_BC8(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and ((coords_inlet[0,8] <= x[1] <= coords_inlet[0,9] and near(x[0], lefBound)))
class inlet_BC9(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and ((coords_inlet[0,9] <= x[1] <= coords_inlet[0,10] and near(x[0], lefBound)))
class inlet_BC10(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and ((coords_inlet[0,10] <= x[1] <= coords_inlet[0,11] and near(x[0], lefBound)))
class inlet_BC11(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and ((coords_inlet[0,11] <= x[1] <= coords_inlet[0,12] and near(x[0], lefBound)))
class inlet_BC12(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and ((coords_inlet[0,12] <= x[1] <= coords_inlet[0,13] and near(x[0], lefBound)))
class inlet_BC13(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and ((coords_inlet[0,13] <= x[1] <= coords_inlet[0,14] and near(x[0], lefBound)))
class inlet_BC14(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and ((coords_inlet[0,14] <= x[1] <= coords_inlet[0,15] and near(x[0], lefBound)))
class inlet_BC15(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and ((coords_inlet[0,15] <= x[1] <= coords_inlet[0,16] and near(x[0], lefBound)))
class inlet_BC16(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and ((coords_inlet[0,16] <= x[1] <= coords_inlet[0,17] and near(x[0], lefBound)))
class inlet_BC17(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and ((coords_inlet[0,17] <= x[1] <= coords_inlet[0,18] and near(x[0], lefBound)))
class inlet_BC18(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and ((coords_inlet[0,18] <= x[1] <= coords_inlet[0,19] and near(x[0], lefBound)))
class inlet_BC19(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and ((coords_inlet[0,19] <= x[1] <= coords_inlet[0,20] and near(x[0], lefBound)))
# class inlet_BC20(SubDomain):
#     def inside(self, x, on_boundary):
#         return on_boundary and ((coords_inlet[0,20] <= x[1] <= coords_inlet[0,21] and near(x[0], lefBound)))

# point-wise location of inlet of pipes (20 points at inlet, totally 22 points on boundary based on the resolution)   
class point_BC1(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[1], coords_inlet[0,1]) and near(x[0], lefBound)
class point_BC2(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[1], coords_inlet[0,2]) and near(x[0], lefBound)
class point_BC3(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[1], coords_inlet[0,3]) and near(x[0], lefBound)
class point_BC4(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[1], coords_inlet[0,4]) and near(x[0], lefBound)
class point_BC5(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[1], coords_inlet[0,5]) and near(x[0], lefBound)
class point_BC6(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[1], coords_inlet[0,6]) and near(x[0], lefBound)
class point_BC7(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[1], coords_inlet[0,7]) and near(x[0], lefBound)
class point_BC8(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[1], coords_inlet[0,8]) and near(x[0], lefBound)
class point_BC9(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[1], coords_inlet[0,9]) and near(x[0], lefBound)
class point_BC10(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[1], coords_inlet[0,10]) and near(x[0], lefBound)
class point_BC11(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[1], coords_inlet[0,11]) and near(x[0], lefBound)
class point_BC12(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[1], coords_inlet[0,12]) and near(x[0], lefBound)
class point_BC13(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[1], coords_inlet[0,13]) and near(x[0], lefBound)
class point_BC14(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[1], coords_inlet[0,14]) and near(x[0], lefBound)
class point_BC15(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[1], coords_inlet[0,15]) and near(x[0], lefBound)
class point_BC16(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[1], coords_inlet[0,16]) and near(x[0], lefBound)
class point_BC17(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[1], coords_inlet[0,17]) and near(x[0], lefBound)
class point_BC18(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[1], coords_inlet[0,18]) and near(x[0], lefBound)
class point_BC19(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[1], coords_inlet[0,19]) and near(x[0], lefBound)
class point_BC20(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[1], coords_inlet[0,20]) and near(x[0], lefBound)

if __name__ == "__main__":
    #random.seed(0)
    
    outerFolderName = "output/DeCNN_database_journal/DeCNN_simulation_output/Lshape_sepSimu_dis_first/"
#    meshFileXML = File(outerFolderName + "mesh.xml")
#    meshFileXML << mesh

    inObj = Lshape_in()
    outObj = Lshape_out()
    
    print("Start time: " + str(time.ctime()))
    startTime = time.time()
    
    for z in range(1): # number of iterations
        print("Iteration " + str(z))
#        i = 1        
        rho = interpolate(Expression("x[1] > 0", degree = 1), A) # L-shape up
        
#        Vel_mult = random.uniform(0.0, 1.0) #0.01 # Change here
#        Vel_mult = 1.0  
        
        for path, _, files in os.walk("output/DeCNN_database_journal/DeCNN_simulation_input/Lshape_inVel_first"):
            for file in glob.glob(os.path.join(path, "*.npy")):
                velocity_field = np.load(file)
                # Outlet velocities at each point in direction of x-axis
                velocity_x1 = velocity_field[19, 0]
                velocity_x2 = velocity_field[18, 0]
                velocity_x3 = velocity_field[17, 0]
                velocity_x4 = velocity_field[16, 0]
                velocity_x5 = velocity_field[15, 0]
                velocity_x6 = velocity_field[14, 0]
                velocity_x7 = velocity_field[13, 0]
                velocity_x8 = velocity_field[12, 0]
                velocity_x9 = velocity_field[11, 0]
                velocity_x10 = velocity_field[10, 0]
                velocity_x11 = velocity_field[9, 0]
                velocity_x12 = velocity_field[8, 0]
                velocity_x13 = velocity_field[7, 0]
                velocity_x14 = velocity_field[6, 0]
                velocity_x15 = velocity_field[5, 0]
                velocity_x16 = velocity_field[4, 0]
                velocity_x17 = velocity_field[3, 0]
                velocity_x18 = velocity_field[2, 0]
                velocity_x19 = velocity_field[1, 0]
                velocity_x20 = velocity_field[0, 0]
                
                # Outlet velocities at each point in direction of y-axis
                velocity_y1 = velocity_field[19, 1]
                velocity_y2 = velocity_field[18, 1]
                velocity_y3 = velocity_field[17, 1]
                velocity_y4 = velocity_field[16, 1]
                velocity_y5 = velocity_field[15, 1]
                velocity_y6 = velocity_field[14, 1]
                velocity_y7 = velocity_field[13, 1]
                velocity_y8 = velocity_field[12, 1]
                velocity_y9 = velocity_field[11, 1]
                velocity_y10 = velocity_field[10, 1]
                velocity_y11 = velocity_field[9, 1]
                velocity_y12 = velocity_field[8, 1]
                velocity_y13 = velocity_field[7, 1]
                velocity_y14 = velocity_field[6, 1]
                velocity_y15 = velocity_field[5, 1]
                velocity_y16 = velocity_field[4, 1]
                velocity_y17 = velocity_field[3, 1]
                velocity_y18 = velocity_field[2, 1]
                velocity_y19 = velocity_field[1, 1]
                velocity_y20 = velocity_field[0, 1]
                
                # Define inlet velocities at each point for the next pipes
    #            inVelVal = Expression(('Vel_mult*' + inflowVelExpression(k), '0.0'), Vel_mult = Vel_mult, degree=2) # Inlet velocity
                inVelVal1 = Expression(('velocity_x','velocity_y'), velocity_x = velocity_x1, velocity_y = velocity_y1, degree=2)
                inVelVal2 = Expression(('velocity_x','velocity_y'), velocity_x = velocity_x2, velocity_y = velocity_y2, degree=2)
                inVelVal3 = Expression(('velocity_x','velocity_y'), velocity_x = velocity_x3, velocity_y = velocity_y3, degree=2)
                inVelVal4 = Expression(('velocity_x','velocity_y'), velocity_x = velocity_x4, velocity_y = velocity_y4, degree=2)
                inVelVal5 = Expression(('velocity_x','velocity_y'), velocity_x = velocity_x5, velocity_y = velocity_y5, degree=2)
                inVelVal6 = Expression(('velocity_x','velocity_y'), velocity_x = velocity_x6, velocity_y = velocity_y6, degree=2)
                inVelVal7 = Expression(('velocity_x','velocity_y'), velocity_x = velocity_x7, velocity_y = velocity_y7, degree=2)
                inVelVal8 = Expression(('velocity_x','velocity_y'), velocity_x = velocity_x8, velocity_y = velocity_y8, degree=2)
                inVelVal9 = Expression(('velocity_x','velocity_y'), velocity_x = velocity_x9, velocity_y = velocity_y9, degree=2)
                inVelVal10 = Expression(('velocity_x','velocity_y'), velocity_x = velocity_x10, velocity_y = velocity_y10, degree=2)
                inVelVal11 = Expression(('velocity_x','velocity_y'), velocity_x = velocity_x11, velocity_y = velocity_y11, degree=2)
                inVelVal12 = Expression(('velocity_x','velocity_y'), velocity_x = velocity_x12, velocity_y = velocity_y12, degree=2)
                inVelVal13 = Expression(('velocity_x','velocity_y'), velocity_x = velocity_x13, velocity_y = velocity_y13, degree=2)
                inVelVal14 = Expression(('velocity_x','velocity_y'), velocity_x = velocity_x14, velocity_y = velocity_y14, degree=2)
                inVelVal15 = Expression(('velocity_x','velocity_y'), velocity_x = velocity_x15, velocity_y = velocity_y15, degree=2)
                inVelVal16 = Expression(('velocity_x','velocity_y'), velocity_x = velocity_x16, velocity_y = velocity_y16, degree=2)
                inVelVal17 = Expression(('velocity_x','velocity_y'), velocity_x = velocity_x17, velocity_y = velocity_y17, degree=2)
                inVelVal18 = Expression(('velocity_x','velocity_y'), velocity_x = velocity_x18, velocity_y = velocity_y18, degree=2)
                inVelVal19 = Expression(('velocity_x','velocity_y'), velocity_x = velocity_x19, velocity_y = velocity_y19, degree=2)
                inVelVal20 = Expression(('velocity_x','velocity_y'), velocity_x = velocity_x20, velocity_y = velocity_y20, degree=2)
                
                outPreVal = Constant(0.0) # Outlet pressure
                noSlip = fluidVelBC(value = ('0.0', '0.0')) # Noslip velocity
                
                w   = forward(rho) # Calculate Navier Stokes
                
                (u, p) = w.split(deepcopy = True)
                
                npyname = os.path.basename(file)
                filename = os.path.splitext(npyname)[0]
#                filename = filename[:-6]
                folderName = outerFolderName + filename + "/"
                print(folderName)
                velFilePVD = File(folderName + "fluid/u.pvd")
                preFilePVD = File(folderName + "fluid/p.pvd")
                rhoFilePVD = File(folderName + "fluid/rho.pvd")
    #                velFileXML = File(folderName + "fluid/u.xml")
    #                preFileXML = File(folderName + "fluid/p.xml")
    #                rhoFileXML = File(folderName + "fluid/rho.xml")
    #            
                rhoFilePVD << rho
                velFilePVD << u
                preFilePVD << p
    #                
    #                rhoFileXML << rho
    #                velFileXML << u
    #                preFileXML << p
                
    #                with XDMFFile(folderName + "data.xdmf") as outfile:
    #                    outfile.write_checkpoint(u, "velocity")
    #                    outfile.write_checkpoint(p, "pressure")
    #                    outfile.write_checkpoint(rho, "rho")
    #                    outfile.read_checkpoint(prevInitializedCompatibleFunction, "solutionFieldName")
                    
    #            f = open(folderName + "velocityValues.txt",'a')
    #            f.write(str(z) + ', ' + str(i) + ', ' + str(k) + ', ' + str(Vel_mult) + '\n')
    #            f.close()
                
    endTime = time.time()
    print("Seconds elapsed: " + str(endTime - startTime))
    print("End time: " + str(time.ctime()))
