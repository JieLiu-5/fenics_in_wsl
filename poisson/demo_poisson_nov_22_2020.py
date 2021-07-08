# 
# .. _demo_poisson_equation:
# 
# Poisson equation
# ================

from dolfin import *

import getopt, sys
import math
import time



full_cmd_arguments = sys.argv

argument_list = full_cmd_arguments[1:]


print('Number of arguments:', len(argument_list), 'arguments.')
print('Argument List:', str(argument_list))

print("")

degree_custom = int(argument_list[0])
n_refine = int(argument_list[1])


print("element degree:", degree_custom)
print("# of refinements:", n_refine)

print()


file_error = open('data_error.txt', 'a')
file_error.write("n_refine n_dofs error_l2 error_h1semi cpu_time\n")

i = 0
k = 1

n_cells = 1

grid_size = 1.0

n_dofs = 1


l2_error_u_lst = []
errornorm_u_lst = []
err_u_lst = []

while i<=n_refine-1:
    
    t = time.time()
    
    print("########################################")
    print(i+1,"th refinement")
    print("# of rectangular cells in one direction:",k)
    
    # Create mesh and define function space
    mesh = UnitSquareMesh(k, k,"crossed")
    #mesh = RectangleMesh(Point(0.0,0.0), Point(1.0,1.0), k, k, "crossed")

    n_cells = mesh.num_cells()
    
    grid_size = 1.0/(2.0*k)
    
    print("# of cells:", n_cells)
    #print("# of edges:", mesh.num_edges())
    #print("coordinates of the mesh\n",mesh.coordinates())

    print("grid size:", grid_size)
    

    V = FunctionSpace(mesh, "Lagrange", degree_custom)

    #print(V.dolfin_element())
    
    dofmap = V.dofmap()

    dofs = dofmap.dofs()
    n_dofs = len(dofs)
    
    print("# of dofs:",n_dofs)
    #print(dofs)
    
    ## Get coordinates as len(dofs) x gdim array
    #dofs_x = dofmap.tabulate_all_coordinates(mesh).reshape((-1, gdim))

    #for dof, dof_x in zip(dofs, dofs_x):
        #print (dof, ':', dof_x)

    # Define Dirichlet boundary (x = 0 or x = 1)
    def boundary(x):
        #print("boundary()", end = " ")
        #print(x)
        return x[0] < DOLFIN_EPS or x[0] > 1.0 - DOLFIN_EPS

    #print('DOLFIN_EPS:', DOLFIN_EPS)


    #print("Dirichlet boundary")
    #for x in mesh.coordinates():
        #if(boundary(x)):
            #print(x)
    

    # Define boundary condition
    #u0 = Constant(0.0)
    #u0 = Expression(("1.0 + pow(x[0]-0.5,2) + pow(x[1]-0.5,2)"), degree = 2)
    u0 = Expression(("exp(-pow(x[0]-0.5,2))"), degree = degree_custom)


    #print("Info of u0:")
    #print("name", end = ": ")
    #print(u0.name())
    #print("id", end = ": ")
    #print(u0.id())
    #print("value size", end = ": ")
    #print(u0.value_size())
    #print("values", end = ": ")
    #print(u0.values())


    bc = DirichletBC(V, u0, boundary)


    # Define variational problem
    u = TrialFunction(V)
    v = TestFunction(V)
    #f = Expression("10*exp(-(pow(x[0] - 0.5, 2) + pow(x[1] - 0.5, 2)) / 0.02)", degree=2)
    #f = Expression("-4", degree=2)
    f = Expression("-exp(-pow(x[0]-0.5,2))*(pow(2*(x[0]-0.5),2)-2)", degree=degree_custom)



    #g = Expression("sin(5*x[0])", degree=2)
    #g = Expression("1.0", degree=2)                         # already taking the normal vector into account
    g = Expression("0.0", degree=degree_custom)


    a = inner(grad(u), grad(v))*dx
    L = f*v*dx + g*v*ds

    # Compute solution
    u = Function(V)
    solve(a == L, u, bc)


    # Save solution in VTK format
    file = File("poisson.pvd")
    file << u
    
    x = SpatialCoordinate(mesh)
    uexact = exp(-pow(x[0]-0.5,2))
    M = (u - uexact)**2*dx(degree=5)                            # adopted in undocumented/poisson-disc.py
    #M0 = uexact**2*dx(degree=5)
    err_u = sqrt(assemble(M))              # / assemble(M0))
        
    err_u_lst.append(err_u)
    print("error using assemble()",err_u)
    
    
    

    #u_e = Function(V)
    ##u_e.assign(Expression("1.0 + pow(x[0]-0.5,2) + pow(x[1]-0.5,2)", degree=2))
    #u_e.assign(Expression("exp(-pow(x[0]-0.5,2))", degree=degree_custom+2))

    #errornorm_u = errornorm(u_e,u, 'l2', 3,mesh)
    
    #print("error using errornorm(): ", errornorm_u) 
    #errornorm_u_lst.append(errornorm_u)
    
    
    #u_e = u_e.vector()
    #u = u.vector()
    
    #diff_u = u-u_e
    #l2_error_u = norm(diff_u, 'l2',3)
    
    #print("error using norm(): ", l2_error_u)
    
    #l2_error_u_lst.append(l2_error_u)
    
    elapsed = time.time() - t
    
    print("time elapsed:", "{:6.4f}".format(elapsed))
    


    file_error.write("%s %s %0.2e %s %0.2e\n" %(i+1, n_dofs, err_u, 1, elapsed))
    
    i += 1
    k = pow(2,i)

print("")
print("summary of the err")
for i in range(len(err_u_lst)):
    print(err_u_lst[i])
    
print("")
print("order of convergence for err")
for i in range(len(err_u_lst)-1):
    print(math.log(err_u_lst[i]/err_u_lst[i+1],10)/math.log(2,10))    
    
    
    
#print("")
#print("summary of the errornorm_u")
#for i in range(len(errornorm_u_lst)):
    #print(errornorm_u_lst[i])
    
#print("")
#print("order of convergence for errornorm_u")
#for i in range(len(errornorm_u_lst)-1):
    #print(math.log(errornorm_u_lst[i]/errornorm_u_lst[i+1],10)/math.log(2,10))  
    
    
    
    
#print("")
#print("summary of the error")
#for i in range(len(l2_error_u_lst)):
    #print(l2_error_u_lst[i])
    
    
#print("")
#print("order of convergence")
#for i in range(len(l2_error_u_lst)-1):
    #print(math.log(l2_error_u_lst[i]/l2_error_u_lst[i+1],10)/math.log(2,10))    

file_error.close() 


# Plot solution
#import matplotlib.pyplot as plt
#plot(u)
#plt.show()
