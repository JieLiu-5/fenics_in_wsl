# 
# .. _demo_poisson_equation:
# 
# Poisson equation
# ================

import sys
sys.path.append('../')

from classes_shared.demo_classes_shared import *


id_dim = 0

gdim = id_dim + 1


full_cmd_arguments = sys.argv

argument_list = full_cmd_arguments[1:]

print('Number of arguments:', len(argument_list), 'arguments.')
print('Argument List:', str(argument_list))


if len(argument_list) != 7:
    raise Exception('Using <id_solution> <segment> <id_coeff_diff> <segment> <degree> <initial refinement level> <total refinements>')


id_solution = int(argument_list[0])

id_coeff_diff = int(argument_list[2])

degree_custom = int(argument_list[4])
level_initial_refinement=int(argument_list[5])
total_refinements = int(argument_list[6])


  
print("")
print('{:>25} {:>2}'.format("dimension:", vec_dim[id_dim]))
print('{:>25} {:>2}'.format("id_solution:", id_solution))
print('{:>25} {:>2}'.format("id_coeff_diff:", id_coeff_diff))
print('{:>25} {:>2}'.format("element degree:", degree_custom))
print('{:>25} {:>2}'.format("Initial refinement level:", level_initial_refinement))
print('{:>25} {:>2}'.format("# of total refinements:", total_refinements))
  
print("")
obj_string_u = String_U(id_solution)
print('{:>10} {:>2}'.format("u:", obj_string_u.func_value_u()))
print('{:>10} {:>2}'.format(r"u_x:", obj_string_u.func_gradient_u_in_x_direction()))
print('{:>10} {:>2}'.format(r"u_y:", obj_string_u.func_gradient_u_in_y_direction()))
print('{:>10} {:>2}'.format("\Delta u:", obj_string_u.func_delta_u()))

print("")
obj_string_coeff_diff = String_Coeff_Diff(id_coeff_diff)
print('{:>10} {:>2}'.format("D:", obj_string_coeff_diff.func_value_coeff_diff()))
print('{:>10} {:>2}'.format(r"\nabla D:", obj_string_coeff_diff.func_gradient_coeff_diff()))
    
print("")
obj_string_neumann_boundary = String_Neumann_Boundary(id_solution, id_coeff_diff)
print('{:>33} {:>2}'.format("neumann_boundary_in_x_direction:", obj_string_neumann_boundary.func_neumann_boundary_in_x_direction()))
print('{:>33} {:>2}'.format("neumann_boundary_in_y_direction:", obj_string_neumann_boundary.func_neumann_boundary_in_y_direction()))

print("")
obj_string_f = String_F(id_solution, id_coeff_diff)
print("f:", obj_string_f.func_f())
 

print("")

file_error = open('data_output_error_1_fenics_0_sm_0_real.txt', 'a')
file_error.write("current_refinement_level n_vertices n_dofs error_l2 error_h1_semi error_h2_semi l2_norm_u l2_norm_coeff_diff l2_norm_coeff_helm cpu_time\n")

current_refinement_level = level_initial_refinement
id_refinement = 0
n_rect_cell_one_direction = pow(2,level_initial_refinement)




array_error = np.zeros((total_refinements,3))
array_order_of_convergence = np.zeros((total_refinements-1,3))
array_time_cpu = np.zeros((total_refinements,1))



while id_refinement < total_refinements:
    
    t = time.time()
    
    print("")
    print("########################################")
    print("refinement level: ", current_refinement_level)                                # initial refinement level is denoted by i=0
    print("    # of rectangular cells in one direction:", n_rect_cell_one_direction)
    
    # Create mesh and define function space
    
    if id_dim == 0:
        mesh = IntervalMesh(n_rect_cell_one_direction, 0, 1)
    elif id_dim == 1:
        mesh = UnitSquareMesh(n_rect_cell_one_direction, n_rect_cell_one_direction,"crossed")
    #mesh = RectangleMesh(Point(0.0,0.0), Point(1.0,1.0), n_rect_cell_one_direction, n_rect_cell_one_direction, "crossed")

    n_cells = mesh.num_cells()
    n_vertices = mesh.num_vertices()
    
    grid_size = 1.0/(2.0*n_rect_cell_one_direction)
    
    #coor = mesh.coordinates()
    #print("    coordinates of the grid")
    #for index_coor in range(len(coor)):
        #print("        {:2.2e} {:2.2e}".format(coor[index_coor][0],coor[index_coor][1]))
    
    print("    # of active cells:", n_cells)
    print("    # of vertices:", n_vertices)
    #print("# of edges:", mesh.num_edges())
    #print("coordinates of the mesh\n",mesh.coordinates())

    print("    grid size of the active rectangular cell:", grid_size)
    
    
    print("Initializing the function space")
    
    V = FunctionSpace(mesh, "Lagrange", degree_custom)
    
    
    dofmap = V.dofmap()
    dofs = dofmap.dofs()
    n_dofs = len(dofs)
    
    print("    # of dofs:",n_dofs)
    #print(dofs)
    
    ## Get coordinates as len(dofs) x gdim array
    dofs_x = V.tabulate_dof_coordinates().reshape((-1, gdim))

    #print("    coordinates of dofs(", n_dofs, "):")
    #for dof, dof_x in zip(dofs, dofs_x):
        #print ("        ", dof, ':', dof_x)
        
        
    #Quad = FunctionSpace(mesh, "CG", 2)                            # to do: nr. of quadrature points
        
    #xq = V.tabulate_all_coordinates(mesh).reshape((-1, gdim))
    #xq0 = xq[V.sub(0).dofmap().dofs()]
    
    
    

    # Define Dirichlet boundary (x = 0 or x = 1)
    def boundary_diri_location(x):
        #print("boundary()", end = " ")
        #print(x)
        return  x[0] < DOLFIN_EPS or x[0] > 1.0 - DOLFIN_EPS         # 

    #print('DOLFIN_EPS:', DOLFIN_EPS)


    #print("Location of Dirichlet boundary")
    #for x in mesh.coordinates():
        #if(boundary_diri_location(x)):
            #print(x)

    x = SpatialCoordinate(mesh)                             # important for eval()
    
    
    # Define boundary condition
    
    #u0 = Constant(0.0)
    u0 = Expression(obj_string_u.func_value_u(), degree = degree_custom)                               # 
                                                                            # change this for different problems, and also
                                                                            # 1. the rhs
                                                                            # 2. the projection of \nabla u on the boundary: \nabla u \cdot \underline(n), on line ~143
                                                                            # 3. the exact solution on line ~171
                                                                            
    bc_diri = DirichletBC(V, u0, boundary_diri_location)


    # Define variational problem
    u = TrialFunction(V)
    v = TestFunction(V)
    
    f = Expression(obj_string_f.func_f(), degree = degree_custom)

    n = FacetNormal(mesh)   
    
    if id_dim == 0:
        expression_neumann_boundary = Expression(obj_string_neumann_boundary.func_neumann_boundary_in_x_direction(), degree = degree_custom)
        g = expression_neumann_boundary * n[0]
    elif id_dim == 1:
        expression_neumann_boundary = Expression((obj_string_neumann_boundary.func_neumann_boundary_in_x_direction(), obj_string_neumann_boundary.func_neumann_boundary_in_y_direction()), degree = degree_custom)
        
        g = dot(expression_neumann_boundary, n)
    
    #print(Cell(mesh, ufc_cell.index).normal(ufc_cell.local_facet))
    

    a = inner(grad(u), grad(v))*dx                              
                                                                        # a is a bilinear form
                                                                        # eval(String_Coeff_Diff(id_coeff_diff).func_value_coeff_diff()) * 
    
    L = f*v*dx + g *v*ds 
    
    #print("LHS before applying BC:")
    #LHS= assemble(a)
    #print(LHS.array())
    
    #print("RHS before applying Neumann BC:")
    #RHS= assemble(L)
    #print(RHS.get_local())    
    
    
    #L += g*dot(v,n[0])*ds
    
    #L += u0*dot(g, n)*ds
    
    #if x[1] > 1-DOLFIN_EPS:
    #L += g*v*ds

    #print("RHS after applying Neumann BC (no matter if it exists):")
    #RHS_after_neum_bc= assemble(L)
    #print(RHS_after_neum_bc.get_local())       
    
    print("Solving the system")
    
    u = Function(V)
    solve(a == L, u, bc_diri)                           # a == L is the equation for the variational form
                                                        # , solver_parameters={'linear_solver' : 'mumps'}
    
    #print("LHS after solving:")
    #LHS_after_solving= assemble(a)
    #print(LHS_after_solving.array())
    
    #print("RHS after solving:")
    #RHS_after_solving= assemble(L)
    #print(RHS_after_solving.get_local())
        
    
    
    #u_P1 = project(u, V)                        # since u is a function defined on V, we assume u_P1 is the exact representation of V
    #u_nodal_values = u_P1.vector()

    #print("nodal values of u")
    #for index_nodal in range(len(u_nodal_values)):
        #print("    ({:2.2e}, {:2.2e}) {:2.2e}".format(dofs_x[index_nodal][0], 1, u_nodal_values[index_nodal]))


    # Save solution in VTK format
    file = File("poisson.pvd")
    file << u
    
    # Compute errors
    print("Computing the error")
    

    #uexact = (1.0 - x**2) / 4.0
    uexact = eval(obj_string_u.func_value_u())                    # where is 'eval' from?
    
    
    M = (u - uexact)**2*dx(degree = 5)                            # adopted in undocumented/poisson-disc.py
    M0 = uexact**2*dx(degree = 5)
    l2_error_u = sqrt(assemble(M))
    l2_norm_u = sqrt(assemble(M0))
    
    grad_diff_u = grad(u-uexact)
    grad_u = grad(u)
    h1_semi_error_u = sqrt(assemble(inner(grad_diff_u, grad_diff_u)*dx))                    # the argument of assemble() is a function
    h1_seminorm_u = sqrt(assemble(inner(grad_u,grad_u)*dx))
    
    grad_grad_diff_u = grad(grad(u-uexact))
    grad_grad_u = grad(grad(u))
    h2_semi_error_u = sqrt(assemble(inner(grad_grad_diff_u, grad_grad_diff_u)*dx))
    h2_seminorm_u = sqrt(assemble(inner(grad_grad_u, grad_grad_u)*dx))
        
    print("    Using assemble()")
    print("        l2_error_u:", "{:2.2e}".format(l2_error_u))
    print("        h1_semi_error_u:", "{:2.2e}".format(h1_semi_error_u))
    print("        h2_semi_error_u:", "{:2.2e}".format(h2_semi_error_u))
    
    print()
    print("        l2_norm_u:", "{:2.2e}".format(l2_norm_u))
    print("        h1_seminorm_u:","{:2.2e}".format(h1_seminorm_u))     
    print("        h2_seminorm_u:","{:2.5e}".format(h2_seminorm_u))     
    
    
    
    #print("    Using norm()")
    #print("        norm(u):","{:2.2e}".format(norm(u)))
    #print("        norm(u,'H10'):","{:2.2e}".format(norm(u,'H10')))
    #print("        norm(u,'H1'):","{:2.2e}".format(norm(u,'H1')))
    
    
    
    array_error[id_refinement][0] = l2_error_u
    array_error[id_refinement][1] = h1_semi_error_u
    array_error[id_refinement][2] = h2_semi_error_u
    
    if id_refinement > 0 and id_refinement <= total_refinements:
        for k in range(3):
            array_order_of_convergence[id_refinement-1][k] = math.log(array_error[id_refinement-1][k]/array_error[id_refinement][k],10)/math.log(2,10)

    time_cpu_elapsed = time.time() - t
    
    array_time_cpu[id_refinement] = time_cpu_elapsed
    
    print("time elapsed:", "{:6.4f}".format(time_cpu_elapsed))
    
    file_error.write("%s %s %s %0.2e %0.2e %0.2e %0.2e %0.2e %0.2e %0.2e\n" %(current_refinement_level, n_vertices, n_dofs, l2_error_u, h1_semi_error_u, h2_semi_error_u, l2_norm_u, l2_norm_coeff_diff, l2_norm_coeff_helm, time_cpu_elapsed))
    
    current_refinement_level += 1
    id_refinement += 1
    n_rect_cell_one_direction = n_rect_cell_one_direction * 2

print()
print("====================================")
print("SUMMARY")
np.set_printoptions(formatter={'float': '{: 0.2e}'.format})

print("@error")
for line in array_error:
    print(line)

print("@order of convergence")
np.set_printoptions(formatter={'float': '{: 0.2f}'.format})
for line in array_order_of_convergence:
    print(line)
    
print("@cpu time")
np.set_printoptions(formatter={'float': '{: 0.2e}'.format})
for line in array_time_cpu:
    print(line)


print()
    
file_error.close() 


# Plot solution
#import matplotlib.pyplot as plt
#plot(u)
#plt.show()
