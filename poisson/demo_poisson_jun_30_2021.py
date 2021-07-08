# 
# .. _demo_poisson_equation:
# 
# Poisson equation
# ================

import sys
sys.path.append('../')


from classes_shared.demo_classes_shared import *
from classes_shared.demo_functions_shared import *

full_cmd_arguments = sys.argv

argument_list = full_cmd_arguments[1:]

print('Number of arguments:', len(argument_list), 'arguments.')
print('Argument List:', str(argument_list))


if len(argument_list) != 7:
    raise Exception('Using <id_solution> <segment> <id_coeff_diff> <segment> <degree> <initial refinement level> <total refinements>')


id_solution = int(argument_list[0])
id_coeff_diff = int(argument_list[2])
degree_custom = int(argument_list[4])
level_initial_refinement = int(argument_list[5])
n_total_refinements = int(argument_list[6])

  
print("")
print('{:>25} {:>2}'.format("dimension:", vec_dim[id_dim]))
print('{:>25} {:>2}'.format("id_solution:", id_solution))
print('{:>25} {:>2}'.format("id_coeff_diff:", id_coeff_diff))
print('{:>25} {:>2}'.format("element degree:", degree_custom))
print('{:>25} {:>2}'.format("Initial refinement level:", level_initial_refinement))
print('{:>25} {:>2}'.format("n_total_refinements:", n_total_refinements))
  
print("")
obj_string_u = String_U(id_solution)
obj_string_u.func_for_printing()


print("")
obj_string_coeff_diff = String_Coeff_Diff(id_coeff_diff)
obj_string_coeff_diff.func_for_printing()  
print("")
obj_string_neumann_boundary = String_Neumann_Boundary(id_solution, id_coeff_diff)
obj_string_neumann_boundary.func_for_printing()
print("")
obj_string_f = String_F(id_solution, id_coeff_diff)
obj_string_f.func_for_printing()
 

print("")

file_error = open('data_output_error_1_fenics_0_sm_0_real.txt', 'a')
file_error.write("current_refinement_level n_vertices n_dofs error_l2 error_h1_semi error_h2_semi l2_norm_u l2_norm_coeff_diff l2_norm_coeff_helm cpu_time\n")

current_refinement_level = level_initial_refinement
id_refinement = 0
n_rect_cell_one_direction = pow(2,level_initial_refinement)

obj_vectors_for_storage = Vectors_for_storage(n_total_refinements)

while id_refinement < n_total_refinements:
    
    t = time.time()
    
    print("")
    print("########################################")
    print("refinement level: ", current_refinement_level)                                # initial refinement level is denoted by i=0
    print("########################################")
    
    print("Generalizing the mesh")
    
    # Create mesh and define function space
    
    print("    n_rect_cell_one_direction:", n_rect_cell_one_direction)
    
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
    
    print("    n_cells:", n_cells)
    print("    n_vertices:", n_vertices)
    #print("# of edges:", mesh.num_edges())
    #print("coordinates of the mesh\n",mesh.coordinates())

    print("    grid_size:", grid_size)
    
    
    print("Initializing the function space")
    
    V = FunctionSpace(mesh, "Lagrange", degree_custom)
    
    
    dofmap = V.dofmap()
    dofs = dofmap.dofs()
    n_dofs = len(dofs)
    
    print("    n_dofs:", n_dofs)
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
                                                                        # 'dot' also works here
    
    L = f*v*dx + g*v*ds 
    
    #print("LHS before applying BC:")
    #LHS= assemble(a)
    #print(LHS.array())
    
    #print("RHS before applying Neumann BC:")
    #RHS= assemble(L)
    #print(RHS.get_local())
    

    #print("RHS after applying Neumann BC (no matter if it exists):")
    #RHS_after_neum_bc= assemble(L)
    #print(RHS_after_neum_bc.get_local())       
    
    print("Solving the system")
    
    u = Function(V)
    solve(a == L, u, bc_diri)                           # a == L is the equation for the variational form
                                                        # , solver_parameters={'linear_solver' : 'mumps'}
                                                        
    u_nodal_values = u.vector().get_local()

    print("    nodal values of u")
    for index_nodal in range(len(u_nodal_values)):
        print("    ({:2.2e}) {:2.2e}".format(dofs_x[index_nodal][0], u_nodal_values[index_nodal]))               # , {:2.2e}   , 0

    # Save solution in VTK format
    file = File("poisson.pvd")
    file << u
    
    # Compute errors
    print("Computing the error using the exact solution")

    print("    @the reference solution is obtained by using eval() on an expression")
    
    
    x = SpatialCoordinate(mesh)                                                  # important for eval()
        
        
    u_exact_in_the_form_of_Expression = Expression(obj_string_u.func_value_u(), degree = degree_custom + 3)
    #u_exact = exp(-pow(x[0]-0.5, 2.0))
    u_exact = eval(obj_string_u.func_value_u())
    #u_exact = interpolate(u_exact_in_the_form_of_Expression, V)                 # u_exact is a function in this case
                                                                                 # when u_exact is only built by Expression, only the error of u can be computed
                                                                                 # the error is 'Cannot determine geometric dimension from expression'
    
    difference_u = u - u_exact
    l2_error_u = sqrt(assemble((difference_u)**2*dx))                            # adopted in undocumented/poisson-disc.py
                                                                                 # degree affects the error
    
    l2_norm_u = sqrt(assemble(u_exact**2*dx))
    
    grad_difference_u = grad(difference_u)
    grad_u = grad(u)
    h1_semi_error_u = sqrt(assemble(inner(grad_difference_u, grad_difference_u)*dx))                    # the argument of assemble() is a function
    h1_seminorm_u = sqrt(assemble(inner(grad_u, grad_u)*dx))
    
    grad_grad_difference_u = grad(grad_difference_u)
    grad_grad_u = grad(grad_u)
    h2_semi_error_u = sqrt(assemble(inner(grad_grad_difference_u, grad_grad_difference_u)*dx))
    h2_seminorm_u = sqrt(assemble(inner(grad_grad_u, grad_grad_u)*dx))          
    
        
    print("    l2_error_u:", "{:2.2e}".format(l2_error_u))                      # print("    Using assemble()")
    print("    h1_semi_error_u:", "{:2.2e}".format(h1_semi_error_u))
    print("    h2_semi_error_u:", "{:2.2e}".format(h2_semi_error_u))
    
    print()
    print("    l2_norm_u:", "{:2.2e}".format(l2_norm_u))
    print("    h1_seminorm_u:","{:2.2e}".format(h1_seminorm_u))     
    print("    h2_seminorm_u:","{:2.2e}".format(h2_seminorm_u))     
    
    
    #print("    norm(u):","{:2.2e}".format(norm(u)))                            # print("    Using norm()")
    #print("    norm(u,'H10'):","{:2.2e}".format(norm(u,'H10')))
    #print("    norm(u,'H1'):","{:2.2e}".format(norm(u,'H1')))
    

    
    print("    #using errornorm")
    l2_error_u_using_errornorm = errornorm_self_defined(u_exact_in_the_form_of_Expression, u, 'L2')
    h1_semi_error_u_using_errornorm_self_defined = errornorm_self_defined(u_exact_in_the_form_of_Expression, u, 'H10')
    h2_semi_error_u_using_errornorm_self_defined = errornorm_self_defined(u_exact_in_the_form_of_Expression, u, 'H20')
    
    print("    l2_error_u_using_errornorm:", "{:2.2e}".format(l2_error_u_using_errornorm))
    print("    h1_semi_error_u_using_errornorm_self_defined:", "{:2.2e}".format(h1_semi_error_u_using_errornorm_self_defined))
    print("    h2_semi_error_u_using_errornorm_self_defined:", "{:2.2e}".format(h2_semi_error_u_using_errornorm_self_defined))

    
    
    #print()
    #print("*************************************************")
    
    #print("Generalizing the mesh")
    
    #n_rect_cell_one_direction_finer = n_rect_cell_one_direction * 2
    
    #print("    n_rect_cell_one_direction:", n_rect_cell_one_direction_finer)
    
    #if id_dim == 0:
        #mesh_finer = IntervalMesh(n_rect_cell_one_direction_finer, 0, 1)
    #elif id_dim == 1:
        #mesh_finer = UnitSquareMesh(n_rect_cell_one_direction_finer, n_rect_cell_one_direction_finer, "crossed")

    #n_cells_finer = mesh_finer.num_cells()
    #n_vertices_finer = mesh_finer.num_vertices()
    
    #grid_size_finer = 1.0/(2.0*n_rect_cell_one_direction_finer)
    
    #print("    n_cells:", n_cells_finer)
    #print("    n_vertices:", n_vertices_finer)
    #print("    grid_size:", grid_size_finer)
    
    
    #print("Initializing the function space")
    
    #V_finer = FunctionSpace(mesh_finer, "Lagrange", degree_custom)
    
    
    #dofmap_finer = V_finer.dofmap()
    #dofs_finer = dofmap_finer.dofs()
    #n_dofs_finer = len(dofs_finer)
    
    #print("    n_dofs:", n_dofs_finer)
    
    #x_finer = SpatialCoordinate(mesh_finer)
    
    #u0_finer = Expression(obj_string_u.func_value_u(), degree = degree_custom)
                                                                            
    #bc_diri_finer = DirichletBC(V_finer, u0_finer, boundary_diri_location)
    
    #u_finer = TrialFunction(V_finer)
    #v_finer = TestFunction(V_finer)
    
    #n_finer = FacetNormal(mesh_finer)   
    
    #if id_dim == 0:
        #g = expression_neumann_boundary * n_finer[0]
    #elif id_dim == 1:
        #g = dot(expression_neumann_boundary, n_finer)
    
    #a_finer = inner(grad(u_finer), grad(v_finer))*dx
    
    #L_finer = f*v_finer*dx + g *v_finer*ds 
    
    #print("Solving the system")
    
    #u_finer = Function(V_finer)
    #solve(a_finer == L_finer, u_finer, bc_diri_finer)
    
    #file = File("poisson_finer.pvd")
    #file << u_finer
    
    #u_projected_from_finer = project(u_finer, V)
    
    #print("Computing the error using the solution on a finer grid")
    
    #difference_u_based_on_finer = u - u_projected_from_finer
    #l2_error_u_from_finer = sqrt(assemble(difference_u_based_on_finer**2*dx(degree = degree_custom)))
    
    #grad_difference_u_based_on_finer = grad(difference_u_based_on_finer)
    #h1_semi_error_u_from_finer = sqrt(assemble(inner(grad_difference_u_based_on_finer, grad_difference_u_based_on_finer)*dx))

    #grad_grad_difference_u_based_on_finer = grad(grad_difference_u_based_on_finer)
    #h2_semi_error_u_from_finer = sqrt(assemble(inner(grad_grad_difference_u_based_on_finer, grad_grad_difference_u_based_on_finer)*dx))        
        
    
    #print("    l2_error_u_from_finer:", "{:2.2e}".format(l2_error_u_from_finer))
    #print("    h1_semi_error_u_from_finer:", "{:2.2e}".format(h1_semi_error_u_from_finer))
    #print("    h2_semi_error_u_from_finer:", "{:2.2e}".format(h2_semi_error_u_from_finer))
    
    #print("*************************************************")
    
    print()
    time_cpu_elapsed = time.time() - t
    
    
    
    obj_vectors_for_storage.array_error[id_refinement][0] = l2_error_u
    obj_vectors_for_storage.array_error[id_refinement][1] = h1_semi_error_u
    obj_vectors_for_storage.array_error[id_refinement][2] = h2_semi_error_u
    
    obj_vectors_for_storage.array_error_from_finer[id_refinement][0] = l2_error_u_from_finer
    obj_vectors_for_storage.array_error_from_finer[id_refinement][1] = h1_semi_error_u_from_finer
    obj_vectors_for_storage.array_error_from_finer[id_refinement][2] = h2_semi_error_u_from_finer
    
    if id_refinement > 0 and id_refinement <= n_total_refinements:
        for k in range(3):
            obj_vectors_for_storage.array_order_of_convergence[id_refinement-1][k] = math.log(obj_vectors_for_storage.array_error[id_refinement-1][k]/obj_vectors_for_storage.array_error[id_refinement][k],10)/math.log(2,10)
                      
            #obj_vectors_for_storage.array_order_of_convergence_from_finer[id_refinement-1][k] = math.log(obj_vectors_for_storage.array_error_from_finer[id_refinement-1][k]/obj_vectors_for_storage.array_error_from_finer[id_refinement][k],10)/math.log(2,10)
    
    obj_vectors_for_storage.array_time_cpu[id_refinement] = time_cpu_elapsed
    
    print("time elapsed:", "{:6.4f}".format(time_cpu_elapsed))
    
    file_error.write("%s %s %s %0.2e %0.2e %0.2e %0.2e %0.2e %0.2e %0.2e\n" %(current_refinement_level, n_vertices, n_dofs, l2_error_u, h1_semi_error_u, h2_semi_error_u, l2_norm_u, l2_norm_coeff_diff, l2_norm_coeff_helm, time_cpu_elapsed))
    
    current_refinement_level += 1
    id_refinement += 1
    n_rect_cell_one_direction = n_rect_cell_one_direction * 2

print()
print("====================================")
print("SUMMARY")

print("@error")
np.set_printoptions(formatter={'float': '{: 0.2e}'.format})
for line in obj_vectors_for_storage.array_error:
    print(line)
    
print("@order of convergence")
np.set_printoptions(formatter={'float': '{: 0.2f}'.format})
for line in obj_vectors_for_storage.array_order_of_convergence:
    print(line)
    
#print("@error from finer")
#np.set_printoptions(formatter={'float': '{: 0.2e}'.format})    
#for line in obj_vectors_for_storage.array_error_from_finer:
    #print(line)
    
#print("@order of convergence from finer")
#np.set_printoptions(formatter={'float': '{: 0.2f}'.format})
#for line in obj_vectors_for_storage.array_order_of_convergence_from_finer:
    #print(line)

print("@cpu time")
np.set_printoptions(formatter={'float': '{: 0.2e}'.format})
for line in obj_vectors_for_storage.array_time_cpu:
    print(line)


print()
    
file_error.close() 


# Plot solution
#import matplotlib.pyplot as plt
#plot(u)
#plt.show()
