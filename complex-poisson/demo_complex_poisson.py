# 
# .. _demo_complex_poisson_equation:
# 
# Poisson equation

# https://fenicsproject.discourse.group/t/complex-equation-for-scattering-problem/3837
# ================

import sys
sys.path.append('../')

from classes_shared.demo_classes_shared import *
from classes_shared.demo_functions_shared import *


full_cmd_arguments = sys.argv

argument_list = full_cmd_arguments[1:]

print('Number of arguments:', len(argument_list), 'arguments.')
print('Argument List:', str(argument_list))

if len(argument_list) != 16:
    raise Exception('Using <id_solution_real> <id_solution_imag> <segment> <id_coeff_diff_real> <id_coeff_diff_imag> <segment> <id_coeff_helm_real> <coeff_helm_inner_real> <id_coeff_helm_imag> <coeff_helm_inner_imag> <segment> <degree> <initial refinement level> <total refinements>')


id_solution_real = int(argument_list[0])
id_solution_imag = int(argument_list[1])

id_coeff_diff_real = int(argument_list[3])
coeff_diff_inner_real = float(argument_list[4])

id_coeff_diff_imag = int(argument_list[5])
coeff_diff_inner_imag = float(argument_list[6])

id_coeff_helm_real = int(argument_list[8])
coeff_helm_inner_real = float(argument_list[9])
id_coeff_helm_imag = int(argument_list[10])
coeff_helm_inner_imag = float(argument_list[11])

degree_custom = int(argument_list[13])
level_initial_refinement = int(argument_list[14])
n_total_refinements = int(argument_list[15])

obj_vectors_for_storage = Vectors_for_storage(n_total_refinements)



tol = 1E-14
def boundary_diri_location(x, on_boundary):
    return on_boundary and (abs(x[0]) < tol or abs(1.0 - x[0]) < tol)


obj_string_u_real = String_U(id_solution_real)
obj_string_u_imag = String_U(id_solution_imag)

print("Real part of the solution")
obj_string_u_real.func_for_printing()

print()
print("Imaginary part of the solution")
obj_string_u_imag.func_for_printing()


print("")
print("Real part of the diffusion coefficient")
obj_string_coeff_diff_real = String_Coeff_Diff(id_coeff_diff_real, coeff_diff_inner_real)
obj_string_coeff_diff_real.func_for_printing()  
print("")
print("Imaginary part of the diffusion coefficient")
obj_string_coeff_diff_imag = String_Coeff_Diff(id_coeff_diff_imag, coeff_diff_inner_imag)
obj_string_coeff_diff_imag.func_for_printing()

expression_coeff_diff_real = Expression(obj_string_coeff_diff_real.func_value_coeff_diff(), degree = degree_custom)
expression_coeff_diff_imag = Expression(obj_string_coeff_diff_imag.func_value_coeff_diff(), degree = degree_custom)


print("")
print("Neumann boundary condition")
obj_string_neumann_boundary = String_Neumann_Boundary_Complex(id_solution_real, id_solution_imag, id_coeff_diff_real, coeff_diff_inner_real, id_coeff_diff_imag, coeff_diff_inner_imag)
obj_string_neumann_boundary.func_for_printing()
print("")


expression_neumann_boundary_real = Expression((obj_string_neumann_boundary.func_neumann_boundary_real_in_x_direction(), obj_string_neumann_boundary.func_neumann_boundary_real_in_y_direction()), degree = degree_custom)

expression_neumann_boundary_imag = Expression((obj_string_neumann_boundary.func_neumann_boundary_imag_in_x_direction(), obj_string_neumann_boundary.func_neumann_boundary_imag_in_y_direction()), degree = degree_custom)


dirichlet_bc_real = Expression(obj_string_u_real.func_value_u(), degree = degree_custom)
dirichlet_bc_imag = Expression(obj_string_u_imag.func_value_u(), degree = degree_custom)


print("")
print("Real part of the Helmholtz coefficient")
obj_string_coeff_helm_real = String_Coeff_Helm(id_coeff_helm_real, coeff_helm_inner_real)
obj_string_coeff_helm_real.func_for_printing()  
print("")
print("Imaginary part of the Helmholtz coefficient")
obj_string_coeff_helm_imag = String_Coeff_Helm(id_coeff_helm_imag, coeff_helm_inner_imag)
obj_string_coeff_helm_imag.func_for_printing()

expression_coeff_helm_real = Expression(obj_string_coeff_helm_real.func_value_coeff_helm(), degree = degree_custom)
expression_coeff_helm_imag = Expression(obj_string_coeff_helm_imag.func_value_coeff_helm(), degree = degree_custom)



print("")
obj_string_f = String_F_Complex_1D(id_solution_real, id_solution_imag, id_coeff_diff_real, coeff_diff_inner_real, id_coeff_diff_imag, coeff_diff_inner_imag, id_coeff_helm_real, coeff_helm_inner_real, id_coeff_helm_imag, coeff_helm_inner_imag)
print("RHS")
obj_string_f.func_for_printing()


#obj_string_f_test = String_F_Complex_2D(id_solution_real, id_solution_imag, id_coeff_diff_real, coeff_diff_inner_real, id_coeff_diff_imag, coeff_diff_inner_imag, id_coeff_helm_real, coeff_helm_inner_real, id_coeff_helm_imag, coeff_helm_inner_imag)
#print("RHS test")
#obj_string_f_test.func_for_printing()

f_real = Expression(obj_string_f.func_f_real(), degree = degree_custom)
f_imag = Expression(obj_string_f.func_f_imag(), degree = degree_custom)



file_error = open('data_output_error_1_fenics_0_sm_1_complex.txt', 'a')
file_error.write("current_refinement_level n_vertices n_dofs error_l2 error_h1_semi error_h2_semi l2_norm_u l2_norm_coeff_diff l2_norm_coeff_helm cpu_time\n")

current_refinement_level = level_initial_refinement
id_refinement = 0
n_rect_cell_one_direction = pow(2, level_initial_refinement)


while id_refinement < n_total_refinements:
    
    t = time.time()
    
    print("")
    print("########################################")
    print("refinement level: ", current_refinement_level)                                # initial refinement level is denoted by i=0
    print("########################################")
    
    print("Generalizing the mesh")

    print("    n_rect_cell_one_direction:", n_rect_cell_one_direction)

    #mesh and function space
    mesh = RectangleMesh(Point(0, 0), Point(1, 1), n_rect_cell_one_direction, n_rect_cell_one_direction)
    
    n_cells = mesh.num_cells()
    n_vertices = mesh.num_vertices()
    grid_size = 1.0/(2.0*n_rect_cell_one_direction)
    

    print("    n_cells:", n_cells)
    print("    n_vertices:", n_vertices)
    print("    grid_size:", "{:2.2e}".format(grid_size))    


    P_p = FiniteElement("Lagrange", mesh.ufl_cell(), degree_custom)
    V = FunctionSpace(mesh, P_p*P_p)

    dirichlet_bcs = [DirichletBC(V.sub(0), dirichlet_bc_real, boundary_diri_location),
        DirichletBC(V.sub(1), dirichlet_bc_imag, boundary_diri_location)]


    n = FacetNormal(mesh)   

    neumann_bc_times_n_real = dot(expression_neumann_boundary_real, n)
    neumann_bc_times_n_imag = dot(expression_neumann_boundary_imag, n)


    (u_real, u_imag) = TrialFunction(V)
    (v_real, v_imag) = TestFunction(V)
    

    a = inner(expression_coeff_diff_real * grad(u_real), grad(v_real))           + inner(expression_coeff_helm_real * u_real, v_real)
    a -= inner(expression_coeff_diff_imag * grad(u_imag), grad(v_real))          + inner(expression_coeff_helm_imag * u_imag, v_real)
    
    a +=  inner(expression_coeff_diff_real * grad(u_imag), grad(v_imag))         + inner(expression_coeff_helm_real * u_imag, v_imag)
    a +=  inner(expression_coeff_diff_imag * grad(u_real), grad(v_imag))         + inner(expression_coeff_helm_imag * u_real, v_imag)

    a *= dx


    L = (inner(f_real, v_real) + inner(f_imag, v_imag)) * dx

    L += (neumann_bc_times_n_real * v_real + neumann_bc_times_n_imag * v_imag) * ds 

    u = Function(V)
    solve(a == L, u, dirichlet_bcs)

    u_real, u_imag = u.split()

    file = File("complex_poisson_u_real.pvd")
    file << u_real

    file = File("complex_poisson_u_imag.pvd")
    file << u_imag


    u_real_exact = Expression(obj_string_u_real.func_value_u(), degree = 3)
    u_imag_exact = Expression(obj_string_u_imag.func_value_u(), degree = 3)


    l2_error_u_real = errornorm_self_defined(u_real_exact, u_real, 'L2')
    l2_error_u_imag = errornorm_self_defined(u_imag_exact, u_imag, 'L2')
    l2_error_u = sqrt(pow(l2_error_u_real, 2.0) + pow(l2_error_u_imag, 2.0))

    h1_semi_error_u_real = errornorm_self_defined(u_real_exact, u_real, 'H10')
    h1_semi_error_u_imag = errornorm_self_defined(u_imag_exact, u_imag, 'H10')
    h1_semi_error_u = sqrt(pow(h1_semi_error_u_real, 2.0) + pow(h1_semi_error_u_imag, 2.0))
    
    h2_semi_error_u_real = errornorm_self_defined(u_real_exact, u_real, 'H20')
    h2_semi_error_u_imag = errornorm_self_defined(u_imag_exact, u_imag, 'H20')
    h2_semi_error_u = sqrt(pow(h2_semi_error_u_real, 2.0) + pow(h2_semi_error_u_imag, 2.0))


    l2_norm_u_real = sqrt(assemble(u_real**2*dx))
    l2_norm_u_imag = sqrt(assemble(u_imag**2*dx))
    l2_norm_u = sqrt(pow(l2_norm_u_real, 2.0) + pow(l2_norm_u_imag, 2.0))
    

    print()
    print("{:>30}".format("l2_error_u: "), "{:2.2e}".format(l2_error_u), " (real ", "{:2.2e}".format(l2_error_u_real), ", imag ", "{:2.2e}".format(l2_error_u_imag), ")",sep="")
    print("{:>30}".format("h1_semi_error_u: "), "{:2.2e}".format(h1_semi_error_u), " (real ", "{:2.2e}".format(h1_semi_error_u_real), ", imag ", "{:2.2e}".format(h1_semi_error_u_imag), ")",sep="")    
    print("{:>30}".format("h2_semi_error_u: "), "{:2.2e}".format(h2_semi_error_u), " (real ", "{:2.2e}".format(h2_semi_error_u_real), ", imag ", "{:2.2e}".format(h2_semi_error_u_imag), ")",sep="")      
    
    
    print()
    print("{:>30}".format("l2_norm_u: "), "{:2.2e}".format(l2_norm_u), " (real ", "{:2.2e}".format(l2_norm_u_real), ", imag ", "{:2.2e}".format(l2_norm_u_imag), ")",sep="")


    print()
    print("*************************************************")
    
    print("Generalizing the mesh")

    n_rect_cell_one_direction_finer = n_rect_cell_one_direction * 2
    
    print("    n_rect_cell_one_direction_finer:", n_rect_cell_one_direction_finer)

    mesh_finer = RectangleMesh(Point(0, 0), Point(1, 1), n_rect_cell_one_direction_finer, n_rect_cell_one_direction_finer)
    
    n_cells_finer = mesh_finer.num_cells()
    n_vertices_finer = mesh_finer.num_vertices()
    grid_size_finer = 1.0/(2.0*n_rect_cell_one_direction_finer)
    
    print("    n_cells:", n_cells_finer)
    print("    n_vertices:", n_vertices_finer)
    print("    grid_size:", grid_size_finer)
    
    

    P_p_finer = FiniteElement("Lagrange", mesh_finer.ufl_cell(), degree_custom)
    V_finer = FunctionSpace(mesh_finer, P_p_finer*P_p_finer)

    bc_diri_finer = [DirichletBC(V_finer.sub(0), dirichlet_bc_real, boundary_diri_location),
        DirichletBC(V_finer.sub(1), dirichlet_bc_imag, boundary_diri_location)]


    n_finer = FacetNormal(mesh_finer)   

    neumann_bc_times_n_finer_real = dot(expression_neumann_boundary_real, n_finer)
    neumann_bc_times_n_finer_imag = dot(expression_neumann_boundary_imag, n_finer)


    (u_finer_real, u_finer_imag) = TrialFunction(V_finer)
    (v_finer_real, v_finer_imag) = TestFunction(V_finer)
    

    a_finer = inner(expression_coeff_diff_real * grad(u_finer_real), grad(v_finer_real)) + inner(expression_coeff_helm_real * u_finer_real, v_finer_real)
    a_finer -= inner(expression_coeff_diff_imag * grad(u_finer_imag), grad(v_finer_real)) + inner(expression_coeff_helm_imag * u_finer_imag, v_finer_real)
    
    a_finer +=  inner(expression_coeff_diff_real * grad(u_finer_imag), grad(v_finer_imag)) + inner(expression_coeff_helm_real * u_finer_imag, v_finer_imag)
    a_finer +=  inner(expression_coeff_diff_imag * grad(u_finer_real), grad(v_finer_imag)) + inner(expression_coeff_helm_imag * u_finer_real, v_finer_imag)

    a_finer *= dx


    L_finer = (inner(f_real, v_finer_real) + inner(f_imag, v_finer_imag)) * dx

    L_finer += (neumann_bc_times_n_finer_real * v_finer_real + neumann_bc_times_n_finer_imag * v_finer_imag) * ds     
    
    u_finer = Function(V_finer)
    solve(a_finer == L_finer, u_finer, bc_diri_finer)
    
    

    u_finer_real, u_finer_imag = u_finer.split()

    file = File("complex_poisson_u_finer_real.pvd")
    file << u_finer_real

    file = File("complex_poisson_u_finer_imag.pvd")
    file << u_finer_imag
    
    
    l2_error_u_finer_real = errornorm_self_defined(u_finer_real, u_real, 'L2')
    l2_error_u_finer_imag = errornorm_self_defined(u_finer_imag, u_imag, 'L2')
    l2_error_u_finer = sqrt(pow(l2_error_u_finer_real, 2.0) + pow(l2_error_u_finer_imag, 2.0))
    
    h1_semi_error_u_finer_real = errornorm_self_defined(u_finer_real, u_real, 'H10')
    h1_semi_error_u_finer_imag = errornorm_self_defined(u_finer_imag, u_imag, 'H10')
    h1_semi_error_u_finer = sqrt(pow(h1_semi_error_u_finer_real, 2.0) + pow(h1_semi_error_u_finer_imag, 2.0))
    
    h2_semi_error_u_finer_real = errornorm_self_defined(u_finer_real, u_real, 'H20')
    h2_semi_error_u_finer_imag = errornorm_self_defined(u_finer_imag, u_imag, 'H20')
    h2_semi_error_u_finer = sqrt(pow(h2_semi_error_u_finer_real, 2.0) + pow(h2_semi_error_u_finer_imag, 2.0))
    
    print()
    print("{:>30}".format("l2_error_u_finer: "), "{:2.2e}".format(l2_error_u_finer), " (real ", "{:2.2e}".format(l2_error_u_finer_real), ", imag ", "{:2.2e}".format(l2_error_u_finer_imag), ")",sep="")    
    print("{:>30}".format("h1_semi_error_u_finer: "), "{:2.2e}".format(h1_semi_error_u_finer), " (real ", "{:2.2e}".format(h1_semi_error_u_finer_real), ", imag ", "{:2.2e}".format(h1_semi_error_u_finer_imag), ")",sep="")    
    print("{:>30}".format("h2_semi_error_u_finer: "), "{:2.2e}".format(h2_semi_error_u_finer), " (real ", "{:2.2e}".format(h2_semi_error_u_finer_real), ", imag ", "{:2.2e}".format(h2_semi_error_u_finer_imag), ")",sep="")    
    
    print()
    print("*************************************************")
    
        
    

    print()
    time_cpu_elapsed = time.time() - t
    
    
    
    obj_vectors_for_storage.array_error[id_refinement][0] = l2_error_u
    obj_vectors_for_storage.array_error[id_refinement][1] = h1_semi_error_u
    obj_vectors_for_storage.array_error[id_refinement][2] = h2_semi_error_u
    
    if id_refinement > 0 and id_refinement <= n_total_refinements:
        for k in range(3):
            obj_vectors_for_storage.array_order_of_convergence[id_refinement-1][k] = math.log(obj_vectors_for_storage.array_error[id_refinement-1][k]/obj_vectors_for_storage.array_error[id_refinement][k],10)/math.log(2,10)

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
    
print("@cpu time")
np.set_printoptions(formatter={'float': '{: 0.2e}'.format})
for line in obj_vectors_for_storage.array_time_cpu:
    print(line)


print()
    
file_error.close()     












