


from dolfin import *

import getopt, sys
import math
import time

import numpy as np


vec_dim = ["1d", "2d"]
id_dim = 1                                          # only for real-valued problems

gdim = id_dim + 1


class String_U:
  def __init__(self, id_solution):
    self.id_solution = id_solution
    
  def func_for_printing(self):
    print('{:>10} {:>2}'.format("u:", self.func_value_u()))
    print('{:>10} {:>2}'.format(r"u_x:", self.func_gradient_u_in_x_direction()))
    print('{:>10} {:>2}'.format(r"u_y:", self.func_gradient_u_in_y_direction()))
    print('{:>10} {:>2}'.format("\Delta u:", self.func_delta_u()))
    
  def func_value_u(self):
    switcher = {
        1: "pow(x[0]-0.5, 2.0)",                                               
        2: "exp(-pow(x[0]-0.5, 2.0))",
        61: "(pow(x[0]-0.5, 2.0) + pow(x[1]-0.5, 2.0) + (x[0]-0.5) * (x[1]-0.5))",
        62: "exp(-(pow(x[0]-0.5, 2.0) + pow(x[1]-0.5, 2.0)))"
    }
    return(switcher.get(self.id_solution, "x[0]"))
  def func_gradient_u_in_x_direction(self):
    switcher = {
        1: "2.0*(x[0]-0.5)",
        2: "exp(-pow(x[0]-0.5,2.0)) * (-2.0*(x[0]-0.5))",
        61: "(2.0*(x[0]-0.5) + (x[1]-0.5))",
        62: "exp(-(pow(x[0]-0.5, 2.0) + pow(x[1]-0.5, 2.0))) * (-2.0*(x[0]-0.5))"
    }
    return(switcher.get(self.id_solution, "x[0]"))
  def func_gradient_u_in_y_direction(self):
    switcher = {
        1: "0.0",
        2: "0.0",
        61: "(2.0*(x[1]-0.5) + (x[0]-0.5))",
        62: "exp(-(pow(x[0]-0.5, 2.0) + pow(x[1]-0.5, 2.0))) * (-2.0*(x[1]-0.5))"
    }
    return(switcher.get(self.id_solution, "x[1]"))
  def func_delta_u(self):
    switcher = {
        1: "2.0",
        2: "exp(-pow(x[0]-0.5,2.0)) * (pow(2*(x[0]-0.5),2.0) - 2.0)",
        61: "4.0",
        62: "exp(-(pow(x[0]-0.5, 2.0) + pow(x[1]-0.5, 2.0))) * (4.0 * pow(x[0]-0.5, 2.0) + 4.0 * pow(x[1]-0.5, 2.0) - 4.0)"
    }
    return(switcher.get(self.id_solution, "x[0]"))

        #2: "-exp(-pow(x[0]-0.5,2))*(pow(2*(x[0]-0.5),2)-2)*(1.0+x[0])-exp(-(pow(x[0]-0.5,2)))*(-2.0*(x[0]-0.5))"             
            # "exp(-(pow(x[0]-0.5, 2) + pow(x[1]-0.5, 2)))*(4.0*(pow(x[0]-0.5, 2) + pow(x[1]-0.5, 2) - 1.0))"


class String_Coeff_Diff:
  def __init__(self, id_coeff_diff, coeff_diff_inner):
    self.id_coeff_diff = id_coeff_diff
    self.coeff_diff_inner = coeff_diff_inner
    
  def func_for_printing(self):
    print('{:>10} {:>2}'.format("D:", self.func_value_coeff_diff()))
    print('{:>10} {:>2}'.format(r"D_x:", self.func_gradient_coeff_diff_in_x_direction()))
    print('{:>10} {:>2}'.format(r"D_y:", self.func_gradient_coeff_diff_in_y_direction()))
  
  def func_value_coeff_diff(self):
    switcher = {
        1: str(self.coeff_diff_inner),
        2: "(1.0+x[0])",                # +x[0]*x[0]
        3: "exp(-pow(x[0]-0.5,2.0))",
        4: "(0.5 + pow(cos(x[0]), 2.0))",
        5: "(0.01 + x[0]) * (1.01 - x[0])"
    }
    return(switcher.get(self.id_coeff_diff, "x[0]"))      
  def func_gradient_coeff_diff_in_x_direction(self):
    switcher = {
        1: "0.0",
        2: "(1.0)",                # +2.0*x[0]
        3: "exp(-pow(x[0]-0.5,2.0))*(-2.0*(x[0]-0.5))",
        4: "2.0*cos(x[0])*(-sin(x[0]))",
        5: "(1.0 - 2.0 * x[0])"
    }
    return(switcher.get(self.id_coeff_diff, "x[0]"))
  def func_gradient_coeff_diff_in_y_direction(self):
    switcher = {
        1: "0.0",
        2: "0.0",
        3: "0.0",
        4: "0.0",
        5: "0.0"
    }
    return(switcher.get(self.id_coeff_diff, "x[0]"))

class String_Coeff_Helm:
  def __init__(self, id_coeff_helm, coeff_helm_inner):
    self.id_coeff_helm = id_coeff_helm
    self.coeff_helm_inner = coeff_helm_inner
    
  def func_for_printing(self):
    print('{:>10} {:>2}'.format("r:", self.func_value_coeff_helm()))
  
  def func_value_coeff_helm(self):
    switcher = {
        1: str(self.coeff_helm_inner),
        2: "(1.0 + x[0])"
    }
    return(switcher.get(self.id_coeff_helm, "x[0]"))  


class String_Neumann_Boundary_Real:
  def __init__(self, id_solution, id_coeff_diff, coeff_diff_inner):
    self.id_solution = id_solution
    self.id_coeff_diff = id_coeff_diff
    self.coeff_diff_inner = coeff_diff_inner
  def func_for_printing(self):
    print('{:>33} {:>2}'.format("neumann_boundary_in_x_direction:", self.func_neumann_boundary_in_x_direction()))
    print('{:>33} {:>2}'.format("neumann_boundary_in_y_direction:", self.func_neumann_boundary_in_y_direction()))
        
  def func_neumann_boundary_in_x_direction(self):
    return(String_Coeff_Diff(self.id_coeff_diff, self.coeff_diff_inner).func_value_coeff_diff() + "*" + String_U(self.id_solution).func_gradient_u_in_x_direction())
  def func_neumann_boundary_in_y_direction(self):
    return(String_Coeff_Diff(self.id_coeff_diff, self.coeff_diff_inner).func_value_coeff_diff() + "*" + String_U(self.id_solution).func_gradient_u_in_y_direction())


class String_Neumann_Boundary_Complex:
  def __init__(self, id_solution_real, id_solution_imag, id_coeff_diff_real, coeff_diff_inner_real, id_coeff_diff_imag, coeff_diff_inner_imag):
    self.id_solution_real = id_solution_real
    self.id_solution_imag = id_solution_imag
    self.id_coeff_diff_real = id_coeff_diff_real
    self.coeff_diff_inner_real = coeff_diff_inner_real
    self.id_coeff_diff_imag = id_coeff_diff_imag
    self.coeff_diff_inner_imag = coeff_diff_inner_imag
  def func_for_printing(self):
    print('{:>33} {:>2}'.format("real part in x direction:", self.func_neumann_boundary_real_in_x_direction()))
    print('{:>33} {:>2}'.format("real part in y direction:", self.func_neumann_boundary_real_in_y_direction()))
    print('{:>33} {:>2}'.format("imag part in x direction:", self.func_neumann_boundary_imag_in_x_direction()))
    print('{:>33} {:>2}'.format("imag part in y direction:", self.func_neumann_boundary_imag_in_y_direction()))
        
  def func_neumann_boundary_real_in_x_direction(self):
    return( "(" + String_Coeff_Diff(self.id_coeff_diff_real, self.coeff_diff_inner_real).func_value_coeff_diff() + "*" + String_U(self.id_solution_real).func_gradient_u_in_x_direction() + " - " + String_Coeff_Diff(self.id_coeff_diff_imag, self.coeff_diff_inner_imag).func_value_coeff_diff() + "*" + String_U(self.id_solution_imag).func_gradient_u_in_x_direction() + ")")
  def func_neumann_boundary_real_in_y_direction(self):
    return( "(" + String_Coeff_Diff(self.id_coeff_diff_real, self.coeff_diff_inner_real).func_value_coeff_diff() + "*" + String_U(self.id_solution_real).func_gradient_u_in_y_direction() + " - " + String_Coeff_Diff(self.id_coeff_diff_imag, self.coeff_diff_inner_imag).func_value_coeff_diff() + "*" + String_U(self.id_solution_imag).func_gradient_u_in_y_direction() + ")")
  def func_neumann_boundary_imag_in_x_direction(self):
    return( "(" + String_Coeff_Diff(self.id_coeff_diff_real, self.coeff_diff_inner_real).func_value_coeff_diff() + "*" + String_U(self.id_solution_imag).func_gradient_u_in_x_direction() + " + " + String_Coeff_Diff(self.id_coeff_diff_imag, self.coeff_diff_inner_imag).func_value_coeff_diff() + "*" + String_U(self.id_solution_real).func_gradient_u_in_x_direction() + ")")
  def func_neumann_boundary_imag_in_y_direction(self):
    return( "(" + String_Coeff_Diff(self.id_coeff_diff_real, self.coeff_diff_inner_real).func_value_coeff_diff() + "*" + String_U(self.id_solution_imag).func_gradient_u_in_y_direction() + " + " + String_Coeff_Diff(self.id_coeff_diff_imag, self.coeff_diff_inner_imag).func_value_coeff_diff() + "*" + String_U(self.id_solution_real).func_gradient_u_in_y_direction() + ")")

class String_F(String_U, String_Coeff_Diff, String_Coeff_Helm):
  def __init__(self, id_solution, id_coeff_diff, coeff_diff_inner, id_coeff_helm, coeff_helm_inner):
    self.id_solution = id_solution
    self.id_coeff_diff = id_coeff_diff
    self.coeff_diff_inner = coeff_diff_inner
    self.id_coeff_helm = id_coeff_helm
    self.coeff_helm_inner = coeff_helm_inner
  def func_for_printing(self):
    print("f:", self.func_f())
  def func_f(self):
    return("-1.0 * (" + String_Coeff_Diff(self.id_coeff_diff, self.coeff_diff_inner).func_gradient_coeff_diff_in_x_direction() + "*" + String_U(self.id_solution).func_gradient_u_in_x_direction() + " + " + String_Coeff_Diff(self.id_coeff_diff, self.coeff_diff_inner).func_value_coeff_diff() + "*"+ String_U(self.id_solution).func_delta_u() + ") + " + String_Coeff_Helm(self.id_coeff_helm, self.coeff_helm_inner).func_value_coeff_helm() + " * "  + String_U(self.id_solution).func_value_u() )

    #return("-1.0*" + str(self.coeff_diff_inner) + "*" + String_U(self.id_solution).func_delta_u())                          # If only Poisson problems are considered

class String_F_Complex_1D(String_U, String_Coeff_Diff, String_Coeff_Helm):
  def __init__(self, id_solution_real, id_solution_imag, id_coeff_diff_real, coeff_diff_inner_real, id_coeff_diff_imag, coeff_diff_inner_imag, id_coeff_helm_real, coeff_helm_inner_real, id_coeff_helm_imag, coeff_helm_inner_imag):
    self.id_solution_real = id_solution_real
    self.id_solution_imag = id_solution_imag
    self.id_coeff_diff_real = id_coeff_diff_real
    self.coeff_diff_inner_real = coeff_diff_inner_real
    self.id_coeff_diff_imag = id_coeff_diff_imag
    self.coeff_diff_inner_imag = coeff_diff_inner_imag
    self.id_coeff_helm_real = id_coeff_helm_real
    self.coeff_helm_inner_real = coeff_helm_inner_real
    self.id_coeff_helm_imag = id_coeff_helm_imag
    self.coeff_helm_inner_imag = coeff_helm_inner_imag
  def func_for_printing(self):
    print("f_real:", self.func_f_real())
    print("f_imag:", self.func_f_imag())
  def func_f_real(self):
    return("-1.0 * (" + String_Coeff_Diff(self.id_coeff_diff_real, self.coeff_diff_inner_real).func_gradient_coeff_diff_in_x_direction() + "*" + String_U(self.id_solution_real).func_gradient_u_in_x_direction() + " + " + String_Coeff_Diff(self.id_coeff_diff_real, self.coeff_diff_inner_real).func_value_coeff_diff() + "*"+ String_U(self.id_solution_real).func_delta_u() + " - " + "( " + String_Coeff_Diff(self.id_coeff_diff_imag, self.coeff_diff_inner_imag).func_gradient_coeff_diff_in_x_direction() + "*" + String_U(self.id_solution_imag).func_gradient_u_in_x_direction() + " + " + String_Coeff_Diff(self.id_coeff_diff_imag, self.coeff_diff_inner_imag).func_value_coeff_diff() + "*"+ String_U(self.id_solution_imag).func_delta_u() + ")) + " + String_Coeff_Helm(self.id_coeff_helm_real, self.coeff_helm_inner_real).func_value_coeff_helm() + " * "  + String_U(self.id_solution_real).func_value_u() + "-" + String_Coeff_Helm(self.id_coeff_helm_imag, self.coeff_helm_inner_imag).func_value_coeff_helm() + " * "  + String_U(self.id_solution_imag).func_value_u())
  def func_f_imag(self):
    return("-1.0 * (" + String_Coeff_Diff(self.id_coeff_diff_real, self.coeff_diff_inner_real).func_gradient_coeff_diff_in_x_direction() + "*" + String_U(self.id_solution_imag).func_gradient_u_in_x_direction() + " + " + String_Coeff_Diff(self.id_coeff_diff_real, self.coeff_diff_inner_real).func_value_coeff_diff() + "*"+ String_U(self.id_solution_imag).func_delta_u() + " + " + "( " + String_Coeff_Diff(self.id_coeff_diff_imag, self.coeff_diff_inner_imag).func_gradient_coeff_diff_in_x_direction() + "*" + String_U(self.id_solution_real).func_gradient_u_in_x_direction() + " + " + String_Coeff_Diff(self.id_coeff_diff_imag, self.coeff_diff_inner_imag).func_value_coeff_diff() + "*"+ String_U(self.id_solution_real).func_delta_u() + ")) + " + String_Coeff_Helm(self.id_coeff_helm_real, self.coeff_helm_inner_real).func_value_coeff_helm() + " * "  + String_U(self.id_solution_real).func_value_u() + "+" + String_Coeff_Helm(self.id_coeff_helm_real, self.coeff_helm_inner_real).func_value_coeff_helm() + " * "  + String_U(self.id_solution_imag).func_value_u())
  

class String_F_Complex_2D(String_U, String_Coeff_Diff, String_Coeff_Helm):                      # Note: we only consider matrix D to be diagonal
  def __init__(self, id_solution_real, id_solution_imag, id_coeff_diff_real, coeff_diff_inner_real, id_coeff_diff_imag, coeff_diff_inner_imag, id_coeff_helm_real, coeff_helm_inner_real, id_coeff_helm_imag, coeff_helm_inner_imag):                            # D is allowed to change in the y direction
    self.id_solution_real = id_solution_real
    self.id_solution_imag = id_solution_imag
    self.id_coeff_diff_real = id_coeff_diff_real
    self.coeff_diff_inner_real = coeff_diff_inner_real
    self.id_coeff_diff_imag = id_coeff_diff_imag
    self.coeff_diff_inner_imag = coeff_diff_inner_imag
    self.id_coeff_helm_real = id_coeff_helm_real
    self.coeff_helm_inner_real = coeff_helm_inner_real
    self.id_coeff_helm_imag = id_coeff_helm_imag
    self.coeff_helm_inner_imag = coeff_helm_inner_imag
  def func_for_printing(self):
    print("f_real:", self.func_f_real())
    print("f_imag:", self.func_f_imag())
  def func_f_real(self): 
    return ("-1.0 * (" + String_Coeff_Diff(self.id_coeff_diff_real, self.coeff_diff_inner_real).func_gradient_coeff_diff_in_x_direction() + "*" + String_U(self.id_solution_real).func_gradient_u_in_x_direction() + " + " + String_Coeff_Diff(self.id_coeff_diff_real, self.coeff_diff_inner_real).func_gradient_coeff_diff_in_y_direction() + "*" + String_U(self.id_solution_real).func_gradient_u_in_y_direction() + " + " + String_Coeff_Diff(self.id_coeff_diff_real, self.coeff_diff_inner_real).func_value_coeff_diff() + "*"+ String_U(self.id_solution_real).func_delta_u() + " - " + "( " + String_Coeff_Diff(self.id_coeff_diff_imag, self.coeff_diff_inner_imag).func_gradient_coeff_diff_in_x_direction() + "*" + String_U(self.id_solution_imag).func_gradient_u_in_x_direction() + " + " + String_Coeff_Diff(self.id_coeff_diff_imag, self.coeff_diff_inner_imag).func_gradient_coeff_diff_in_y_direction() + "*" + String_U(self.id_solution_imag).func_gradient_u_in_y_direction() + " + " + String_Coeff_Diff(self.id_coeff_diff_imag, self.coeff_diff_inner_imag).func_value_coeff_diff() + "*"+ String_U(self.id_solution_imag).func_delta_u() + ")) + " + String_Coeff_Helm(self.id_coeff_helm_real, self.coeff_helm_inner_real).func_value_coeff_helm() + " * "  + String_U(self.id_solution_real).func_value_u() + "-" + String_Coeff_Helm(self.id_coeff_helm_imag, self.coeff_helm_inner_imag).func_value_coeff_helm() + " * "  + String_U(self.id_solution_imag).func_value_u())
  def func_f_imag(self):
    return("-1.0 * (" + String_Coeff_Diff(self.id_coeff_diff_real, self.coeff_diff_inner_real).func_gradient_coeff_diff_in_x_direction() + "*" + String_U(self.id_solution_imag).func_gradient_u_in_x_direction() + " + " + String_Coeff_Diff(self.id_coeff_diff_real, self.coeff_diff_inner_real).func_gradient_coeff_diff_in_y_direction() + "*" + String_U(self.id_solution_imag).func_gradient_u_in_y_direction() + " + " + String_Coeff_Diff(self.id_coeff_diff_real, self.coeff_diff_inner_real).func_value_coeff_diff() + "*"+ String_U(self.id_solution_imag).func_delta_u() + " + " + "( " + String_Coeff_Diff(self.id_coeff_diff_imag, self.coeff_diff_inner_imag).func_gradient_coeff_diff_in_x_direction() + "*" + String_U(self.id_solution_real).func_gradient_u_in_x_direction() + " + " + String_Coeff_Diff(self.id_coeff_diff_imag, self.coeff_diff_inner_imag).func_gradient_coeff_diff_in_y_direction() + "*" + String_U(self.id_solution_real).func_gradient_u_in_y_direction() + " + " + String_Coeff_Diff(self.id_coeff_diff_imag, self.coeff_diff_inner_imag).func_value_coeff_diff() + "*"+ String_U(self.id_solution_real).func_delta_u() + ")) + " + String_Coeff_Helm(self.id_coeff_helm_real, self.coeff_helm_inner_real).func_value_coeff_helm() + " * "  + String_U(self.id_solution_real).func_value_u() + "+" + String_Coeff_Helm(self.id_coeff_helm_real, self.coeff_helm_inner_real).func_value_coeff_helm() + " * "  + String_U(self.id_solution_imag).func_value_u())
  
    

n_cells = 99
n_vertices = 2

grid_size = 1.0

n_dofs = 1



l2_error_u = 1.0
h1_semi_error_u = 1.0
h2_semi_error_u = 1.0

l2_error_u_from_finer = 1.0
h1_semi_error_u_from_finer = 1.0
h2_semi_error_u_from_finer = 1.0


l2_norm_u = 1.0
h1_seminorm_u = 1.0
h2_seminorm_u = 1.0

l2_norm_coeff_diff = 1.0
l2_norm_coeff_helm = 0.0





class Vectors_for_storage:
    def __init__(self, n_total_refinements):
        self.n_total_refinements = n_total_refinements
        
        self.array_error = np.zeros((n_total_refinements, 3))
        self.array_error_from_finer = np.zeros((n_total_refinements, 3))
        self.array_order_of_convergence = np.zeros((n_total_refinements-1, 3))
        self.array_order_of_convergence_from_finer = np.zeros((n_total_refinements-1, 3))
        self.array_time_cpu = np.zeros((n_total_refinements, 1))
        


            











