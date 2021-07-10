
from dolfin import *


def errornorm_self_defined(u_e, u, norm_type):
    
    #print('    norm_type:', norm_type)
    
    V = u.function_space()
    mesh = V.mesh()
    degree = V.ufl_element().degree()
    
    #print("    degree:", degree)
    
    W = FunctionSpace(mesh, 'P', degree + 3)            #  + 3
                                                        # function spaces of different degree would produce different errors
                                                        # better to choose higher order, such as +1, +2, etc., because this is the same with using the eval() function
    
    #print("    degree of function space W:", W.ufl_element().degree())
    
    u_e_W = interpolate(u_e, W)
    u_W = interpolate(u, W)
    e_W = Function(W)
    e_W.vector()[:] = u_e_W.vector().get_local() - u_W.vector().get_local()
        
    #print('    size of e_W.vector():', end = '')
    #print(len(e_W.vector().get_local()))
    
    #print('    e_W.vector():')
    #print(e_W.vector().get_local())
    
    
    if norm_type == 'L2':
        error = e_W**2*dx
    elif norm_type == 'H10':
        
        #grad_e_W_1, grad_e_W_2 = grad(e_W).split()
        
        #print('    grad(e_W):')
        #print(grad_e_W_1.vector().get_local())
        
        error = inner(grad(e_W), grad(e_W))*dx
    elif norm_type == 'H20':
        
        error = inner(grad(grad(e_W)), grad(grad(e_W)))*dx
        
        #print("projecting grad(grad(u_e_W)) grad(grad(u_W)) grad(grad(e_W))")
        #V_for_tensor = TensorFunctionSpace(mesh, 'P', degree + 3)
        #grad_grad_u_e = project(grad(grad(u_e_W)), V_for_tensor)
        #grad_grad_u = project(grad(grad(u_W)), V_for_tensor)
        #grad_grad_e_W = project(grad(grad(e_W)), V_for_tensor)
        
        #for i in range(0, len(grad_grad_e_W.vector().get_local())):
            #print("{:2.2e} {:2.2e} {:2.2e}".format(grad_grad_u_e.vector().get_local()[i], grad_grad_u.vector().get_local()[i], grad_grad_e_W.vector().get_local()[i]))
        
                
    
    return sqrt(abs(assemble(error)))                       # we can give an argument to assemble(). For example, (degree = self.degree + 2)
                                                            # the value of this argument does not affect the error by now
