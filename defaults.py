"""Contains defaults..."""

initial_data = lambda x: 0
exact_solution = lambda x, t: 0

# mesh
T = 1
elements_number = 8
courant_factor = (
    10  # factor for calculation number of time steps depending on number of simplices
)
a = 0
b = 1

# finite element space
polynomial_degree = 1

# solver
stabilization_parameter = 0.1
stabilization_gradient_approximation = "nodal"

# eoc
refine_number = 4
