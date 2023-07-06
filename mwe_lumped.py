from firedrake import *
import finat
import FIAT

import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt

import math


def RickerWavelet(t, freq, amp=1.0):
    # Shift in time so the entire wavelet is injected
    t = t - (math.sqrt(6.0) / (math.pi * freq))
    return amp * (
        (1.0 - 2.0 * (math.pi * freq * t) ** 2) * np.exp(-(math.pi * freq * t) ** 2)
    )

def delta_expr(x0, x, y, sigma_x=2000.0):
    sigma_x = Constant(sigma_x)
    return exp(-sigma_x * ((x - x0[0]) ** 2 + (y - x0[1]) ** 2))

mesh = UnitSquareMesh(50, 50, quadrilateral=True)
element = FiniteElement("KMV", mesh.ufl_cell(), degree=2, variant="KMV")
V = FunctionSpace(mesh, element)

x, y = SpatialCoordinate(mesh)

u = TrialFunction(V)
v = TestFunction(V)

u_np1 = Function(V, name="u_np1")  # timestep n+1
u_n = Function(V, name="u_n")    # timestep n
u_nm1 = Function(V, name="u_nm1")  # timestep n-1

R = Function(V, name="R")

c = Constant(1)

T = 1.0
dt = 0.001
t = 0
step = 0

freq = 5
source = Constant([0.5, 0.5])
ricker = Constant(0.0)
ricker.assign(RickerWavelet(t, freq))

quad_rule = finat.quadrature.make_quadrature(V.finat_element.cell, V.ufl_element().degree(), "KMV")
dxlump=dx(scheme=quad_rule)

m = (u - 2.0 * u_n + u_nm1) / Constant(dt * dt) * v * dxlump
a = c*c*dot(grad(u_n), grad(v)) * dx
F = m + a - delta_expr(source, x, y)*ricker * v * dx
a, r = lhs(F), rhs(F)
A = assemble(a)
params={"mat_type": "matfree", "ksp_type": "preonly", "pc_type": "jacobi"}
solver = LinearSolver(A, solver_parameters=params)

step = 0
while t < T:
    step += 1

    # Update the RHS vector according to the current simulation time `t`
    ricker.assign(RickerWavelet(t, freq))
    R = assemble(r, tensor=R)
    # Call the solver object to do point-wise division to solve the system.
    solver.solve(u_np1, R)
    # Exchange the solution at the two time-stepping levels.
    u_nm1.assign(u_n)
    u_n.assign(u_np1)
    # Increment the time and write the solution to the file for visualization in ParaView.
    t += dt
    if step % 20 == 0:
        print("Elapsed time is: "+str(t))

