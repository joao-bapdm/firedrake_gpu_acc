# Matrix free tests with Firedrake and GPU acceleration

This repo contains simple tests and instruction for recreating an environment for testing matrix free simulation in [Firedrake](https://github.com/firedrakeproject/firedrake) using GPUs.

## Environment

Install with the script in this repo, not the one from Firedrake master branch.

```python firedrake-install --minimal-petsc --tinyasm --slepc --package-branch tsfc gpu --package-branch pyop2 gpu```

A few packages in firedrake/src/ should be pointing to specific branches

    firedrake JDBetteridge/gpu2
    PyOP2     JDBetteridge/gpu2
    tsfc      JDBetteridge/gpu2
    loopy     main@3988272b

## Test scripts

The most basic script that should work is `test_offloading.py` in the tests directory.

There are two wave propagation test scripts, both matrix free adaptations from
the mass lumped example in Firedrake advanced tutorials. One uses the same KMV elements, the other, spectral elements, as shown below:

```python
from firedrake import *
import finat
import FIAT

import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt

import math


def gauss_lobatto_legendre_line_rule(degree):
    fiat_make_rule = FIAT.quadrature.GaussLobattoLegendreQuadratureLineRule
    fiat_rule = fiat_make_rule(FIAT.ufc_simplex(1), degree + 1)
    finat_ps = finat.point_set.GaussLobattoLegendrePointSet
    finat_qr = finat.quadrature.QuadratureRule
    return finat_qr(finat_ps(fiat_rule.get_points()), fiat_rule.get_weights())

def gauss_lobatto_legendre_cube_rule(dimension, degree):
    make_tensor_rule = finat.quadrature.TensorProductQuadratureRule
    result = gauss_lobatto_legendre_line_rule(degree)
    for _ in range(1, dimension):
        line_rule = gauss_lobatto_legendre_line_rule(degree)
        result = make_tensor_rule([result, line_rule])
    return result

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
element = FiniteElement("CG", mesh.ufl_cell(), degree=2, variant="spectral")
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

qr_x = gauss_lobatto_legendre_cube_rule(dimension=V.mesh().geometric_dimension(), degree=V.ufl_element().degree())
dxlump=dx(scheme=qr_x)

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
```
