from fenics import *
import numpy as np

# 1) Mesh and function space
mesh = UnitSquareMesh(20, 20)
V = VectorFunctionSpace(mesh, "Lagrange", 1)   # displacement

# 2) Material parameters
E  = Constant(1e5)
nu = Constant(0.3)
mu    = E / (2*(1 + nu))
lmbda = E*nu / ((1 + nu)*(1 - 2*nu))

# 3) Variational problem
u = TrialFunction(V)
v = TestFunction(V)
def sigma(w):
    return lmbda*tr(sym(grad(w)))*Identity(2) + 2.0*mu*sym(grad(w))
a = inner(sigma(u), sym(grad(v)))*dx

# 4) BC: fix left edge
bc = DirichletBC(V, Constant((0.0, 0.0)), "near(x[0], 0.0)")

# 5) Loading: traction on right (x=1)
traction = Constant((1e2, 0.0))
boundary_markers = MeshFunction("size_t", mesh, mesh.topology().dim()-1, 0)
class Right(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], 1.0) and on_boundary
Right().mark(boundary_markers, 1)
ds = Measure("ds", domain=mesh, subdomain_data=boundary_markers)
L = dot(traction, v)*ds(1)

# 6) Solve
u_sol = Function(V)
solve(a == L, u_sol, bc)

# 7) Compute stress and project
W = TensorFunctionSpace(mesh, "DG", 0)        # piecewise-constant tensor space
S_expr = sigma(u_sol)
S_proj = project(S_expr, W)

# 8) Extract nodal data
coords = mesh.coordinates()
u_vals = u_sol.compute_vertex_values(mesh).reshape((2, -1)).T

# For each vertex, get the projected stress at that point:
s_vals = np.array([S_proj(Point(x, y)) for x, y in coords])  # shape (N,2,2)

# 9) BC mask
bc_mask = np.isclose(coords[:,0], 0.0)

# 10) Save
np.savez("sample0.npz",
         coords=coords,
         disp=u_vals,
         stress=s_vals,
         bc_mask=bc_mask)
print("Wrote sample0.npz")
