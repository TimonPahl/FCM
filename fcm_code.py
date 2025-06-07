import mlhp

D = 3

print("1. Setting up mesh and basis", flush=True)

# Setup triangulation domain
triangulation = mlhp.readStl("/Applications/Datein_Timon/Bildung/Uni_Rostock/Master_MaschBau/3_Semester/Studienarbeit/Code/FCM-StA/FCM/Würfel1.stl")
kdtree = mlhp.buildKdTree(triangulation)
domain = mlhp.implicitTriangulation(triangulation, kdtree)

# Setup discretization
youngsModulus = 1e11 # youngs modulus in N/m2
poissonsRatio = 0.3

polynomialDegree = 1
nelements = [100]*D # original value was 50
alphaFCM = 1e-5
penalty = 1e5 * youngsModulus

origin, max = triangulation.boundingBox()

# ##################### FIX ####################
lengths = [m - o for o, m in zip(origin, max)] 

origin = [o - 1e-10 for o in origin]
max = [m + 1e-10 for m in max]
#debugging
print("STL Bounding Box:")
print("  Origin :", origin)
print("  Max    :", max)
print("  Length :", lengths)
print("  Fix unten bei z <", origin[2] + lengths[2]*0.01)
print("  Drück oben bei z >", origin[2] + lengths[2]*0.99)



lengths = [m - o for o, m in zip(origin, max)]

grid = mlhp.makeRefinedGrid(nelements, lengths, origin)
basis = mlhp.makeHpTensorSpace(grid, polynomialDegree, nfields=D)

print(basis)

print("2. Allocating linear system", flush=True)

matrix = mlhp.allocateSparseMatrix(basis)
vector = mlhp.allocateRhsVector(matrix)

print("2. Computing weak boundary integrals", flush=True)

def createBoundaryQuadrature(func):
     filtered = mlhp.filterTriangulation(triangulation, mlhp.implicitFunction(D, func))
     intersected, celldata = mlhp.intersectTriangulationWithMesh(grid,filtered)
     quadrature = mlhp.triangulationQuadrature(intersected, celldata,polynomialDegree + 1)
     return intersected, celldata, quadrature

shrink = 0.1
intersected0, celldata0, quadrature0 = createBoundaryQuadrature(f"x <{origin[0] + shrink*lengths[0]}")
intersected1, celldata1, quadrature1 = createBoundaryQuadrature(f"x >{origin[0] + lengths[0] - shrink*lengths[0] }") #original value was 1e-2

integrand0 = mlhp.l2BoundaryIntegrand(mlhp.vectorField(D, [penalty] * D), mlhp.vectorField(D, [0.0, 0.0, 0.0]))
integrand1 = mlhp.l2BoundaryIntegrand(mlhp.vectorField(D, [penalty] * D), mlhp.vectorField(D, [penalty*10.0, 0.0, 0.0])) #defines a Dirichlet BC with displacement in x of 5
#integrand1 = mlhp.neumannIntegrand(mlhp.vectorField(D, [1e3, 0.0, 0.0]))


mlhp.integrateOnSurface(basis, integrand0, [matrix, vector], quadrature0)
mlhp.integrateOnSurface(basis, integrand1, [matrix, vector], quadrature1)
#mlhp.integrateOnSurface(basis, integrand1, [vector], quadrature1)

print("4. Computing domain integral", flush=True)

E = mlhp.scalarField(D, youngsModulus)
nu = mlhp.scalarField(D, poissonsRatio)
rhs = mlhp.vectorField(D, [0.0, 0.0, 0.0]) #Body forces applied on the body. 

kinematics = mlhp.smallStrainKinematics(D)
constitutive = mlhp.isotropicElasticMaterial(E, nu)
integrand = mlhp.staticDomainIntegrand(kinematics, constitutive, rhs)

quadrature = mlhp.momentFittingQuadrature(domain, depth=polynomialDegree, epsilon=alphaFCM)

mlhp.integrateOnDomain(basis, integrand, [matrix, vector],
quadrature=quadrature)

print("6. Solving linear system", flush=True)

#P = mlhp.additiveSchwarzPreconditioner(matrix, basis, dirichlet[0])
P = mlhp.diagonalPreconditioner(matrix)

dofs, norms = mlhp.cg(matrix, vector, M=P, maxiter=10000, residualNorms=True)

#print(f"cond K after domain integral:{numpy.linalg.cond(matrix.todense())}")
#import matplotlib.pyplot as plt
#plt.loglog(norms)
#plt.show()

print("7. Postprocessing solution", flush=True)

#Output solution on FCM mesh and boundary surface
gradient = mlhp.projectGradient(basis, dofs, quadrature)

processors = [mlhp.solutionProcessor(D, dofs, "Displacement"),
               mlhp.stressProcessor(gradient, kinematics, constitutive),
               mlhp.vonMisesProcessor(dofs, kinematics, constitutive,"VonMises1"),
#               mlhp.vonMisesProcessor(gradient, kinematics, constitutive, "VonMises2"),
#               mlhp.strainEnergyProcessor(gradient, kinematics, constitutive),
               mlhp.functionProcessor(domain)]

intersected, celldata = mlhp.intersectTriangulationWithMesh(grid, triangulation, kdtree)

surfmesh = mlhp.associatedTrianglesCellMesh(intersected, celldata)

outputDir="/Applications/Datein_Timon/Bildung/Uni_Rostock/Master_MaschBau/3_Semester/Studienarbeit/Code/FCM-StA/FCM/outputsWuerfel1"

writer0 = mlhp.PVtuOutput(filename=outputDir+"/linear_elasticity_fcm_stl_boundary")
writer1 = mlhp.PVtuOutput(filename=outputDir+"/linear_elasticity_fcm_stl_fcmmesh")

mlhp.writeBasisOutput(basis, surfmesh, writer0, processors)
mlhp.writeBasisOutput(basis, writer=writer1, processors=processors)

# Output boundary surfaces
surfmesh0 = mlhp.associatedTrianglesCellMesh(intersected0, celldata0)
surfmesh1 = mlhp.associatedTrianglesCellMesh(intersected1, celldata1)

surfwriter0 = mlhp.VtuOutput(filename=outputDir+"/linear_elasticity_fcm_stl_boundary0")
surfwriter1 = mlhp.VtuOutput(filename=outputDir+"/linear_elasticity_fcm_stl_boundary1")

mlhp.writeMeshOutput(grid, surfmesh0, surfwriter0, [])
mlhp.writeMeshOutput(grid, surfmesh1, surfwriter1, [])