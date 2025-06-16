import mlhp
import datetime
import os
import numpy as np
import matplotlib.pyplot as plt

##
time1=datetime.datetime.now()
print("Start time:", time1)
print("1. Setting up mesh and basis", flush=True)
D = 3

# Setup triangulation domain
triangulation = mlhp.readStl("/Applications/Datein_Timon/Bildung/Uni_Rostock/Master_MaschBau/3_Semester/Studienarbeit/Code/FCM-StA/FCM/Wuerfel1.stl")
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

grid = mlhp.makeGrid(nelements, lengths, origin) #grobes gitter
grid = mlhp.makeRefinedGrid(mlhp.makeFilteredGrid(grid, domain=domain, nseedpoints=polynomialDegree + 2))#filterung mit stl-domain

basis = mlhp.makeHpTensorSpace(grid, polynomialDegree, nfields=D)

print(basis)
time2=datetime.datetime.now()
print("Time needed: ", time2-time1)

##
time1=time2
print("2. Allocating linear system", flush=True)

matrix = mlhp.allocateSparseMatrix(basis)
vector = mlhp.allocateRhsVector(matrix)

time2=datetime.datetime.now()
print("Time needed: ", time2-time1)
##
time1=time2
print("3. Computing weak boundary integrals", flush=True)

def createBoundaryQuadrature(func):
     filtered = mlhp.filterTriangulation(triangulation, mlhp.implicitFunction(D, func))
     intersected, celldata = mlhp.intersectTriangulationWithMesh(grid,filtered)
     quadrature = mlhp.triangulationQuadrature(intersected, celldata,polynomialDegree + 1)
     return intersected, celldata, quadrature

intersected0, celldata0, quadrature0 = createBoundaryQuadrature(f"z <{origin[2] + 0.01*lengths[2]}")
intersected1, celldata1, quadrature1 = createBoundaryQuadrature(f"z >{origin[2] + 0.99*lengths[2] }") 

integrand0 = mlhp.l2BoundaryIntegrand(mlhp.vectorField(D, [penalty] * D), mlhp.vectorField(D, [0.0, 0.0, 0.0]))#wird in alle richtungen festgehalten & bestraft
integrand1 = mlhp.l2BoundaryIntegrand(mlhp.vectorField(D, [penalty] * D), mlhp.vectorField(D, [0, 0.0, -lengths[2]*0.01*penalty])) # verschiebung in z im 1%
#integrand1 = mlhp.neumannIntegrand(mlhp.vectorField(D, [1e3, 0.0, 0.0]))


mlhp.integrateOnSurface(basis, integrand0, [matrix, vector], quadrature0)
mlhp.integrateOnSurface(basis, integrand1, [matrix, vector], quadrature1)
#mlhp.integrateOnSurface(basis, integrand1, [vector], quadrature1)


time2=datetime.datetime.now()
print("Time needed: ", time2-time1)
##
time1=time2
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

time2=datetime.datetime.now()
print("Time needed: ", time2-time1)
##
time1=time2
print("5. Solving linear system", flush=True)

#P = mlhp.additiveSchwarzPreconditioner(matrix, basis, dirichlet[0])
P = mlhp.diagonalPreconditioner(matrix)

dofs, norms = mlhp.cg(matrix, vector, M=P, maxiter=10000, residualNorms=True)

#print(f"cond K after domain integral:{numpy.linalg.cond(matrix.todense())}")
#import matplotlib.pyplot as plt
#plt.loglog(norms)
#plt.show()

time2=datetime.datetime.now()
print("Time needed: ", time2-time1)
##
time1=time2
print("6. Postprocessing solution", flush=True)

outputDir="/Applications/Datein_Timon/Bildung/Uni_Rostock/Master_MaschBau/3_Semester/Studienarbeit/Code/FCM-StA/outputsWuerfel1" 

Ku = matrix * dofs
strainEnergy = 0.0
for ku, u in zip(Ku, dofs):
    strainEnergy += ku * u
strainEnergy = np.sqrt(strainEnergy)
print("Strain energy: %e" % strainEnergy )

with open(outputDir+"/convergence.csv", "a") as myfile:
    myfile.write("%d %e\n" % (dofs.size, strainEnergy))
    myfile.close()

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

time2=datetime.datetime.now()
print("Time needed: ", time2-time1)
print("Finish time:", time2)