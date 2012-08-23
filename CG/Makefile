


GPU Implementation of Iterative Solvers in Numerical Weather Predicting Models
Sinan Shi

June 20, 2012












MSc in High Performance Computing
The University of Edinburgh
Year of Presentation: 2012 
Abstract
This is the bit where you summarise what is in your dissertation.
 
Contents
Chapter 1 Introduction	1
1.1 Introduction	1
Chapter 2 Literature Review	3
Chapter 3 Background	4
3.1 Numerical Weather Predicting Models and Elliptic Equations	4
3.1.1 Met Office Unified Model	4
3.1.2 Helmholtz Equation in Atmospheric Dynamics	5
3.1.3 Characteristics of Discretised Helmholtz Equation	6
3.2 Iterative Methods for Linear System	9
3.2.1 Krylov Subspace Iterative Methods	9
3.2.2 Preconditioners	11
3.2.3 Memory Requirements for Conjugate Gradient Solver	12
3.3 GPU Architecture and CUDA Programming Model	13
Chapter 4 Matrix Free Implementation of CG Method	16
4.1 Matrix Free Matrix Vector Multiplication	16
4.2 Matrix Free Implementation of Preconditioner	17
4.3 Serial Code Performance Analysis	20
Chapter 5 GPU Implementation of Matrix Free CG Method	23
5.1 Structure and Design of GPU CG Solver	23
5.2 GPU Matrix Kernels	25
5.2.1 GPU Kernel: Matrix Vector Multiplication	25
5.2.2 GPU Kernel: Block Jacobi Preconditioner	26
5.2.3 GPU Kernel: SOR Preconditioner	27
5.3 Correctness and Rounding Errors	28
Chapter 6 GPU Kernel Performance Analysis and Optimisation	30
6.1 Optimisation Methodology and Strategies	30
6.1.1 Motivation	30
6.1.2 Performance Tuning Cases and Environment	31
6.1.3 Kernel Performance Bottleneck Analysis Method	31
6.1.4 Kernel Optimisation Strategies	32
6.2 Matrix Vector Multiplication Performance Analysis and Tuning	33
6.2.1 Algorithmic Analysis	33
6.2.2 Memory Latency Hiding and Memory Throughput	34
6.2.3 Memory Access Pattern	35
6.2.4 On Chip Memory: Shared Memory Usage	37
6.2.5 Performance Tuning	38
6.3 Performance Issues and Optimisation of Block Jacobi Preconditioner (Thomas Algorithm)	40
6.3.1 Thomas Algorithm Performance Analysis	40
6.3.2 Memory Access Pattern of Thomas Algorithm Implementation	41
6.3.3 Optimisation Options: On-Chip Memories	41
Chapter 7 Performance Testing Results and Analysis	44
7.1 Performance Tests for Kernels (notice gflop/peak)	44
7.1.1 Matrix Vector Multiplication Performance Testing Results	44
7.1.2 Preconditioners	45
7.2 Testing Results for CG Solver	46
Chapter 8 Discussions and Future Work	50
8.1 Further Optimisation	50
8.2 Boundary Conditions Issue	51
Chapter 9 Conclusion	52
Appendix A	Derivation of Helmholtz Equation in Dynamical Core	53
Appendix B	Stuff which no-one will read	55
Bibliography	56
 List of Tables
Table 1: Non-zero Entries Value in Each Row of Matrix A	6
Table 2: Operations in Iterative Methods	8
Table 3: List of Routines in Preconditioned CG	19
Table 4: Routines Execution Time Proportion in Total Execution Time	19
Table 5: BLAS Function and CUBLAS Routines	22
Table 6: 2D and 3-D Decomposition Implementation Comparison	24
Table 7: Memory Requirement Comparasion	25
Table 8: Naive 3-D ‘apply’ Function Tuning Factors	37
Table 9: GPU Kernels Speedup	45
List of Figures
Figure 2.1：Numerical Weather Forecasting with Different Resolution of Grid (reference and also explain what is shown)	4
Figure 2.2: Matrix Plot of 3-D Helmholtz System (4×4×4)	7
Figure 2.3: Memory Requirement for Iterative Methods	11
Figure 2.4: GPU Hardware(Fermi) Architecture [15]	13
Figure 2.5: 3-D Threads Scheduling (Block Size=4,4,4)	13
Figure 2.6: Warp Execution Plot	14
Figure 3.1: Demonstration of Matrix Vector Multiplication	16
Figure 3.2: Matrix Plot of Block Jacobi Preconditioner	18
Figure 3.3: Block Jacobi Preconditioner CG Solver Profiling Results	20
Figure 3.4: SOR Preconditioner CG Solver Profiling Results	21
Figure 4.1: CG Code Structure	22
Figure 4.2: Memory Access Pattern of 2D and 3-D Thread Partitioning	24
Figure 4.3: Single Precision CG Solver Residual Evolution	27
Figure 4.4: Double Precision CG Solver Residual Evolution	28
Figure 5.1: Source Code Modification Analysis: ‘apply’	32
Figure 5.2: Memory Throughput (GB/s) over Threads per Block	33
Figure 5.3:Memory Access Pattern of ‘apply’ function	35
Figure 5.4: Extra Memory Access Times	36
Figure 5.5: Global Memory Throughput	37
Figure 5.6: GPU Execution Time versus Thread Block Size	38
Figure 5.7: Source Code Modification Analysis: Thomas Algorithm (Block Jacobi)	39
Figure 5.8: Threads in Warps Assigned on Memory Address	40
Figure 5.9: Global Memory Throughput	41
Figure 6.1: Flop Rate of ‘apply’ Function	44
Figure 6.2: Flop Rate of Preconditioners	45
Figure 6.3: Speedup (problem size=(x,x,128))(double!)	46
Figure 6.4: Memory Transfer Proportion in CG Execution Time	47
Figure 6.5: Relative Difference between Summation of GPU Kernels Execution Time and Total Execution Time	48
Figure 7.1: Thread Scheduling after Changing Memory Allocation Coordinate	50
  
Acknowledgements 

 


Introduction
Numerical weather prediction models, such as the Unified Model developed by the UK Met Office[1], often require solving a 3-D linear positive definite Helmholtz partial differential equation in the dynamical core of the model. Numerical solution of such a large system normally consumes huge amounts of computation time. The Met Office currently is using a preconditioned Krylov subspace iterative solver to compute the numerical solution of such a Helmholtz equation. Recently, a new implementation method has been developed aiming to solve the Helmholtz equation faster. 
Matrix free implementation was designed for the Krylov subspace solver to solve the NWP Helmholtz equation. Matrix free implementation uses coefficients to implicitly store the matrix so that it reduces the memory requirement. In addition, by implementing a matrix free method, memory access requests will also be reduced, which will largely improve the performance in GPU implementation. 
With the increasing demand of NWP accuracy, the problem size for this Helmholtz equation may become much larger. The iterative solver in NWP is currently parallelised in multiple CPUs in MPI. The motivation of this project is to explore the implementation on GPU and investigate on the GPU performance, so that giving the knowledge of GPU implementation and the perspective of using GPU in the future, of NWP Krylov subspace iterative methods.
In this project, a Krylov subspace method called preconditioned Conjugate Gradient (CG) has been chosen to examine the benefits and effect of the matrix free implementation of Krylov subspace iterative solver on GPU. In CG solver, the most time consuming parts are matrix vector multiplication and preconditioners. GPU porting and optimisation will focus on these two kernels. In this project, a detailed analysis of matrix vector multiplication and two preconditioners, block Jacobi and red & black ordering SOR preconditioner, which both use Thomas (tridiagonal) algorithm, have been carried out. Optimisation guided by bottleneck analysis has also been carried out to provide a better memory access pattern and hiding global memory latency of GPU. By porting and initial optimisation on Fermi architecture GPU C2050, the performance of matrix vector multiplication kernel achieves 20 GFLOPS for single precision and 12 GFLOPS for double precision. The performance of a preconditioner is much lower due to bad memory access patterns caused by decencies of Thomas algorithm. The performances of both preconditioners are about 1 to 2.5 GFLOPS in both single and double precision while the performance of the SOR preconditioner is weaker. The block Jacobi preconditioned CG achieved 10X the speedup over the corresponding host CPU, while SOR preconditioned CG is about 4X the speedup.  According to the results from initial porting and optimisation, the matrix free preconditioned CG method obtained an acceptable speedup on GPU. It still has a large amount of potential for further improvement, i.e. changing the coordinate of memory alignment described in Chapter 7.




Literature Review
Iterative methods are particularly efficient on large linear sparse systems. The 3-D Helmholtz problem can also lead to a large sparse linear system, which is also a symmetric and positive definite system. It can be efficiently solved by Krylov subspace methods, such as Conjugate Gradient method [2].  The convergence rate was analysed in a detail by [2, 3]. According to the analysis, to impose an appropriate preconditioner when using CG method, may improve the convergence rate dramatically. The general theory of how to choose an appropriate preconditioner can be seen on [4, 5].  A tridiagonal preconditioner is suitable for solving the Helmholtz problem derived from a non-hydrostatic NWP model due to its vertical coupling attributes[6].  
CG can be paralleled on various devices, such as on shared memory or message passing systems [2]. Recently a new trend of using GPGPU as a platform of accelerating iterative solvers has attracted much attention. A large number of researches are focused on improving the performance of sparse matrix vector multiplication and preconditioners. In many cases, an appropriate matrix storage scheme can largely improve the performance. Commonly used storage schemes are Diagonal, COO (Coordinate), CRS (Compress Sparse Row), ELLPACK, Hybrid and etc.[7]. Speedup and Flop rate were commonly used to measure the performance of GPU kernels. In [7], the maximum flop rate they achieved is 35GFLOPS by DIA format and the 7-point stencil Laplace (same memory layout as Helmholtz)matrix is around 22 GFLOPS by using ELLPACK storage format.   Sparse matrix vector multiplication kernel was reported to be 12 times faster by Tesla C1050 to Xeon E5504 CPU, and the 3-D Poisson matrix (same memory layout as 3-D Helmholtz)can be solved in 5GFLOPS [8]. Parallelism of tridiagonal preconditioner solver is extremely hard. Some treatments have taken such Sparse Approximate Inverse algorithms to enable parallelism. In this project, by using block preconditioners, this problem can be solved by parallel executing small blocks. Due to less efficient parallelise in preconditioner, CG speedup normally is about a factor to 3[8]. In addition, generally, the performance of double precision GPU implementation can be much lower than single precision. Since the GPU double precision ability is very weak, some research resorts to avoid double precision CG by mixed precision iterative refinement algorithms [9]. Though double precision ability has been improved about 8 times in Fermi and in previous architecture, the double precision is still two times slower than single precision operations. It is also worthwhile to check that how different operation precision may affect the performance. 


Background 
Numerical Weather Predicting Models and Elliptic Equations
Met Office Unified Model
The Unified Model has been developed and used by the UK Met Office for both weather forecasting and climate prediction over two decades.  It simulates physical processes in the atmosphere by given initial values that come from observing factors such as, temperature, pressure and so on. One of the key components within the Unified Model is called dynamical core, which is used for calculating the evolution of large scale wind and thermodynamic properties. The dynamical core has been overhauled several times, and the name of currently used dynamical core is the New Dynamical Core[1].(maybe you could explain this better. Do you mean the name being used currently is the new dynamical core?)
Various configurations of Unified Models will be used for different purposes such as regional & global weather forecasting while the governing equations in a same dynamical core remain the same, which solves the three dimensional fully compressible, non-hydrostatic Euler equations which describe the atmospheric flow. These equations will normally be discretised on a regular latitude-longitude grid in horizontal. Without considering the affection to accuracy of the dynamic model itself, to reduce the grid size is equivalent to a more precise solution while forecasting the weather, as we may see the effect of different resolutions of grid that affect the weather prediction. Meanwhile, for the climate predicting model, it requires a fast solution for a long period simulation. Both of these requirements expect a fast solving of the governor equations in the dynamical core. Met Office dynamical core, currently runs global five day forecasts on a 1024×769×70 (5.5×〖10〗^7) grid and it requires to be completed within an hour. With the demand of more accurate predictions, the horizontal resolution will be largely increased; the more degree of freedom has to be processed in the same time. The Met Office Unified Model has been parallelised with MPI on CPUs, but with the requirement of much larger computation, it is worthwhile looking at other platforms such as GPU, as well.  
Helmholtz Equation in Atmospheric Dynamics 
The governor equations in the Unified model are fully compressible, non-hydrostatic, deep atmosphere three dimensional Euler equations. Here, we would briefly explain these atmospheric dynamics terminologies. First of all, fully compressible means that the change of density of fluid in the earth’s atmosphere can be represented by pressure. Traditionally, a numerical weather predicting system will use hydrostatic equations to assume a hydrostatic balance rather than non-hydrostatic equation in the Unified Model.  The non-hydrostatic system enabling the use of deep atmosphere approximation takes vertical momentum into account that leads to a more precise mathematical model. A non-hydrostatic model is necessary to be solved in a 3-D equation, as opposed to a 2-D, which is computationally more demanding as it has a larger number of degrees of freedom. In the compressible model there are fast acoustic modes, which limit the size of the explicit time step and is the reason for using semi-Lagrangian semi-implicit time stepping. Therefore, a 3-D Helmholtz equation should be solved in each time step for pressure correction. 
Generally, the governor equation set can be expressed by six separate equations in dry air, which contain momentum conservation equations of three wind components, a temperature equation, a mass conservation equation and a state equation. The full governor equations for a new dynamical core were given below[1, 10]. 
The final entire derived Helmholtz equation can be seen on a paper by Davies, T., et al [1]. The entire equation is too long to be displayed here. However, in this dissertation, we will focus on a simplified Helmholtz problem derived from the original in a unit cube with Cartesian geometry, for convenience of implementation while the key properties still remain[11]. The simplified model can be expressed as, 
	-ω^2 (∆u+λ^2  (∂^2 u)/(∂z^2 ))+u=R.H.S	2 1

	ω^2~((c_s αΔt)/R_earth )^2,         λ^2~(R_earth/H)^2  1/(1+(αΔt)^2 (N^(*0) )^2 )  	2 2

All coefficients listed above in ω^2and λ^2 are constant except Δt discretised time step which can be determined after the determination of problem size in implementation. Although, the coefficients vary for different problem sizes, it will keep constant when solving a single Helmholtz equation which requires to be solved in each time step in dynamical core. The coefficient ω^2and λ^2 which has been given the values for a range of problem sizes, also lead to another important property of Helmholtz equation in numerical weather predicting models is that,〖 λ〗^2 is much larger than ω^2 due to the fact that the depth of the atmosphere is much smaller than the radius of the earth. The solution variation is dominated by vertical components, i.e. ∂^2 u/∂z^2.This property is called vertical coupling which can be a guide to the using of a preconditioner. Research shows that preconditioners with only z retained can achieve a high convergence rate of CG. It is because that it will be very efficient when using a preconditioner that is similar to the original system to derive a similar result as original system so that in our case, if we solve an equation in which only the diagonal terms and the vertical coupling is kept, the solution will be similar to the solution of the full equation. 
Characteristics of Discretised Helmholtz Equation
To get the numerical solution of a partial differential equation will resort to a finite difference scheme, where the solution is only represented at the centre of grid cells on a regularly spaced three dimensional grid. Finite difference scheme is based on Taylor series expansion. The second order differentiation can be expressed as below. We assume the boundary in x,y and z direction’s are [a,b],[c,d] and [e,f] respectively which will then be partitioned into h_n equal parts. Therefore, the problem size in x,y,z  directions are all integers, and h_x=(b-a)/n_x , h_y=(d-c)/n_y , h_z=(f-e)/n_z .
	(∂^2 u)/(∂x^2 )=(u_((i+1,j,k) )+2u_((i,j,k) )+u_(i-1,j,k))/(h_x^2 )+O(h_1^2)	2 3

	(∂^2 u)/(∂y^2 )=(u_((i,j+1,k) )+2u_((i,j,k) )+u_((i,j-1,k)))/(h_y^2 )+O(h_2^2)	2 4

	(∂^2 u)/(∂z^2 )=(u_((i+1,j,k) )+2u_((i,j,k) )+u_((i-1,j,k)))/(h_z^2 )+O(h_3^2)	2 5

i∈[0,n_x-1], j∈[0,n_y-1], k∈[0,n_z-1]

-ω^2 ((u_((i+1,j,k) )-2u_((i,j,k) ) 〖+u〗_((i-1,j,k) ))/(h_1^2 )+(u_((i,j+1,k) )-2u_((i,j,k) ) 〖+u〗_((i,j-1,k) ))/(h_2^2 )+λ^2  (u_((i,j,k+1) )-2u_((i,j,k) ) 〖+u〗_((i,j,k-1) ))/(h_3^2 ))+u_((i,j,k) )=R.H.S
2 6
i∈[0,n_x-1], j∈[0,n_y-1], k∈[0,n_z-1]
According to those equations listed above, the 3-D Helmholtz equation can be represented in a discretised form as equation 2 6. When using Dirichlet boundary condition, boundary values are given in the fields where i=-1 and i=n_x. Since the boundary is not required to be calculated, only interior degrees of freedom will which are equal to n_x×n_y×n_z will be.  A larger problem size of always indicates a smaller h and therefore a smaller discretisation error term O(h_n^2). Therefore, justifying the statement in 1.1.1- a higher resolution of grid- will lead to a more precise weather forecasting model.  
For simplicity, in the following sections of this dissertation, letter N will be denoted as the total problem size, i.e. N=n_x 〖×n〗_y×n_z. 

	Ax=b	3 15


According to the equation (3 14), a N×N sparse matrix A can be generated for solving N unknown linear equations. Each row of the matrix contains only 7 non-zero entries out of N and this discretisation scheme is also called 7 stencil discretisation. The derived matrix plot from 3-D Helmholtz equation with homogenous Dirichlet boundary conditions has been shown in Figure 2, the blue points stand for non-zero entries in the matrix. It is necessary to mention that not all rows have seven non-zero entries. This is because the homogenous Dirichlet boundary condition has been imposed, where the boundary values have been set as zero. As we may see in Table 1, the value elements of each row in the matrix can be represented by coefficients,ω^2, λ^2 and discretisation density h which is related with the problem size. Therefore, it is no longer necessary to store the matrix; this is also discussed in section 2.2.3. This is the basic idea of matrix free implementation that a matrix can be substituted by coefficients. The advantages of using matrix free implementation will be discussed in Chapter 5.

Meanwhile, this plot also reveals a phenomenon that within this matrix, only 3 non-zero elements in each row out of 7 are aligned consecutively. This characteristic will also be adopted by memory allocation when implement in GPU. This matrix will be projected on a linear memory space. There are also 3 elements in this space that are consecutive. The detailed memory access pattern will be discussed in Chapter 9, 10.


Non-zero Entry Index	1st  & 5th	2nd & 6th	3rd & 7th	4th
Notation	A_x	A_y	A_z	A_0
Equation Term	u_((i-1,j,k) )	u_((i,j-1,k) )	u_((i,j,k-1) )	u_((i,j,k) )
Value 	-ω^2/(h_1^2 ) 	-ω^2/(h_2^2 ) 	-〖λ^2 ω〗^2/(h_3^2 ) 	(2(〖h_1^2+h〗_2^2+h_3^2 ) ω^2+1)/(〖h_1^2 h〗_2^2 h_3^2 ) 
Table 1: Non-zero Entries Value in Each Row of Matrix A

 
Figure 2.2: Matrix Plot of 3-D Helmholtz System (4×4×4)
Iterative Methods for Linear System
Krylov Subspace Iterative Methods
Solving a large sparse linear system, to use direct methods will be very inefficient both in terms of storage requirements and computational complexity. In last two decades, iterative methods have been the most popular way of solving large sparse systems. Currently, for solving the Helmholtz equation form the Met Office Dynamical Core was solved by a preconditioned Krylov subspace iterative solver Generalised Conjugate Residual (GCR) with Alternating Direct Implicit type preconditioner (ADI).[1]
For direct methods such as gauss elimination the complexity is O(N^3), while for iterative methods (CG) it is O(N), for one iteration, whereas in current Met Office dynamical core, N is around 5.5×〖10〗^7. This advantage will become obvious when the problem size is very large like the problem we handled in this project. The definition of an iterative method is a method that uses successive approximations by steps to reach an acceptable solution to solve a linear system[12]. Iterative methods such as Jacobi, Gauss-Seidel can be concluded as stationary whereas CG, CGR and GMRES non-stationary. Stationary methods are normally low cost in computation per iteration; however they may have a worse convergence rate than stationary methods for large systems, which means more iteration’s may be taken. The searching direction of stationary methods is arbitrary without considering the spectrum of the matrix which is difficult to estimate since it requires priori information. To overcome this limitation and attain a higher convergence rate, Krylov subspace methods are able to adaptively choose its parameters (searching direction) for each step of searching [5]. For solving a large system such as the Helmholtz equation we confronted, Krylov the convergence rate, in general will be faster than stationary methods.
The most expansive operation in iterative methods (without preconditioner) is matrix vector multiplication. Both stationary and non-stationary methods listed, require one such operation. In this project, Krylov subspace method, CG, will be focused on. From a computer science perspective, implementations of all Krylov subspace methods are very similar although numerically they can have different convergence properties and limitations. 
The idea of Krylov subspace iterative solver is to minimise the quadratic form f(x) generated by the linear system, Ax=b
	f(x)=1/2 x^T Ax-b^T x+c	4 1

Krylov subspace methods including CG, GCR, BiCG, GMRES and etc. are those which approach the solution by utilizing a projection process with a search subspace K_m which is a subspace of〖 R〗^n. 
	K_m (A,v)≡span{v,Av,A^2 v,…,A^(m-1) v
4 2

The approximate solution x_m will be successively replaced by x_m=x_0+K_m in m iterations, where x_0 is the initial guess of the solution. In order to have the fastest convergent rate, the search space must be perpendicular to the constraint space〖 L〗_m. (Petrov-Galerkin condition)
	b-Ax_m⊥L_m	4 3

Here,  L_m is a constraint subspace. Different Krylov subspace methods depend on the different choosing of〖 L〗_m, for example, in Conjugate Gradient Methods, L_m is identical to〖 K〗_m, while in GMRES, L_m=AK_m[2]. The name of this constraint in the CG method can be called A-orthogonal or conjugate. It is also means that the search direction d_(i+1) in CG should be conjugated(is that what you meant?) to previous search direction〖 d〗_i,
	d_(i+1)⊥Ad_i	4 4

Conjugate constraints guarantee the convergence of the method, while if this constraint was not well followed in the implementation, the CG method may fail. However, by introducing the rounding error, which is inevitable in the real implementations, this constraint cannot be fulfilled very well. A large number of iterations may provide enormous rounding errors and destroy the conjugate constraint which will cause the failure of CG. Hence, to control the rounding error within a certain tolerance is important to a successful CG implementation. The CG algorithm was shown below. In a preconditioned CG algorithm, one matrix vector multiplication and preconditioner inverting was needed which is the most computationally intensive part. Besides these two matrix operations, 5 vector operations are also required, which can be seen in Table 1.
Algorithm 1: Preconditioned Conjugate Gradient Algorithm

Initialize: 
r_0=b-Ax_0  ; z_0=M^(-1) r_0 
p_0=z_0; k=0
Loop:
      α_k=(r_k^T z_k )/(P_k^T AP_k )
      x_(k+1)=x_k+α_k P_k
      r_(k+1)=r_k+α_k P_k
      If r_(k+1) is small enough then exit
      z_(k+1)=M^(-1) r_(k+1)
      β_k=(z_(k+1)^T r_(k+1) )/(z_k^T r_k )
      P_(k+1)=z_(k+1)+β_k P_k
      k=k+1
Return




Dot Product	(S/D)axpy 	Matrix Vector Multiplication	Preconditioner
Conjugate Gradient	2	3	1	1
Table 2: Operations Requirement of Preconditioned CG Algorithm

Preconditioners
Theoretically, in absence of rounding errors, CG guarantees a N×N symmetric, positive definite system where the equation is solved in N steps. In practise, the error is reduced much faster, and already a small number of iterations can lead to a good solution. The convergence rate, which indicates how fast an iterative solver can converge, of Krylov subspace method is dominated by the spectrum of  A, and more specifically, the ratio of largest and smallest eigenvalues of A, i.e. spectrum property κ=λ_min 〖/λ〗_max. The maximum iteration i to achieve a certain factor of tolerance ϵ, (‖e_((i)) ‖_A≤ϵ‖e_((0)) ‖_A) is, 
	i≤[1/2 √κ  ln⁡(1/ϵ)]	4 5

The detailed derivation of convergent rate may refer to Kaniel(1966)[13] and Sewchunk (1994)[3]. By understanding the dominant factor of convergent rate, one may think that if we can solve a linear system by solving an equivalent system, which κ is much smaller than current one, then the convergence rate can be largely improved. According to this idea, the preconditioner has been developed by introducing a much better spectrum property preconditioner matrix M. 
	M^(-1) Ax=M^(-1) b	4 6

Hence, the original system A has been changed in M^(-1) A. What extra work we need to do here is to invert matrix M and calculate〖 M〗^(-1) b. It may introduce a tremendous amount of extra work by inverting the matrix M.
In addition, the Helmholtz equation is dominated by the vertical terms i.e.λ^2 dx^2/dz^2 is much larger than 1. It has been found that the efficiency of preconditioner can be largely improved by preconditioner matrix with only the vertical element retained[6]. It is easy to find on Figure 2, that if we only retain elements in the z direction, a tridiagonal matrix will emerge. Thus the main problem of a preconditioner we confronted in NWP will be a tridiagonal system. A relatively efficient way of inverting a tridiagonal matrix is using the Thomas algorithm. This and its implementation details will be discussed in session 5.2.
By using preconditioners, a large improvement of convergence rate may be obtained, while sometimes may also introduce a tremendous amount of extra work for each iteration. Using and choosing preconditioners should be well considered and investigated in different specific cases. This consideration is out of the scope of this project. However, its improved convergence rate cannot be ignored. Therefore, in this project, the performance of preconditioners, which were vertically inverted by the Thomas algorithm on GPU, will be investigated in order to give an initial and basic knowledge of the performance of preconditioners used by numerical weather predicting models. The two examples used in this project are Block Jacobi and red and black ordering Successive Over Relax (SOR) preconditioner, they both require solving a tridiagonal system at each horizontal grid point, i.e (x,y).

Memory Requirements for Conjugate Gradient Solver
The memory usage is also a big issue for the implementation of iterative methods for large sized problems, especially for their GPU implementation. In general, current GPU systems will have several Giga bytes memory. The GPU used in this project is TESLA C2050, which has 3GB memory. 3GB memory is equivalent to storing 8×〖10〗^8 / 4×〖10〗^8 single/double precision floating point numbers respectively. In current implementations, the Met Office requires(is that the right word?) the problem size as 5.6×〖10〗^7. To store such a big matrix in dense format the scheme requires storage of 3×〖10〗^15  (N^2 ) single precision numbers. Sparse matrix storage schemes, such as COO will cost around 1.5×〖10〗^9 (7×N×3) floating point numbers. Neither of these can meet the current requirement of problem size from a single TESLA C2050 GPU. Memory storage requirements for iterative methods have been listed below. As is commonly known, the most expansive memory transfer for the GPU programme is Host Device data transferring. Hence, there are two benefits from matrix free implementation caused in terms of no matrix storage. First, the most expansive matrix memory transfer from host to device can be eliminated. Second, a lesser number of GPUs will be needed in multiple implementations, since more data can be executed on a single GPU. As we may see in the table below, except the matrix, iterative algorithms still require storage of multiple vectors. With the increasing demand of accuracy on weather forecasting, a much larger problem size may be used in the near future. Multiple GPU implementations are required when the vector storage exceeds the capacity of GPU memory. 
Iterative Methods	Matrix storage	Vector storage
Conjugate Gradient	1	6
Figure 2.3: Memory Requirement for Iterative Methods

GPU Architecture and CUDA Programming Model 
GPU was originally designed for graphic processing with massive parallel threads architecture. In recent years, GPU has become more and more important in high performance computing and scientific computing. The difference of GPU architecture and CPU architecture gave both devices advantages in different prospects. CPU is optimised for sequential performance and good at instruction level parallels, pipelining and etc. With a powerful hierarchical cache, and scheduling mechanism, it has a very good sequential performance. A large proportion of transistors in CPU are contribute to scheduling mechanism or cache rather than calculation units, such as integer or floating point units. In contrast, GPU was designed for high instruction throughput(?) with a much weaker cache or scheduling ability. In GPU programming, users have to spend more time to ensure the good scheduling, load balancing and memory access, which is done automatically on a CPU. As a result, GPU kernels, defined by users, are always simple and are computation intensive. 
GPU has massive simple cores to execute a large number of calculations simultaneously. Nowadays, compared with the performance of GPU and CPU, GPU normally has a better performance with a factor of seven in both memory bandwidth and instruction throughput [14]. When facing intensive computing applications, such as the problem we faced in NWP, GPU is an ideal solution to speed up such applications. 
In current GPU architecture Fermi, 512 CUDA cores are distributed into 6 Streaming Multiprocessors. Memory systems of GPU include one off-chip global memory, and on-chip memories, such as L1 cache/shared, constant and texture memory. Each SM also contains a constant number of registers which is the fastest accessing memory on GPU. On-chip memories are about hundreds times faster than global memory. As a result, generally, to implementing on-chip memory is a very effective strategy when optimising GPU code.




 
Figure 2.4: GPU Hardware(Fermi) Architecture [15]
The hardware recourses, i.e. registers, cannot be directly scheduled by users and it is scheduled automatically by GPU according to the thread number defined by users. Threads are organised in 3-D thread blocks and thread blocks are organised in 2D grids. Each thread has a unique index to help the systems distinguish each other. The thread index in CUDA can be expressed as, Thread Index=BlockIdx.x×Block_Size+ThreadIdx.x. BlockIdx has two dimensions and ThreadIdx can have three dimensions. Thirty-two threads were organised in a warp and following the SPMD (Single Program Multiple Data) fashion, thirty-two threads in a warp execute one single instruction at the same time. Once a kernel launched, CUDA runtime will assign threads into warps. Warp execution in SM will be scheduled by a warp scheduler. The 3-D thread scheduling mechanism is that slices the thread block in a z dimension and distributes threads into warps. 

 
Figure 2.5: 3-D Threads Scheduling (Block Size=4,4,4)
The design of GPU warp scheduling system is based on memory latency hiding concerns. Rather than controlling memory latency to a tolerable extent with instruction levels parallel and cache like CPU, GPU can hide the latency by rapid switching among threads. When one warp stalls when waiting for data, the next will be executed, by the scheduler, simultaneously. Therefore, the latency can only be hidden with a large number of parallels. Little’s Law [16] tells us exact how much parallelism is needed for hiding the latency.
Concurrency=Memory Throughput×Memroy Latency
In Little’s Law, concurrency indicates how many concurrent executed threads should be launched to hide the latency. As long as the concurrency achieved throughput time’s latency, it is no longer to increase more concurrent threads. Sometimes, increasing threads will over use the hardware resources so that the level of performance drops. 
GPU optimisation strategy is different to CPU. The principle of GPU optimisation is to balance the usage of resource. Instruction throughput, memory throughput and hiding memory latency are three essential factors in GPU optimisation. (what are the techniques for doing this in practise? For example, by copying data to shared memory, only copy data between host and device when necessary, coalesce memory access and etc. )
 
Figure 2.6: Warp Execution Plot



Matrix Free Implementation of CG Method 
Matrix Free Matrix Vector Multiplication
There are two parts in preconditioned CG algorithms, which require matrix information, they are, matrix vector multiplication and preconditioners. The remaining operations are all about vector operations. 
A matrix can be entirely retrieved from those coefficients according to its geometric information. Therefore, the matrix is no longer necessary to be stored.  The way of how to retrieve the matrix for applying function is related to the memory alignment of the input vector. Data in 3 dimension space from both matrix and vector will be projected to a linear memory space. The current implementation of the projection is along vertical direction (z) and the linear index can be represented as, 
Linear Index (i_x,i_y,i_z )=n_x×i_x×i_y+n_y×i_y+i_z
(why z run slowest?)By setting the index conversion macro in the programme, each element of vector can easily find the correspondent coefficients of in the matrix, which are identical to its own index, for applying functions, since the same projection method has been imposed on both  matrix (virtual, represented by coefficient) and vectors. To explain this strategy in an intuitive way we may consider following cases. 
The mechanism of applying functions is all elements in each row of matrix multiplied by its corresponding vector elements. The first element of derived vector is first row of matrix multiply by input vector(I don’t understand what you mean but think you could explain the sentence better); the second element is the second row by input vector and so forth. Without considering boundary conditions, in the figure below, to derive the six elements of result vectors, red elements in the matrix will multiply by red elements in vector. According to Table 1, the element 1, 3, 5, 6, 7, 9 and 11 can be known as u_((i-1,j,k) ), 〖 u〗_((i,j-1,k) ), u_((i,j,k-1) ), u_((i,j,k) ), u_((i,j,k+1) ), u_((i,j+1,k) ), and u_((i+1,j,k) ) respectively. Thus, by traversing the input vector in the same manner, the applied function can be done. 
 
Figure 3.1: Demonstration of Matrix Vector Multiplication

Algorithm 2: Matrix Free Apply Function Ax=y

Loop i_x=0:n_x; i_y=0:n_y; i_z=0:n_z
    y[Linear Index(i_x,i_y,i_z )]   =A_0*x[Linear Index(i_x,i_y,i_z)]

    y[Linear Index(i_x,i_y,i_z )]+=A_x*x[Linear Index(i_x±1,i_y,i_z)]

    y[Linear Index(i_x,i_y,i_z )]+=A_y*x[Linear Index(i_x,i_y±1,i_z)]

    y[Linear Index(i_x,i_y,i_z )]+=A_z*x[Linear Index(i_x,i_y,i_z±1)]

End Loop
Compared with CRS, how many memory references do you save by using the matrix-free, when you do not have to load the matrix from memory ? of upi 
Matrix Free Implementation of Preconditioner
The Thomas algorithm is an efficient method based on the simplification of Gaussian elimination for inverting tridiagonal matrices. Tridiagonal matrix stands for a matrix with only non-zero elements in triple diagonals i.e. main diagonal (b_i), first diagonal upper the main diagonal (c_i) and first diagonal lower the main diagonal(a_i). Hence, a N×N tridiagonal matrix which stands for N linear equations can be solved by Thomas algorithm.

 
	a_i x_(i-1)+b_i x_i+c_i x_(i+1)=d_i	  5 1


i=1,…N
Algorithm 3: Thomas Algorithm

Coefficients Modification:
c_0^'=c_i/b_i 
Loop i= (1:N-1)
     c_i^'=c_i/(b_i-c_(i-1)^' a_i) 
End Loop
d_0^'=d_i/b_i 
Loop i= (1:N-1)
     d_i^'=〖(d〗_i-d_(i-1)^' a_i)/〖(b〗_i-c_(i-1)^' a_i)
End Loop
Back Substitution:
x_N=d_N^' 
Loop i= (N-2:0)
    x_i=d_i^'-c_i^' x_(i+1) 
End Loop

There is a fatal disadvantage of the Thomas algorithm for parallelism. Actually it cannot be paralleled due to flow dependency (Read after Write). Fortunately, this problem can be solved by block division of a large matrix that divides the large matrix into multiple blocks, and executes small blocks of matrix by the Thomas algorithm in parallel.  
Now we are going to define the block Jacobi preconditioner matrix M. All elements in a predefined subset S_i will have a same value as its corresponding elements in A. As we mentioned above, by only retaining vertical values in the preconditioner can achieve the best convergence rate of NWP Helmholtz equations. The subset was defined as on the following figure where only z direction elements have been preserved for preconditioner matrix M. Actually, each subset contains one vertical column (z=0:n_z ) of data. One can also easily find that the matrix within each subset. this is also a tridiagonal matrix. Therefore, the large matrix M can be inverted by individually inverting each block.  
 
Figure 3.2: Matrix Plot of Block Jacobi Preconditioner
It is also very straight forward to convert a general Thomas algorithm into its matrix free implementation for the block Jacobi preconditioner. The only thing is required chang is substituting〖 a〗_i and  c_i as coefficient〖 A〗_z; b_i as coefficient〖 A〗_0. Since each block takes responsibility for one vertical column, the final implementation can be written as follows. To combine algorithm 3 and 4, the algorithm requires 10 floating point instructions and 16 floating memory points of access are required to be entered. This information will be useful guideline of optimisation process, in session 6.3.1.
Algorithm 4: Block Jacobi Preconditioner

Loop i_x=0:n_x-1;  i_y=0:n_y-1
     Thomas Algorithm: invert S_(i_x,i_y )
End Loop

Another preconditioner will be used in this project as well, which is red and black ordering successive over relax preconditioners (SOR). Like block Jacobi, SOR also require inverting an M as what it does in block Jacobi preconditioner. Due to over relaxation iterative method’s nature, SOR can only be solved in several iterations. A residual requires to be updated in every iteration.
	r=b-Ax	5 2

To calculate the residual r, an applied function to calculate Ax is needed. Therefore, a SOR preconditioner can be roughly treated as a combination of ‘applied’ function and ‘block Jacobi preconditioner’. In SOR, when executing ‘apply’ function, only x and y direction will be counted. 
The red and black checkerboard has also been implemented. For each SOR iteration, only half of the elements will be updated, that can be distinguished by red and black. The colour will be sweep(swept? Changed?) at the end of each iteration. As a result, the problem size in both x and y directions should be divided by 2. Otherwise, the whole algorithm will fail. The SOR preconditioner algorithm is shown below. In the algorithm, niter and ω is the iteration number and relaxation coefficient, which are both very important to the convergence rate. However, it will not affect the performance per iteration. In all following chapters, nitter is assumed as 4 and ω as 1.  
Algorithm 5: SOR Preconditioner

Initialise:
Loop i=0:N
         y_i=0
End Loop
Loop k=1:2*niter
   Loop i=0:N
             if  i+j%2==color
                 〖r_i=b〗_i-∑_(i≠j)▒〖(A_ij∙ y_j)〗
                 r_i=M^(-1)∙r_i
                 y_(i+1)=(1-ω) y_i+ωr_i
             End if
    End Loop
     color++ 
Return

Serial Code Performance Analysis
Except for matrix operations such as ‘apply’ function or ‘preconditioner’, there are five vector operations. These and their functions have been listed in table below. All of these functions will at least be called once in CG iteration except ‘copy’ function which is only called once in initial process of preconditioned CG. 
Routine Name	Function	Calls
Saxpy(α,v1,v2,v_Result)
v_Result=α〖∙v〗_1+v_2	2
Saypx(〖α,v〗_1,v_2,v_Result)
v_Result=α〖∙v〗_1+v_2	1
Norm(v,Norm)
Norm=‖v‖	1
Copy(v_source,v_direction)	v_direction=v_source	1
Dot(v_1,v_2,s_Result)
s_result=v_1∙v_2	3
Table 3: List of Routines in Preconditioned CG
Tests of serial matrix free CG implementation have been done by testing problem sizes varying from 2×〖10〗^6 (128×128×128) to 1.3×〖10〗^8 (1024×1024×128). The function execution time proportion over total iteration execution time of each routine in iterations has been listed below. It reveals that both ‘preconditioner’ and ‘apply’ function is the performance bottleneck of CG code. Both functions occupy 80% and 94% of total execution time in block Jacobi and SOR CG respectively. In addition, the performances of all these subroutines are scaled linearly with an increasing number of problem sizes. In block Jacobi CG, the preconditioner is slightly faster than ‘apply’ function while in SOR-CG; the proportion of preconditioner is tremendous. It is not only because SOR combined ‘apply’ and ‘block Jacobi’, but also because that it has to be repeated four times as much as both ‘apply’ and ’block Jacobi’ by setting nitter=4. Therefore, the basic bottleneck of preconditioned CG is ‘apply’ and ’block Jacobi’. Without improve these two routines, SOR preconditioner will not be improved.  
Routine	Block Jacobi CG	SOR CG
Apply	42%	13%
Prec	38%	80%
Saxpy	8%	3%
Dot	9%	3%
Norm	2%	1%
Sum	99%	100%
Table 4 : Routines Execution Time Proportion in Total Execution Time
 
Figure 3.3: Block Jacobi Preconditioner CG Solver Profiling Results 
 
Figure 3.4: SOR Preconditioner CG Solver Profiling Results



GPU Implementation of Matrix Free CG Method 
Structure and Design of GPU CG Solver
 
Figure 4.1: CG Code Structure
GPU Implementation Code developed in this project is a prototype or a demonstration code of CG iterative solver. It contains 6 ‘.cu’ files including a single precision CG solver, a double precision CG solver, a GPU apply kernel, and two GPU preconditioner kernels. The code was run and debugged on HECToR GPU with TESLA C2050 GPU card. C2050 is a GPU using Fermi architecture and the computation ability is 2.0. The advantage of Fermi architecture has been talked about above. Notice that, when compiling double precision versions coded on compute capability 2.0 system, a compiler flag ‘-arch -sm_20’ must be added to indicate 2.0 architecture to the nvcc compiler. 
Routine Name	Calls	Corresponding CUBLAS Routine
Saxpy(α,v1,v2,v_Result)
2	CublasSaxpy
Saypx(〖α,v〗_1,v_2,v_Result)
1 	cublasCal+cublasSaxpy
Norm(v,Norm)
1	cublasNorm2
Copy(v_source,v_direction)	1	cublasCpy
Dot(v_1,v_2,v_Result)
3	Dot product
Table 5: BLAS Function and CUBLAS Routines
The whole CG solver has been ported on GPU. Except matrix operations, routines in BLAS (Basic Linear Algebra Subroutine), such as dot product, saxpy are required in CG solver. NVIDIA Company provides an optimised GPU BLAS library called CUBLAS library. In our CG solver, CUBLAS library will be used on all vector operations. The vector operation routines in CUBLAS library are very efficient; the maximum speedup to non-optimised CPU code can up to 32X (dot product).  CUBLAS routines which have been used in the CG solver have been listed above. As we may see in the last chapter, vector operations take nearly 20% of total execution time of CG. Hence, the maximum speedup we may get is 5X if we do not optimise those vector operations. Left those vector operations as un-optimised on CPU will cause an even worse effect, that a large amount of extra memory transfer will take place between host and device and the inherently sequential (think you need a word to follow sequential, nature maybe?) of CG algorithm even prevents the chance of executing vector operations on CPU overlapped with GPU kernel executions. By porting the whole CG solver including all matrix vector operations, GPU can eliminate these memory transfers since those accesses of memory will take place on the GPU global memory. In our current CG solver, for solving one Helmholtz equation, it only requires two times of memory transference. i.e. vector b and  x.
Threads number that assigned in thread blocks of CUBLAS routine will be automatically determined by the library. As a result, threads cannot be scheduled manually by users. As a result thread launch configuration cannot keep consistency with GPU kernels. Although consistency may take advantage of data reusability, it is still not feasible for implementation, since an optimised launch configuration for CUBLAS routines may cause a very low performance of other GPU kernels. The CG solver will be optimised as kernels rather than a whole CG solver. It also leads to a more flexible launch configuration for different GPU kernels i.e. well optimised launch configurations for each kernel can be taken without considering other parts of CG code. 
The CG code must be executed in sequence i.e. each current routine cannot correlate the execution with each other. Therefore, to build a big CG solver kernel has no meaning because it requires as many times of thread synchronises as separate kernels. Since there is no chance to parallel the whole CG algorithm, the only thing we could do is to parallel various operations on GPU, such as matrix vector multiplication, preconditioners, and vector operation. 

GPU Matrix Kernels
GPU Kernel: Matrix Vector Multiplication`
The matrix vector multiplication is called ‘apply’ function in the code. In ‘apply’ function, each launched thread takes responsible of deriving one element of the resulting vector. Therefore, the number of thread, in current implementation, has been assigned as the number of the problem size. The thread partitioning method in each thread block was in a 3-D partitioning.  The thread launch configuration (thread block size) is extremely important to the performance. 
A main concern over GPU optimisation is that how to assign a proper amount of work load to each and how to decompose those threads into thread blocks according to the compute resource of GPU. Therefore, threads partitioning method is extremely important. The most significant effect caused by different thread partitioning method is the difference of work load to each thread. In 3-D partitioning, the work load for each thread is one element, while for 2D is one column. We noticed that, in the ‘apply’ function, computation of each element solution can be done independently. Therefore, theoretically, to assign to assign only one element to each thread is feasible.   
 
Figure 4.2: Memory Access Pattern of 2D and 3-D Thread Partitioning
 
ix=blockIdx.x*BLOCK_SIZE+threadIdx.x;
iy=blockIdx.y*BLOCK_SIZE+threadIdx.y;
iz=blockIdx.z*BLOCK_SIZE+threadIdx.z;  

                                                                                                                         The problem of 2D partitioning is that it may achieve a very bad memory access pattern which will be described in optimisation chapters of preconditioner. A bad memory access pattern will downgrade the cache efficiency as well. At the same time, too much work load per thread affected the concurrency of execution. It is a very simple logic that if we assign the more work to every single thread, the less total number of thread that can be launched, while processing same amount of total work. By lacking of concurrently executing threads, the utility of hardware resources (occupancy) may also be very low. As it has been tested, the fastest naïve 2D partitioning ‘apply function’ is around 10 times slower than what only has a 3-D partitioning one. The final thread block size is set as 1, 2 and 64 in x,y and z direction respectively. The reason of setting exactly such a block size will be discussed in Optimisation.
	Naïve  2D Decomposition 	Naïve  3-D Decomposition
Max Occupancy (%)	63	100
Peak Global Memory Throughput (Gb/s)	115	115
Minimum GPU Execution Time (μs)	75353	7626
Flop Rate (Gflop)	1.4	14.3
Table 6: 2D and 3-D Decomposition Implementation Comparison

GPU Kernel: Block Jacobi Preconditioner
In the block Jacobi preconditioner, threads are not able to be assigned in 3-D as in apply function due to the flow decencies in z direction in the Thomas algorithm. Instead of 3-D a 2D partitioning has been used.  Each thread will take charge of processing one column of data as we may see in the figure. The flop rate and speed up to serial code(not sure if this makes sense) of the block Jacobi preconditioner is much worse than what the ‘apply’ function can achieve.  
Besides, there is also one minor change that deserves to be noticed. The size of temporary variable c^' and d^' in the Thomas algorithm is used to be n_z for storing modified coefficients, while in GPU implementation, the size should be enlarged. Since the algorithm is used to be executed in serial each Thomas algorithm was executed one after another. Therefore, by using c^' and d^'  as the size of the inverting matrix, i.e. n_z is enough. However in the GPU parallel implementation, multiple Thomas algorithms are updated by each thread which will then be executed concurrently so that multiple threads tend to modify the same segments of memory in the same time. Therefore, to enlarge the memory size to avoid such correlation is necessary. The size of d^' has been enlarged from n_x to N. In addition, since d^' is the most frequently reusing data in block Jacobi preconditioner, shared memory has been used for storing d^'. Issues about shared memory of block Jacobi preconditioner will be discussed in Chapter 10.
One might think that we forgot to enlarge the variable c^'. Indeed, in the original algorithm,  c^' deserves the same treatment as d^'. However, in the real implementation, the algorithm has been changed slightly. Recall the Thomas algorithm derivation of c^'
c_0^'=c_i/b_i
c_i^'=c_i/(b_i-c_(i-1)^' a_i)
a_i, c_i and b_i  can be represented as Helmholtz coefficients A_z and A_0 respectively, which means once the matrix has been determined, c_i^' can be easily derived without any other constraints. Furthermore, during the whole period of the execution of block Jacobi CG, this c_i^' will maintain the same. According to these attributes, the temporary coefficient c_i^'  will be calculated by CPU and transferred to GPU constant memory before the execution of kernel. Constant memory is on a chip memory, which is much faster than GPU global memory. The detail of implementation of this constant memory will be discussed in the following chapter. Initially, the calculation and memory transferring will be taken only once in the whole CG solver. However, due to the file scope constraint of using constant memory, in the current demonstration code, the calculation and memory transference of c_i^' will be executed before the launching of preconditioner kernel in all iterations. This flaw can be easily removed in future work. All testing results for preconditioner kernels will not count the cost of calculation and memory transference of c^' in CPU, except in the final CG solver benchmarking.   
Hence, the memory requirement for GPU matrix free CG solver can be determined which is 7n+n_z. Therefore, the single precision problem size of CG solver should be no more2508×2508×128, on a single GPU such as Tesla C2050 used in this project. However, it is still about 14 times larger than the problem size of the current Met Office requirement (5.6×〖10〗^7). 
	CPU Matrix Free CG solver	GPU Matrix Free CG solver
Memory requirement	6N+2n_z	7n+n_z
Table 7: Memory Requirement Comparasion

GPU Kernel: SOR Preconditioner
Red black ordering SOR preconditioner has also been implemented on CG solver. The SOR is also using the Thomas algorithm to invert the preconditioner matrix. Within these iterations the code is almost a combination of the ‘apply’ function and ‘block Jacobi’. The current version of SOR preconditioner separated those two parts and made them both a kernel within a loop. The advantage of a flexible thread launch configuration and the possible overhead by multiple launching kernels is worth trading-off. The current ‘apply’ function may have a maximum 30X speedup to the serial code while to invoke a kernel will cost less than 10 μs, [17] which is negligible compared to the speedup of the ‘apply’ function. In the current implementation, separate versions will be used instead of integrated ones due to a highly efficient 3D partitioning ‘apply’. 

Pseudo code: SOR
sor_initial kernel
Loop k=0:2*niter
    sor_apply kernel
    sor_thomas kernel
color=1-color
End Loop
We should notice that the red and black checkerboard ordering removes half of computation (data) in each SOR iteration. The colour will be swapped at the end each SOR iteration and only data in one colour can be executed in each iteration(I would advise you to think of a different word, don’t use iteration several times in a sentence). Since half of the problem has been reduced, half the amount of threads can be launched in SOR than in block Jacobi, in spite of block Jacobi already has a lack of concurrent executing threads. Therefore, the concurrency problem, especially in the Thomas algorithm part will become more serious than in block Jacobi preconditioner.  

Correctness and Rounding Errors
The correctness of GPU is based on comparison of the results with the original serial code, which is assumed as correct. Theoretically, GPU implementation results will not be identical with serial codes due to different rounding errors caused by different sequence of operations. 
CPU and GPU are both using IEEE 745 floating point standard[18]. The rounding error of some functions in preconditioned CG will not have a different rounding error of CPU version, such as dot products, because, usually, the different recursive sum ordering will lead to a different round error [19]. In preconditioned CG codes, matrix vector multiplication, preconditioners, and those CUBLAS routines such as saxpy and norm2. 
In this project, two methods of verification have been carried out. First, the maximum relative error will be monitored in all kernels by comparing serial code results and the GPU code.
Relative Error=|GPU Solution-CPU Solution|/(CPU Solution)
The results show that, the maximum relative error of all kernels in single precision is the factor of 〖10〗^(-7) and in double precision is the factor of 〖10〗^(-16). The single precision in IEEE 745 standard contains 7 significant figure in single precision and 16 in double. This may indicate the correctness of kernels. However, it is impossible to make a hundred per cent sure of the correctness by using this method. 
The second verification is to compare a known correct CG solution with the given initial values with GPU solution. In the given serial code, CG solution can be compared with a known correct solution and calculate its error by using square norm. By serial CG code, the solution error is O(〖10〗^(-5)). By testing GPU implementation, the square norm of GPU version is also O(〖10〗^(-5)). Since the rounding error, the convergence rate of CPU and GPU version codes may be different. In Figure 4.3 and Figure 4.4, the evolution of residual for both single and double precision has been shown. Serial single precision codes can have a large difference between GPU versions, but with the increasing of precision, i.e. by using double precision, the behaviour of both versions are identical.
 
Figure 4.3: Single Precision CG Solver Residual Evolution
 
Figure 4.4: Double Precision CG Solver Residual Evolution



GPU Kernel Performance Analysis and Optimisation 
Optimisation Methodology and Strategies 
Motivation
In the CUDA programming model, optimisation and tuning is extremely important and essential to the performance. GPU is a massively parallel device with a large number of CUDA cores in several Stream Multi-processors. Hardware resources will be distributed via the assignment of threads specified by users while users do not control these resources directly. GPU performance can be varied in a very large scale by adjusting the thread block size. However, GPU optimisation is not based on guessing or randomly tuning[20]. With a careful analysis to the performance and understanding of code, GPU optimisation is not that difficult.
GPU optimisation or tuning always takes place on a specific device. An optimised application on one device does not guarantee the performance on the others [20]. The performance will also vary while changing the size of the processed data or changing data precision e.g. single/double. Despite all these uncertain factors of GPU performance it is still worthwhile to investigate the GPU performance and optimisation, since once the code has been well investigated; all performance bottlenecks have been found and effective optimisation strategies have been successfully applied, an application can be optimised in other specific cases with much less effort. 
In this dissertation, an analysis focused on various performance limitation factors of kernels in matrix free CG implementation will be carried out. An initial optimised version has also been carried out to examine the potential performance of kernels of the matrix free method. The motivation of the optimisation process, in this project, is to provide the knowledge of performance limited factors and the optimisation strategies aimed to mitigate the effect of these factors. The performance limit factors demonstrated in this dissertation should always be considered or traded-off in subsequent optimisation, in the future. 
Performance Tuning Cases and Environment
The performance tuning and optimisation will be based on one case for each kernel. 3-D threads partitioning naïve porting for the ‘apply’ function and 2D threads partitioning naïve porting for preconditioners have been chosen as the case. Kernels having been selected i.e. ‘apply’ function and preconditioners, occupy at least 80% the execution time in serial code.  Both problem sizes were chosen as 256×256×128. The reason for choosing this problem size is that each dimension can be divided by 32 (warp size) and it fulfils one of the conditions of coalesced access that the starting location of memory access requests from a warp can be divided this amount. For coalesced access in other cases, memory padding may be implemented [21] while in this chosen case, it will not be a problem.  These chosen cases will be executed in single precision on the UK’s national supercomputer, HECToR. The testing GPU is Tesla C2050 with CUDA 4.0 version. Many testing results were carried out via NVIDIA COMPUTE profiler, which version is 4.0.17. 
Kernel Performance Bottleneck Analysis Method 
Before starting the optimisation process, it is necessary to figure out the method of defining the performance bottleneck of kernels. Generally, kernels including too much memory accessing than arithmetic instructions can be considered as memory bound kernels while, in contrary, the kernel requiring too much arithmetic computation than memory accessing can be considered as compute bound. When the memory access latency is not well hidden, the kernel can be considered as latency bound [14, 22]. 
To identify kernel bottleneck, two ways can be taken. First, the bottleneck can be defined by instruction byte ratio. GPU devices have a theoretical instruction byte ratio, which is given by peak instruction throughput divided by peak memory throughput. The theoretical instruction byte ratio on Tesla C2050 is 3.5. Any instruction byte ratio that is much larger or smaller than the theoretical ratio can be considered as imbalanced; the larger ratio indicates compute bound and smaller ratio indicates memory bound. This ratio can either be acquired by algorithm analysis and profiler result [22]. Algorithmic analysis is that one manually counts how many instructions and byte requirements are to be transferred within the algorithm and to compare them, while profiler also provides the actual rate according to instruction issued and requests of L2 cache [23]. Though profiler provides an exact result, the result will vary from 0.17 to 5.74 in ‘apply’ function with different thread block size, due to different memory access patterns. Therefore, the profiler provided result will only be a reference to the performance analysis.
The second way to define the kernel bottleneck is by separating source code [14, 22], into memory transfer only, maths only (arithmetic instruction). By measuring the execution time and comparing them with total time. When the memory only execution time is much larger, the code tends to be memory bound; when math only time gets much larger, it tends to be compute bound. The latency bound case is when the total time is much larger than either compute only or memory compute only execution time. This estimation can only give a rough idea of support analysis the code attributes. It cannot be treated as an accurate estimation, and it will only work when those scenarios mentioned above, become obvious. In addition, a kernel can sometimes have both serious problems in either memory or latency 
Kernel Optimisation Strategies
Once kernels’ bottleneck has been defined, the optimisation activities can be carried out aiming to mitigate the tension caused by either bottleneck. 
Latency bound kernel should resort to a large concurrently executed thread which can hide global memory access latency. The concurrency does not require to be maximised. It will be sufficient once Little’s Law has been fulfilled. 
To mitigate memory bound tense, one should focus on improving the memory access pattern and maximise the memory throughput. These two factors can be concluded as the quality and the quantity of memory transfer. A low quality memory access pattern will cause a heavy SIMD penalty, which increases memory requests due to large number of accessing failure. A low speed of global memory throughput will obviously increases the time of memory accessing time. Both statuses can be quantified by profiler results. Global memory throughput will be given by the profiler directly, while to examining the memory access pattern, one can use the extra memory request times, which can be calculated as follow,
Extra memory access times=(gld_read+gld_miss)-gld_request
The extra memory shows difference between actual global memory read and missed and global memory requests. The larger the gap between these two variables, the worse the memory access pattern. Ideally, the extra memory access time is zero, so that no SIMD penalty has taken. 
Implementing on chip memory, such as shared and constant memory is also an efficient way to improve global memory throughput by massive concurrent access and reducing global memory access requests by the reusing of data. 
To mitigate a compute tense can resort to maximise the instruction throughput. Some effective strategies are that to reduce redundant calculation and especially and to avoid brand (which may cause by conditional instructions) where possible. 
Optimisation actions should always aim to solve the bottleneck of the kernel. Effective strategies that aimed to compute bound kernels, sometimes, will not have any effect on memory bound kernels and vice versa. Therefore, analysis of the kernel first before optimisation is essential and it can save optimisation time and effort. In general, the principle of GPU optimisation for kernel is to maximise concurrency and to optimise memory usage and instruction usage [21]. However, these factors, sometimes conflict with each other. Performance leverage or tuning is very important, and some heuristic tuning is necessary. 

Matrix Vector Multiplication Performance Analysis and Tuning
Algorithmic Analysis
Before analysing the behaviour and performance limiting factors of the matrix vector multiplication function, we should have a look at the Algorithm 3.
From memory perspective, to update one element of result vector y, 7 read from different segment of global memory and 7 write to the same segment of memory, i.e. y, should be taken in matrix vector multiplication. Since 3-D partitioning of threads has been used, each thread corresponds to work that updating one element of resulting vector y. Therefore, for each thread, memory access is about 14 times if we do not take account of any cache effect. Accessing 14 times single precision data will have 56 byte memory transfer. Form computation perspective, to calculate an element of result vector requires 13 flops. Almost all threads are doing the same amount of work and the divergent effect is negligible, we can assume that the instruction byte ratio is 0.23 which is much lower than the theoretical ratio 3.57. The result from algorithm analysis indicates the ‘apply’ function is a memory bound kernel. 
By modifying source code on the kernel, testing results also indicates the same conclusion as what derived from algorithm analysis. The tests also reveal that the ‘apply’ kernel also suffers from a large memory latency problem, since the total time is much larger memory only execution time. 
 
Figure 5.1: Source Code Modification Analysis: ‘apply’ 
Therefore, the most urgent optimisation action that needs to be taken should focus around how to improve the quality (memory access pattern) and quantity (memory throughput) of memory access. Sometimes these two factors may contradict each other, so that trade-off is necessary.
Though there are six conditional instructions that are executed in the kernel due to Dirichlet boundary condition which is generally very expansive on GPU kernels, the condition instruction effect will not make a serious problem in such a latency bound code. To mitigate the effect of conditional instructions is one of the most effective ways of optimising compute bound codes, while sometimes this may not be very effective in memory and latency bound codes. Therefore, those branch effects will not be concerned at least during initial optimisation. 

Memory Latency Hiding and Memory Throughput
To hide memory latency, is a useful tactic is increasing the number of concurrently executed threads. Technically, by increasing the number of threads per thread block is equivalent to increasing concurrency.  
Occupancy is a guideline of performance for GPU applications. A higher occupancy always indicates a higher concurrency. The occupancy can be tested by Compute Profiler. Tests show that by increasing the number of threads per block will also increase occupancy. 
While a large number of memory access is requested in a very short time. The memory throughput can achieve a very high level, with the saturated memory bandwidth. Memory throughput of different thread configurations has been plotted. Since the threads were petitioned in 3-D, a total thread number can indicate several combinations in three dimensions. In this figure, only the highest throughput of different combinations was counted. The following figure shows that the maximum memory throughput of ‘apply’ function is 115GB/s, which is nearly 80% the peak memory bandwidth 144GB/s. 
 
Figure 5.2: Memory Throughput (GB/s) over Threads per Block
However, occupancy does not have to be maximised. The motivation of increasing occupancy or concurrency is to cover the memory latency according to the Little’s Law (parallelism=latency*throughput). More parallelism will not improve the performance when Little’s Law has been fulfilled. However, it may affect the other factors that decrease performance.
Memory bandwidth is also not the only guideline for performance. A large memory bandwidth sometimes is caused by repeating memory fetching failure due to bad memory access patterns.  The memory throughput of final optimised naïve  3-D version without shared memory is 60 GB/s, which is much lower than max throughput. In final tuning, memory throughput should be traded-off with other factors. 
Memory Access Pattern
Memory access pattern is extremely important to memory bound applications. The main concern is improving the performance of memory bound applications is to improve the memory access pattern. The access pattern may vary according to the thread launch configuration. The main concern about memory access pattern is whether the memory access is coalesced or not is the principle metric to evaluate access patterns. 
Threads distribution in 3-D decomposition implementation is relatively flexible, i.e. more threads can be launched than 2D. However, no matter how flexible it is, accessing scattered data for each thread is inevitable, i.e. the horizontal components n_((x±1,y,z) )  and n_((x,y±1,z)). With a high coalesced access pattern, the effect of scattered memory access can be mitigated and achieve a very high speed up (79x) , compared with non-coalesced access.
Before talking about memory access pattern, it is necessary to remind the GPU thread scheduling mechanism in kernel. Once a kernel has been launched, threads will be assigned to a warp according to its thread index, during the period from the beginning of a kernel execution till its termination. Following the SIMD model, threads in one warp will execute the same instruction collectively.  In current implementation, thread index corresponds to the working memory index, i.e. thread (ix,iy,iz) corresponds to〖 x〗_((ix,iy,iz)). 
To be coalesced accessing, firstly, the starting location of memory accessing should be divided by 32 (warp size). The starting location problem does not need to be considered in this case, since it has been intentionally eliminated by choosing the problem and block sizes. 
Secondly, threads in each warp should fetch the data that located a consecutive segment. Only those data along z direction are consecutively allocated in memory. Therefore, 32 threads in a warp must correspond to a segment of 32 z direction data in order to make coalesced accessing. To be simplify, let’s consider a case of which block size in x and y direction are both 1. As long as the z dimension block size has been as 32 or any integer times 32, coalesced accessing may achieve, as we may see in the figure below. In current CUDA programming model, the maximum thread number can be assigned in z direction is 64. Therefore, the thread number in z direction should either be chosen by 32 or 64 in this case, which is the optimum option for z direction thread number.  
  
Figure 5.3 :Memory Access Pattern of ‘apply’ function
The block size in x,y direction will also affect the coalesced accessing. The thread scheduling of a warp is running along the z direction, as we mentioned in Section 3.3.2. In the former case, both block sizes in x and y directions were 1. Once we increase the thread number in either direction, though keeping the thread number in z direction maintaining 32. The situation will be changed so that a warp will read half of the non-consecutive data as shown on figure. With this example, one can easily derive a conclusion that with the increasing number of either the x or y direction, the coalesced level will be decreased. To verify this, tests that calculate the non-coalesced level, which is indicated by extra memory access times, have been carried out and the results are shown below.
As we estimated, the larger the block size in x and y directions, the worse the coalesced accessing level is. The extra memory access times can be reduced to almost zero if we minimise the block size in  x and y directions. An interesting phenomenon has been revealed on the figure that the coalesced accessing level is dependent on the multiplication of block size in the direction of both x and y (Block Size_x  ×Block Size_y). This phenomenon implies that either increasing either x or y direction will have the same effect on coalesced accessing level. 

 
Figure 5.4: Extra Memory Access Times

On Chip Memory: Shared Memory Usage
In general, shared memory may reduce global memory access by taking advantage of shared memory data reusability.  According to testing results, by using shared memory, it eliminates over half of global memory requests. To reduce actual memory accessing times is a very effective way to improve the performance of memory and the latency bound kernel.
Shared memory can also increase the memory bandwidth. In ‘apply’ function, it fetches all data needed in each block at the beginning of the kernel execution. For consequent operations, data required by arithmetic instructions will be obtained from shared memory, which is around 20 times faster than global memory. All data requests will come together at the point when once kernel launched. Large amounts of requests created simultaneously are the most effective way to saturate memory bandwidth. 
 
Figure 5.5: Global Memory Throughput 
By taking advantage of higher memory throughput and data reusability, the best GPU ‘apply’ function execution time with shared memory version (5312 μs) is around 1000 μs less than the best case without shared memory version (6300 μs).  Though it does not seem like a vast improvement, if we have a look at the gain of the CPU version, the speedup (execution time) of GPU implementation will become tremendous, about 75X and 63X respectively.
The most important limiting factor is shared memory bank conflict. Without avoiding shard memory bank conflicts, the advantage of shared memory may be concealed. However, in our current implementation, the shared memory bank conflict is very high, according to the data provided by the profiler. Bank conflicts will be increased by increasing the block size in x and y direction.   However, bank conflicts maybe avoided such as by a padding tactic [24]. Due to the limited amount of time of this project, the bank conflict problem has not been treated very well in the current version of apply function.  

Performance Tuning 
To summarise the above factors, occupancy coalesced accessing and cache utility are the three most important factors required to be concerned with apply kernels. These processes largely rely on the tuning of block size of each thread block. Here is a table of the actions that may be taken on improving those factors. 
Performance Limit Factor	Improvement Methods
Coalesced Access		Increase block size in z
	Reduce block size in x &y
Memory Throughput		Maximise block size
	Shared Memory 
Latency Hiding		Increase block size
	Shared memory 
Table 8: Naive 3-D ‘apply’ Function Tuning Factors
The maximum total block size in CUDA is 512 threads and (512, 512, 64) in each dimension. If we minimise block size in both x and y direction, it will cause a very low memory throughput; On the other hand, if we increase them, memory access patterns will get worse. It is impossible to define which factor is more important than the others. Therefore, a heuristic tuning is necessary with the guide of the above analysis. The figure below demonstrated that increasing block size along the z direction, time will decrease the total execution, while increasing block size in x and y direction, the execution time will initially decrease benefits by higher memory throughput, and decrease due to worse memory access patterns. Hence, a trade-off must be taken. The final optimisation code chooses block size of (1, 2, and 64). The memory throughput is 65GB/s.
 
Figure 5.6: GPU Execution Time versus Thread Block Size

Performance Issues and Optimisation of Block Jacobi Preconditioner (Thomas Algorithm)
Thomas Algorithm Performance Analysis
Thomas algorithm Block Jacobi preconditioner implements the Thomas algorithm, which is the most important algorithm dealing with preconditioners in NWP. The Thomas algorithm will be used in both block Jacobi and SOR preconditioner. It spent almost the same amount time as ‘apply’ function, in the matrix free CG implementation.  The coefficients substitution part in the Thomas algorithm is the main bottleneck of the preconditioner due to the flow dependencies disabling the 3-D partitioning of threads.
According to the Thomas algorithm in Algorithm 3, 10 flops and 14 times memory access are required. By fetching or writing 14 times the single precision floating point number, requires 56 bytes. Hence, the instruction byte ratio is 0.17 which indicates a serious memory bound kernel. 
By modifying the source code, the results have been shown below. Memory only code is much more time consuming than compute only code. As a result, block Jacobi preconditioner can be considered as a memory bound code. We may also find the total time is even larger than the summation of memory only and math only code, which indicates it also suffered from a serious latency problems.
 
Figure 5.7: Source Code Modification Analysis: Thomas Algorithm (Block Jacobi)
Memory Access Pattern of Thomas Algorithm Implementation
 
Figure 5.8: Threads in Warps Assigned on Memory Address
The memory access pattern is much easier to understand than the memory access pattern in the ‘apply’ function, while also much more inefficient than that. Due to 2D partitioning of threads in x and y dimensions, threads in warps will never access any consecutive data in z direction. Therefore, the memory access pattern could never be worse than ever, since none consecutive access will take place in a warp. Options to improve the memory access pattern in Thomas algorithm is to either rotate thread projection coordination or rotate the memory allocation coordination, which will be talked about briefly in the following chapter. In this chapter we will only focus on naïve porting of kernels.
 
Optimisation Options: On-Chip Memories 
Keeping such a bad memory access pattern, the performance is hard to be improved. As it has been mentioned above, to optimise the performance memory of latency bound kernels, the most efficient way is to reduce global memory access times. 
Shared memory has been used to improve the performance of Thomas algorithm. Instead of applying it on global input array b as it was in ‘apply’ function, shared memory, in Thomas algorithm, is applied on temporary coefficient d. For array b, it only requires one time of reading, while for temporary array d^'; it requires one write and two read it the whole process. 
According to Algorithm 3, temporary coefficient d^' and c^' are both the most frequently accessed array in the Thomas algorithm. Therefore, by using shared memory can largely reduce global memory access requests, as shown in figure below. It almost reduced 1/3 of global memory requests in each case. With the less number of memory requests, which implies reducing the slowest global memory transfer in kernel, the performance of the fastest configuration of Block Jacobi preconditioner with shared memory can be twice as fast as those without shared memory.
There is another issue when implementing shared memory. On Tesla C2050, the maximum shared memory size for each thread is 64Kb. Due to 2D partitioning of threads, the minimum memory requirement for each thread is 128×4 byte (512), i.e. a vertical column of data is 128 single precision numbers, when the block size has been set as 1. Therefore, the maximum thread number can be assigned in each thread block should be no more than 128 threads, which is only half of the maximum number. 
 
Figure 5.9: Global Memory Throughput
This is dramatically divergent between ‘apply’ and ‘block Jacobi’ that the evolution of memory throughput is affected by using shared memory. By using shared memory in ‘apply’ function, the memory throughput is always higher than those without shared memory. However, the situation is on the contrary in ‘block Jacobi’ case. The reason is that, in ‘apply’ function, the shared memory loading takes place at the beginning of the kernel which means all memory access was requested at the same time. As opposed to‘block Jacobi’, the shared memory load was scattered during kernel execution so that it did not contribute to the improvement of memory throughput. The lower memory throughput is likely caused by less global memory requests. The shared memory here improved the data’s reusability, but did not improve the concurrency of memory accessing.
As being mentioned, array c^' is also the most frequently accessed array in the Thomas algorithm. Some other treatment has been carried out due to the characteristic of c^'. In Thomas algorithm, c^' was calculated as follows, where〖 a〗_k, b_k and c_k can be retrieved once matrix coefficient〖 ω〗^2, λ^2, and matrix size have been determined. It is not required to be calculated in every iteration. Even in the whole CG solver, calculating of coefficients c for only one time is sufficient.
c_0^'=c_i/b_i

c_i^'=c_i/(b_i-c_(i-1)^' a_i)
Hence, another improvement of Thomas algorithm was to remove those redundant calculations of coefficient variable ‘c’. However, by eliminating unnecessary calculations, did not have a significant effect on kernel, since it still has to read variable c from the global memory. For such a reason, the variable c was designed, in this project, to be transferred to constant memory before the launch of the kernel. It will be stored in constant memory for the whole kernel execution period. Tests show that the total execution time of the best case with constant memory is 20% faster than that without. 



Performance Testing Results and Analysis 
Performance Tests for Kernels (notice gflop/peak)
In section 7.1, the performance of matrix vector multiplication and two preconditioner kernels will be investigated. To eliminate other factors that may affect the performance of kernels, each one has been tested by calling it from a separate main program. The metric of performance will be flop rate. Flop rate can be calculated as the minimum algorithm floating point instruction required divided by total execution time. It is a relatively objective metric comparing with the speedup(is tis one word? If not, put speeding up) to CPU code. A direct comparison is not a good indicator of the performance gain from GPU porting as it is dependent on the specific CPU. The performance of CPU version code also varies a lot with different optimisation levels. Most research measures GPU kernels performance in flop rate [8, 17]. Execution time on GPU will be calculated based on GPU time given by NVIDIA Compute Profiler 4.0.17. In addition, memory transferring between host and device will not be counted in kernels.
Matrix Vector Multiplication Performance Testing Results
The performance of the matrix vector multiplication was tested by an increasing number of problem sizes. By testing those cases, which problem size vary form 128*128*128 to 640*640*128, the flop rate of both single and double precision matrix vector multiplications are almost continually constant. It is about 20GFLOPS for single precision and 12GFLOPS for double precision. The memory access pattern is not likely to be changed when facing different problem sizes, because the only difference of the larger size is that more thread blocks will be assigned while the block numbers that can be executed concurrently on SM are constant. Therefore, ideally, after the maximum launched block number has been reached on every SM, the memory access pattern will not be changed i.e. exceeding blocks will be executed after previous launched blocks and will follow their access pattern even after. 
The performance will be affected by the number but not the increasing number of problem size. Since all optimisation processes that were taken were aimed at specific cases, i.e. 256*256*128 in this dissertation, we cannot guarantee the memory access pattern can also be the most optimised on other cases. In application of GPU implementation in NWP, the problem size will be determinate and with simple tuning as has been demonstrated in this dissertation, a similar performance can be achieved.
Double precision performance is lower than single precision. It may be caused by two reasons. First, double precision peak performance (515Gflops) is only about half of the peak performance of single precision (1.03Tflops). Second, the memory access pattern, which was optimised for single precision kernel, may not be the most optimised kernel for double precision. 
 
Figure 6.1: Flop Rate of ‘apply’ Function
Preconditioners
The same tests have been taken on preconditioners. As we analysed in previous chapters, the performance of preconditioners are much worse than the matrix vector multiplication due to bad memory access patterns of 2D thread partitioning. The peak flop rate of the block Jacobi preconditioner is about 2.5Gflops and 2Gflops in single and double precision respectively. The flop rate of SOR single precision is nearly 2.5Gflops. The performance of SOR is worse than the block Jacobi preconditioner and much worse than the matrix vector multiplication. The double precision SOR varies dramatically. This behaviour is very difficult to explain without careful investigation. As the same reason we mentioned in 7.1.1, the code that has been optimised for only one configuration, does not guarantee a good performance for other cases. It is worthwhile to do further investigation when the specific requirements of NWP has been decided. 
 
Figure 6.2: Flop Rate of Preconditioners

Testing Results for CG Solver
The motivation of testing CG solver is slightly different to the kernels measurement. In the testing of CG solver, we want to measure exactly how much that GPU acceleration can benefit matrix free CG in solving the NWP Helmholtz problem. Therefore, the metric of benchmarking CG solver is different to GPU kernels. Speedup to CPU code will be taken as a main measurement of CG solver. The CPU code baseline in following tests will be compiled by ‘-O3’ optimisation level.
 
Figure 6.3: Speedup (problem size=(x,x,128))(double!)
The timing results of GPU CG solvers were carried out by the timing function. The results were measured by CPU clock and indicated the time that the whole GPU CG solvers consume. 
The highest speedup of block Jacobi-CG is 10X while SOR-CG is 4.3X. The figure also reveals that with the increase number of problem size, i.e. more computation, the speedup increases. It is probably caused by computation overhead being covered by large computation. 
	Speedup	Serial Code Proportion(Block Jacobi)	Serial Code Proportion(SOR)
saxpy	11X	8%	3%
dot	32X	9%	3%
nrm	21X	2%	1%
apply	25X	42%	13%
Block Jacobi	~10X	38%	/
SOR	~3.6X	/	80%
SUM	/	99%	100%
Table 9: GPU Kernels Speedup to Un-Optimised Serial CG
The speedup of kernels have been listed in the table above, the CUBLAS library routines achieved a very high speed up due to its high level optimisation. The scalability of CUBLAS routines is also very good i.e. the speedup maintains a constant for all routines when increasing the number of problem sizes. The GPU kernel we made in this project also reaches a relatively high level of performance. 
By comparing it with the kernels and entire CG speedup, we may find that the speedup of entire CG solver is lower than what we expected, if we estimate summing up kernel speedups by its original proportion.  
Here we provided two possible reasons for such a result. Firstly, there is extra memory transferring costs between the host and GPU device, including the transferring of coefficients c^' in Thomas algorithm. The proportion time of memory transferring from host to device in total execution time is almost maintained when increasing the number of problem size, around 3% for block Jacobi-CG and 0.5% for SOR-CG, regardless of the first several cases. The execution time of first case i.e. 128*128*128 takes only 0.1 seconds. Therefore, the memory transferring costs may become relatively large.
 
Figure 6.4: Memory Transfer Proportion in CG Execution Time
The second reason is that the GPU execution provided by compute profiler does not indicate the corresponding program execution time. It only indicates how much time that the GPU has worked. It does not count for kernel launch overhead and synchronisation costs between CPU and GPU. The relative difference of GPU kernels summation and the timing of CG solver can be calculated below. The difference of block Jacobi –CG is extremely high while the difference of SOR-CG is acceptable. With a larger computation, the gap between kernel summation and total time can be mitigating. It can also explain the phenomenon why SOR has a much narrower gap between kernel time summation and total time. However, the overhead of block Jacobi is much higher than what we expected. Some intensive investigations must be carried out in the future to investigate the exact reason of this phenomenon.
 
Figure 6.5: Relative Difference between Summation of GPU Kernels Execution Time and Total Execution Time



Discussions and Future Work 
Further Optimisation
In the apply function, to achieve the best memory access pattern with a large number of threads; the only option is to assign as much threads as possible to z direction. Unfortunately, the maximum thread number that can be assigned on z direction of thread block is 64 in the current CUDA version. Therefore, the way to maximise the thread number of the direction which is identical to contiguous alignment of memory can be either to change the coordinate of thread assignment i.e. threadIdx.x or threadIdx.y so it corresponds to consecutive data in z direction; or change the memory alignment coordinate, i.e. threadIdx.x or threadIdx.y correspond to data in x or y direction, but either x or y direction data are contigious rather than z direction before modification. 
The first is very easy to do; it only requires a minor change of thread projection while the second can be very complicated and trivial. The modification of memory allocation can introduce a very large programming effort, in this project, due to limit of time, this action has not been taken. 
The reason why we suggested changing the memory allocation method is based on the concern of the preconditioner. It is very obvious that, contiguous data in z direction cannot be parallelised due to the flow control problem of the Thomas algorithm. No matter how we change the thread assignment, data in z direction must be executed in a single thread, which means, threads in a warp cannot access contiguous data simultaneously. However, the situation can be changed if we rotate the memory allocating coordinate. Since the current biggest problem of preconditioners is memory access pattern problems, the performance is expected to have a large improvement if the memory access pattern has been improved. Changing the coordination, will benefit the memory access patterns for both matrix vector multiplication and preconditioner kernels, is worthwhile carrying out for further optimisation.

 
Figure 7.1: Thread Scheduling after Changing Memory Allocation Coordinate

Boundary Conditions Issue
The boundary condition in matrix vector multiplication can also be a problem. The ‘if’ conditional instruction that expresses the Dirichlet boundary condition, will cause granularly problems. In CUDA, a warp of threads should follow the same branch, i.e. executing the same instructions. A branch divergent inside a warp will cause stalls of some threads in a warp. 
In our case, 2n^2 threads n^3 problem size may be affected by boundary conditions. Branch divergent problems are not the primary issue of memory bound kernels. However, in future implementation, some different boundary conditions may be imposed, which should be decided by the requirement of NWP. 
By imposing different boundary conditions, the situation may change a lot, such as, for the Dirichlet boundary condition where all boundary values are equal to constant, for Neumann boundary conditions the derivative of the solution is given on boundary. A large number of computation and memory access will be taken place. Thus, it is probable that the characteristic of kernels apart from the above analysis(I don’t quite understand this sentence). Hence, reinvestigating the performance of kernels is necessary when changing boundary conditions.
There is also a choice to mitigate the boundary conditions, especially for eliminating branching. Halo can be imposed on the data to explicitly express the boundary, which can eliminate conditional instruction for boundary detection. For a specific, more computational expansive boundary condition, halo strategy can be considered.

 


Conclusion
In this project, porting and the initial optimisation of preconditioned CG Helmholtz solver has been carried out. Matrix free methods have a large advantage over other matrix storage schemes, due to less memory requirements and less memory access requests will be taken when executing matrix vector multiplication and preconditioners.  
Detailed performance bottleneck of GPU naive porting of matrix vector multiplication and Thomas algorithm for preconditioners have been analysed. Both matrix vector multiplication and Thomas algorithm are found memory bounded where memory is the main concern of improvement of performance. These kernels also suffer a severe problem of high memory latency. With adjusting memory access patterns and using on-chip memory, a large improvement of performance has been made to both kernels. Optimisation processes taken on these GPU kernels based on one single case analysis that a fixed problem size code has been used as the example and run on Tesla C2050 GPU. Though, performance of GPU kernels have a large variation with changing of environment or problem size, the analysis methodology can still be followed and the optimisation strategies can still be adopted in future optimisation of matrix free matrix vector multiplication and Thomas algorithm of preconditioners. 
Besides these two kernels, vector operations such as dot product, saxpy and norm, used in CG algorithm, have also ported on GPU via CUBLAS library. Due to its high level optimisation, these functions achieve a very high speed up to its host CPU. 
By initial porting and optimisation, GPU implementation of the performance of single precision block Jacobi preconditioned CG method improved about 10 times to its host CPU performance and SOR preconditioned CG speedup became about two times worse.  Double precision of both preconditioned CG runs much slower than the single precision. Some suggestions have been carried out for further optimisation according to analysis of kernels. The author believes that the performance of matrix free CG algorithm have further improvement. 
GPU implementation of matrix free Helmholtz iterative solvers, by taking advantage of GPU massive parallel nature and mitigating the bottleneck of GPU performance, may help to provide an efficient (an efficient what?).  


Derivation of Helmholtz Equation in Dynamical Core 
Momentum Equation
	Du/Dt-f_r v+f_ϕ w-(uv tan⁡ϕ)/r+uw/r+(C_p ϑ_v)/rcosϕ  ∂Π/∂λ=P^u	A-1
	Dv/Dt-f_r u+f_ϕ w+(u^2 tanϕ)/r+vw/r+(C_p ϑ_v)/r  ∂Π/∂ϕ=P^v	A-2
	Dw/Dt-f_ϕ u+f_λ v-(u^2+v^2)/r+g+C_p ϑ_v  ∂Π/∂r=P^w	A-3
Transport (Thermodynamic) Equation
	DX/Dt=P^X	A-4
Continuity Equation
	(∂ρ_dry)/∂t+∇_z∙(ρ_dry u)=0	A-5
Equation of State
	κΠϑ_v ρ=P/c_p 	A-6
Helmholtz equation was derived from the governing equations [1] above, inside of this equation set, D/Dt stand for Lagrange derivation or say material derivation. The definition of Lagrange derivation in spherical coordinate can be seen in Davis, T., et al [1], while  u,v,w stand for three wind components, λ,ϕ,r for longitude, latitude and radius coordinates respectively, ϑ_v for virtual potential temperature which is a temperature related with the moisture effect; P for the tendencies attained from the parameterization process and ρ for density. X is mix ratio for properties such as aerosols or chemical species [1]. Since the deep atmosphere model, Exner pressure Π has been used, which can be represented as 
	Π=(P/P_0 )^k	A-7
	k≡R_dry/c_p 	A-8
Where, c_p is the specific heat,P_0 represents reference pressure, R_dry is the gas constant for dry air.  These are the whole governor equations for the Unified Model. As we may see, it takes pressure, wind, moisture and even aerosols or chemical species into account. Together with all these six equations, can be seen as fully compressible, non-hydrostatic, deep atmosphere Euler equations. Meanwhile, these equations can be easily converted into Cartesian coordinates [10]. However, the continuous governor equations were not an elliptic type equation before they had been discretised in semi-implicit, semi-Lagrange scheme (SI LI). The SI LI discretisation process derived the continuous equations into one Helmholtz type elliptic equation, in which all unknown(unknown what?Factors/Element’s?) such as i.e. wind components, mix ratio and virtual temperature, except Exner pressure. By using SI LI discretisation process, both the governor equations of both New Dynamical Core and ENDGame can be derived into a Helmholtz problem.  [1, 11]



Stuff which no-one will read
Bibliography
1.	Davies, T., et al., A new dynamical core for the Met Office's global and regional modelling of the atmosphere. Quarterly Journal of the Royal Meteorological Society, 2005. 131(608): p. 1759-1782.
2.	Saad, Y., Iterative methods for sparse linear systems. 2003: Society for Industrial Mathematics.
3.	Shewchuk, J.R., An introduction to the conjugate gradient method without the agonizing pain. 1994, Carnegie Mellon University, Pittsburgh, PA.
4.	Axelsson, O., A survey of preconditioned iterative methods for linear systems of algebraic equations. BIT Numerical Mathematics, 1985. 25(1): p. 165-187.
5.	Benzi, M., Preconditioning techniques for large linear systems: a survey. Journal of Computational Physics, 2002. 182(2): p. 418-477.
6.	Skamarock, W.C., P.K. Smolarkiewicz, and J.B. Klemp, Preconditioned conjugate-residual solvers for Helmholtz equations in nonhydrostatic models. Monthly weather review, 1997. 125(4): p. 587-599.
7.	Bell, N. and M. Garland, Efficient sparse matrix-vector multiplication on CUDA. NVIDIA Corporation, NVIDIA Technical Report NVR-2008-004, 2008.
8.	Li, R. and Y. Saad, Gpu-accelerated preconditioned iterative linear solvers. Minnesota Supercomputer Institute, University of Minnesota, 2010.
9.	Cevahir, A., A. Nukada, and S. Matsuoka, Fast conjugate gradients with multiple GPUs. Computational Science–ICCS 2009, 2009: p. 893-903.
10.	Buckeridge, S., Numerical Solution of Weather and Climate Systems. 2010.
11.	Mueller, E., Model Problem Derivation, University of Bath Bath. p. 35.
12.	Barrett, R., Templates for the solution of linear systems: building blocks for iterative methods. 1994: Society for Industrial Mathematics.
13.	Kaniel, S., Estimates for some computational techniques in linear algebra. Mathematics of Computation, 1966. 20(95): p. 369-378.
14.	Brodtkorb, A.R., T.R. Hagen, and M.L. Sætra, GPU programming strategies and trends in GPU computing. Journal of Parallel and Distributed Computing, 2012.
15.	Nvidia, Whitepaper: Nvida's net generation CUDA compute architecture Fermi. 2009.
16.	Bailey, D.H., Challenges of future high-end computing. KLUWER INTERNATIONAL SERIES IN ENGINEERING AND COMPUTER SCIENCE, 1998: p. 23-36.
17.	Volkov, V. and J.W. Demmel. Benchmarking GPUs to tune dense linear algebra. 2008: IEEE.
18.	Whitehead, N., Precision &performance: floating point and IEEE 745 compliance for NVIDIA GPUs.
19.	Higham, N.J., The accuracy of floating point summation. SIAM Journal on Scientific Computing, 1993. 14: p. 783-783.
20.	Kirk, D., W.H. Wen-mei, and W. Hwu, Programming massively parallel processors: a hands-on approach. 2010: Morgan Kaufmann.
21.	Nvidia (2012) CUDA C Best Practices Guide v 4.1.
22.	Micikevicius, P., Analysis-Driven Optimization. 2010, Tutorial Slides: New Orleans.
23.	NVIDIA, Compute visual profiler user guide v 4.0. 2011.
24.	Harris, M., S. Sengupta, and J.D. Owens, Parallel prefix sum (scan) with CUDA. GPU Gems, 2007. 3(39): p. 851-876.


