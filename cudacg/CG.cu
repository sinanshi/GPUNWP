/* *********************************************************** *
 ***************************************************************
 ***                                                         ***  
 ***   Solve three dimensional Helmholtz model problem       ***
 ***                                                         ***
 ***  -omega2 * (d^2/dx^2 + d^2/dy^2 + lambda2*d^2/dz^2) u   ***
 ***  + delta u = RHS                                        ***
 ***                                                         ***
 ***  using the preconditioned Conjugate Gradient algorithm  ***
 ***  in a unit cube.                                        ***
 ***                                                         ***
 ***************************************************************
 * *********************************************************** *
 *
 *  The model parameters are omega2, lambda2 and delta. For 
 *  the full model problem, delta = 1, but we can also solve
 *  a Poisson problem by setting delta = 0.
 *
 *  Typically the vertical coupling (lambda2) is very strong 
 *  and vertical line relaxation will be very efficient as a 
 *  preconditioner.
 *
 *  A cell centred finite volume discretisation is used (but 
 *  in the operator we divide by the volume of a gridcell, 
 *  and the discretisation stencil becomes identical to that
 *  from a finite difference approximation).
 *
 *  The centres of the gridcells are located at
 *  
 *    (x,y,z) = ((ix+0.5)*hx, (iy+0.5)*hy, (iz+0.5)*hz)
 * 
 *  where ix = 0...nx-1, iy = 0...ny-1, iz=0...nz-1 and 
 *  hx = 1/nx, hy = 1/ny, hz = 1/nz.
 *
 *  Homogenous Dirichlet boundary conditions are used on all
 *  external boundaries. 
 *
 *     Eike Mueller, University of Bath, Feb 2012
 *
 * *********************************************************** */
#include<time.h>
#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<cublas_v2.h>
#include"CG.h"



//int gpu_apply(const N n,const REAL *x, REAL *y);


N n;

/* *********************************************************** *
 * Copy three dimensional field vector
 * y -> x
 * *********************************************************** */
void copy(const struct N n,
	  const REAL* x, 
	  REAL* y) {
  unsigned long n_lin = n.x * n.y * n.z;
  unsigned long i;
  for (i = 0; i<n_lin; i++) {
    y[i] = x[i];
  } 
}
/* *********************************************************** *
 * saxpy operation for three dimensional field vector
 * y -> y + alpha*x
 * *********************************************************** */
void saxpy(const struct N n,
	   const REAL alpha, 
	   const REAL* x, 
	   REAL* y) {
  unsigned long n_lin = n.x * n.y * n.z;
  unsigned long i;
  for (i = 0; i<n_lin; i++) {
	  y[i] += alpha*x[i];
  } 
}

/* *********************************************************** *
 * modified saxpy operation for three dimensional field vector
 * y -> alpha*y + x
 * *********************************************************** */
void saypx(const struct N n,
		   const REAL alpha, 
		   const REAL* x, 
		   REAL* y) {
  unsigned long n_lin = n.x * n.y * n.z;
  unsigned long i;
  for (i = 0; i<n_lin; i++) {
    y[i] *= alpha;
	  y[i] += x[i];
  } 
}

/* *********************************************************** *
 * scalar product of two three dimensional field vectors
 * <x,y>
 * *********************************************************** */
REAL dotprod(const struct N n,
		      const REAL* x,
			  const REAL* y) {
  unsigned long n_lin = n.x * n.y * n.z;
  unsigned long i;
  REAL d=0.0;
  for (i = 0; i<n_lin; i++) {
    d += x[i]*y[i];
  }
  return d;
}

/* *********************************************************** *
 * squared norm of three dimensional field vector
 * <x,x> = ||x||^2
 * *********************************************************** */
REAL norm2(const struct N n,
		    const REAL* x) {
  unsigned long n_lin = n.x * n.y * n.z;
  unsigned long i;
  REAL d=0.0;
  for (i = 0; i<n_lin; i++) {
    d += x[i]*x[i];
  }
  return d;
}

/* *********************************************************** *
 * apply operator to three dimensional field vector
 * y = A.x
 *
 * The discretisation stencil is:
 *
 *            ( - omega2*lambda2/hz^2 )   
 *                            
 *   (               -omega2/hy^2               )
 *   (                                          )
 *   ( -omega2/hx^2        D       -omega2/hx^2 ) 
 *   (                                          )
 *   (               -omega2/hy^2               )
 *
 *            ( - omega2*lambda2/hz^2 )   
 *
 *
 *  with   D = delta2 + 2*omega2*(1/hx^2+1/hy^2+lambda2/hz^2)
 *
 * Homongenous Dirichlet boundary conditions are enforced on
 * all boundaries.
 *
 * *********************************************************** */
void apply(const struct N n,
		   const REAL* x,
		   REAL* y) {
  int ix, iy, iz;
  // grid spacings in all directions
  REAL hx = 1./n.x;   
  REAL hy = 1./n.y;
  REAL hz = 1./n.z;
  REAL hx_inv2 = 1./(hx*hx);
  REAL hy_inv2 = 1./(hy*hy);
  REAL hz_inv2 = 1./(hz*hz);
  for (ix = 0; ix<n.x; ix++) {
    for (iy = 0; iy<n.y; iy++) {
      for (iz = 0; iz<n.z; iz++) {
        // Diagonal element
	      y[LINIDX(n, ix,iy,iz)]
          = (delta + 2.0*omega2 * (hx_inv2 + hy_inv2 + lambda2*hz_inv2))
          * x[LINIDX(n, ix,iy,iz)];
        // Off diagonal elements, enforce homogenous Dirichlet
        // boundary conditions 
		    if (ix>0)
	        y[LINIDX(n, ix,iy,iz)] += x[LINIDX(n, ix-1,iy,iz)] 
                                  * (-omega2*hx_inv2);
		    if (ix<n.x-1)
	        y[LINIDX(n, ix,iy,iz)] += x[LINIDX(n, ix+1,iy,iz)]
                                  * (-omega2*hx_inv2);
		    if (iy>0)
	        y[LINIDX(n, ix,iy,iz)] += x[LINIDX(n, ix,iy-1,iz)]
                                  * (-omega2*hy_inv2);
		    if (iy<n.y-1)
	        y[LINIDX(n, ix,iy,iz)] += x[LINIDX(n, ix,iy+1,iz)]
                                  * (-omega2*hy_inv2);
		    if (iz>0)
	        y[LINIDX(n, ix,iy,iz)] += x[LINIDX(n, ix,iy,iz-1)]
                                  * (-omega2*lambda2*hz_inv2);
		    if (iz<n.z-1)
	        y[LINIDX(n, ix,iy,iz)] += x[LINIDX(n, ix,iy,iz+1)]
                                  * (-omega2*lambda2*hz_inv2);
	    }
    }
  }
}

/* *********************************************************** *
 * Preconditioner
 * y = M^{-1}.b
 * SOR with vertical line relaxation using RB ordering of 
 * the degrees of freedom in the horizontal.
 *
 *  Initialise y to 0 and carry out the following
 *  relaxation steps n=1,...,niter times:
 *   
 *  y^{r,(n+1)}_i = (1.-alpha)*y^{r,(n)}_i
 *                  + alpha * (A^{rr}_{ii})^{-1} 
 *                      (b_i - \sum_{j!=i} A^{rb}_{ij} y^{(b,(n)}_j)
 *  y^{b,(n+1)}_i = (1.-alpha)*y^{b,(n)}_i
 *                + alpha * (A^{bb}_{ii})^{-1} 
 *                    (b_i - \sum_{j!=i} A^{br}_{ij} y^{(r,(n)}_j)
 *
 * ***********************************************************/
void prec_SOR(const struct N n,
		      const REAL* b,
		      REAL* y) {
  unsigned long n_lin = n.x * n.y * n.z;
  int ix, iy, iz;
  unsigned long i;
  // overrelaxation factor
  REAL alpha = 1.0;
  // Number of relaxation steps
  int niter = 4;
  // Grid spacings in all dirctions
  REAL hx = 1./n.x;
  REAL hy = 1./n.y;
  REAL hz = 1./n.z;
  REAL hx_inv2 = 1./(hx*hx);
  REAL hy_inv2 = 1./(hy*hy);
  REAL hz_inv2 = 1./(hz*hz);
  


  REAL* c;
  REAL* d;
  cudaMallocHost(&c, n.z*sizeof(REAL));
  cudaMallocHost(&d, n.z*sizeof(REAL));

  int iiter;
  // Independently loop over the (ix,iy) with
  // ix+iy mod 2 = 0 (color=red) and
  // ix+iy mod 2 = 1 (color=black)
  int color;
  REAL *r;                               // = malloc(n_lin*sizeof(REAL));
  cudaMallocHost(&r,n_lin*sizeof(REAL));


  // Initialise y to zero
  for (i=0; i<n_lin; i++) y[i] = 0.0;
  color=0; // Start with relaxing red lines 
  // Loop over number of relaxation steps
  for (iiter=0;iiter<2*niter;iiter++) {
    for (ix = 0; ix<n.x; ix++) {
      for (iy = 0; iy<n.y; iy++) {
        // Ignore the dof for the color we are not processing
        if (((ix+iy) % 2) == color) {
          // calculate b_i - sum_{j!=i} A_{ij} y_j
          for (iz=0;iz<n.z;iz++)
            r[LINIDX(n,ix,iy,iz)] = b[LINIDX(n,ix,iy,iz)];
          if (ix>0)
            for (iz=0;iz<n.z;iz++)
              r[LINIDX(n,ix,iy,iz)] -= (-omega2*hx_inv2)
                                              * y[LINIDX(n,ix-1,iy,iz)];
          if (ix<n.x-1)
            for (iz=0;iz<n.z;iz++)
              r[LINIDX(n,ix,iy,iz)] -= (-omega2*hx_inv2)
                                              * y[LINIDX(n,ix+1,iy,iz)];
          if (iy>0)
            for (iz=0;iz<n.z;iz++)
              r[LINIDX(n,ix,iy,iz)] -= (-omega2*hy_inv2)
                                              * y[LINIDX(n,ix,iy-1,iz)];
          if (iy<n.y-1)
            for (iz=0;iz<n.z;iz++)
              r[LINIDX(n,ix,iy,iz)] -= (-omega2*hy_inv2)
                                              * y[LINIDX(n,ix,iy+1,iz)];
          // Do a tridiagonal solve in the vertical direction
          // STEP 1: Calculate modified coefficients
          c[0] = (-omega2*lambda2*hz_inv2)
               / (delta+2.*omega2*(hx_inv2+hy_inv2+lambda2*hz_inv2));
          d[0] = r[LINIDX(n, ix,iy,0)]
               / (delta+2.*omega2*(hx_inv2+hy_inv2+lambda2*hz_inv2));
          for (iz = 1; iz<n.z; iz++) {
            c[iz] = (-omega2*lambda2*hz_inv2) 
                  / ( (delta+2.*omega2*(hx_inv2+hy_inv2+lambda2*hz_inv2)) 
                      - (-omega2*lambda2*hz_inv2) * c[iz-1]);
            d[iz] = (r[LINIDX(n, ix,iy,iz)] 
                     - (-omega2*lambda2*hz_inv2)*d[iz-1])
                  / ( (delta+2.*omega2*(hx_inv2+hy_inv2+lambda2*hz_inv2))
                     - (-omega2*lambda2*hz_inv2)*c[iz-1]);
          }
          // STEP 2: Back-substitution.
          r[LINIDX(n, ix,iy,n.z-1)] = d[n.z-1];
          for (iz = n.z-2; iz>=0; iz--) {
            r[LINIDX(n, ix,iy,iz)] 
              = d[iz] - c[iz]*r[LINIDX(n, ix,iy,iz+1)];
          }
          // Over-relaxation
          for (iz = 0; iz<n.z; iz++) {
            y[LINIDX(n,ix,iy,iz)] *= (1.-alpha);
            y[LINIDX(n,ix,iy,iz)] += alpha*r[LINIDX(n,ix,iy,iz)];
          }
  	    }
      }
    }
    // Swap color
    color=1-color;
  }
  cudaFree(c);
  cudaFree(d);
  cudaFree(r);
}




/* *********************************************************** *

* Block-Jacobi Preconditioner

* y = M^{-1}.b

* *********************************************************** */

void prec_BJ(const struct N n,
	      const REAL* b,
	      REAL* y) {
  int ix, iy, iz;
  // Grid spacings in all dirctions
  REAL hx = 1./n.x;
  REAL hy = 1./n.y;
  REAL hz = 1./n.z;
  REAL hx_inv2 = 1./(hx*hx);
  REAL hy_inv2 = 1./(hy*hy);
  REAL hz_inv2 = 1./(hz*hz);
  
  // Temporary arrays for Thomas algorithm
  REAL* c; //= malloc(n.z*sizeof(REAL));
  REAL* d;// = malloc(n.z*sizeof(REAL));
  cudaMallocHost(&c,n.z*sizeof(REAL));
  cudaMallocHost(&d,n.z*sizeof(REAL));
  // Loop over number of relaxation steps
  for (ix = 0; ix<n.x; ix++) {
    for (iy = 0; iy<n.y; iy++) {
      // Do a tridiagonal solve in the vertical direction
      // STEP 1: Calculate modified coefficients
      c[0] = (-omega2*lambda2*hz_inv2)
	/ (delta+2.*omega2*(hx_inv2+hy_inv2+lambda2*hz_inv2));
      d[0] = b[LINIDX(n, ix,iy,0)]
	/ (delta+2.*omega2*(hx_inv2+hy_inv2+lambda2*hz_inv2));
      for (iz = 1; iz<n.z; iz++) {
        c[iz] = (-omega2*lambda2*hz_inv2) 
	  / ( (delta+2.*omega2*(hx_inv2+hy_inv2+lambda2*hz_inv2)) 
	      - (-omega2*lambda2*hz_inv2) * c[iz-1]);
        d[iz] = (b[LINIDX(n, ix,iy,iz)] 
                 - (-omega2*lambda2*hz_inv2)*d[iz-1])
	  / ( (delta+2.*omega2*(hx_inv2+hy_inv2+lambda2*hz_inv2))
	      - (-omega2*lambda2*hz_inv2)*c[iz-1]);
      }
      // STEP 2: Back-substitution.
      y[LINIDX(n, ix,iy,n.z-1)] = d[n.z-1];
      for (iz = n.z-2; iz>=0; iz--) {
        y[LINIDX(n, ix,iy,iz)] 
          = d[iz] - c[iz]*y[LINIDX(n, ix,iy,iz+1)];
      }
    }
  }
  cudaFree(c);
  cudaFree(d);
}


/* *********************************************************** *
 * RHS for test problem with exact solution 
 * u(x,y,z) = x*(1-x)*y*(1-y)*z*(1-z)
 * *********************************************************** */
REAL frhstest(const REAL x,
              const REAL y,
              const REAL z) {
  REAL gx = x*(1.-x);
  REAL gy = y*(1.-y);
  REAL gz = z*(1.-z);
  return 2.*omega2*((gx+gy)*gz+lambda2*gx*gy)+delta*gx*gy*gz;
}

/* *********************************************************** *
 * Exact solution of test problem
 * u(x,y,z) = x*(1-x)*y*(1-y)*z*(1-z)
 * *********************************************************** */
REAL utest(const REAL x,
           const REAL y,
           const REAL z) {
  return x*(1.-x)*y*(1.-y)*z*(1.-z);
}

/* *********************************************************** *
 * RHS
 * *********************************************************** */
REAL frhs(const REAL x,
          const REAL y,
          const REAL z) {
  return z*(1.-z)*x*(1.-x)*y*(1.-y);
}

/* *********************************************************** *
 * CG solver, solve A.x = b for x
 * *********************************************************** */
void cg_solve(const struct N n,
              const REAL* b, 
    	        REAL* x,
              REAL resreduction) {
  unsigned int maxiter = 1000;
  unsigned int k; 
  unsigned long n_lin = n.x*n.y*n.z;
  REAL* r;// = malloc(n_lin*sizeof(REAL));
  REAL* z;// = malloc(n_lin*sizeof(REAL));
  REAL* p;// = malloc(n_lin*sizeof(REAL));
  REAL* q;// = malloc(n_lin*sizeof(REAL)); 
  cudaMallocHost(&r, n_lin*sizeof(REAL));
  cudaMallocHost(&z, n_lin*sizeof(REAL));
  cudaMallocHost(&p, n_lin*sizeof(REAL));
  cudaMallocHost(&q, n_lin*sizeof(REAL));
  

  REAL alpha, beta;
  REAL rnorm, rnorm0, rnorm_old,rz, rznew;
  clock_t start, end,start_a,end_a,start_p,end_p,start_s,end_s,start_d,end_d,start_n,end_n;
  double atime,ptime,stime,dtime,ntime;
  atime=0;
  ptime=0;
  stime=0;
  dtime=0;
  ntime=0;


  // Initialise
  start = clock();
  copy(n,b,r);// b -> r_0
  apply(n,x,q);       // A.x_0 -> q_0
  saxpy(n,-1.0,q,r);  // r_0 - q_0 = r_0 - A.x_0 -> r_0
  if (use_prec){
    if(prec_options==0)
      prec_BJ(n,r,z);
    if(prec_options==1)
      prec_SOR(n,r,z);
  } // M^{-1}.r_0 -> z_0
  else
    copy(n,r,z);
  copy(n,z,p);          // r_0 -> p_0
  rz = dotprod(n,r,z);
  rnorm0 = sqrt(norm2(n,r));   // ||r_{0}||^2 rnorm
  rnorm_old = rnorm0;
  end = clock();
  if (verbose == 1)
    printf("CG initialisation time = %12.8f s\n",
	   ((double)(end-start))/CLOCKS_PER_SEC); 
  if (verbose == 1)
    printf("Initial residual = %8.4e\n",rnorm0);
  start = clock();


  for (k=1; k<=maxiter; k++) {

    
    start_a=clock();
    apply(n,p,q);               // A.p_k -> q_k
    end_a=clock();
    atime+=(double)(end_a-start_a);
   
    start_d=clock();
    alpha = rz/dotprod(n,p,q);  // alpha_k = <r_k,z_k> / <p_k,A.p_k>
    end_d=clock();
    dtime+=(double)(end_d-start_d);

    start_s=clock();
    saxpy(n,alpha,p,x);         // x_k + alpha_k * p_k -> x_{k+1}
    saxpy(n,-alpha,q,r);        // r_k - alpha_k * A.p_k -> r_{k+1}
    end_s=clock();
    stime+=(double)(end_s-start_s);

    start_n=clock();
    rnorm = sqrt(norm2(n,r));   // ||r_{k+1}||^2 rnorm
    end_n=clock();
    ntime+=(double)(end_n-start_n);
 
    if (verbose == 1){
      printf(" iteration %d ||r|| = %8.3e rho_r = %6.3f\n",k,rnorm,rnorm/rnorm_old);
    }
    if (rnorm/rnorm0 < resreduction) break;
    
    if (use_prec){
      if(prec_options==0){
	start_p=clock();
	prec_BJ(n,r,z);
	end_p=clock();
      }
      if(prec_options==1){
	start_p=clock();
	prec_SOR(n,r,z);
	end_p=clock();
      }
      ptime+=(double)end_p-start_p;//calculate preconditioner time
    } 
    else
      copy(n,r,z);

    start_d=clock();    
    rznew = dotprod(n,r,z);     // <r_{{k+1},z_{k+1}> -> r2new
    end_d=clock();
    dtime+=(double)(end_d-start_d);

    beta = rznew/rz;            // beta_k = <r_{k+1},z_{k+1}> / <r_k,z_k>

    start_s=clock();
    saypx(n,beta,z,p);          // z_{k+1} + beta_k * p_k -> p_{k+1}
    end_s=clock();
    stime+=(double)(end_s-start_s);

    rz = rznew;                 // update <r_k, z_k> -> <r_{k+1},z_{k+1}>
    rnorm_old = rnorm;
  }
  end = clock();//timer:end of whole solver
  if (verbose == 1) {
    double solutiontime;
    solutiontime=(double)(end-start)/CLOCKS_PER_SEC;
    printf("Number of iterations = %4d\n",k);
     printf("rho_{avg}            = %6.3f\n",pow((double)rnorm/rnorm0,(double)1./k));
    printf("Solution time        = %12.4f s\n",solutiontime);
    printf("Apply time           = %12.4f s(%.2f %%)\n",atime/CLOCKS_PER_SEC,atime*100/CLOCKS_PER_SEC/solutiontime); 
    printf("Preconditoiner  time = %12.4f s(%.2f %%)\n",ptime/CLOCKS_PER_SEC,ptime*100/CLOCKS_PER_SEC/solutiontime);
    printf("saxpy  time = %12.4f s(%.2f %%)\n",stime/CLOCKS_PER_SEC,stime*100/CLOCKS_PER_SEC/solutiontime); 
    printf("dotprod   time = %12.4f s(%.2f %%)\n",dtime/CLOCKS_PER_SEC,dtime*100/CLOCKS_PER_SEC/solutiontime); 
    printf("normalize time = %12.4f s(%.2f %%)\n",ntime/CLOCKS_PER_SEC,ntime*100/CLOCKS_PER_SEC/solutiontime); 

    printf("Time per iteration   = %12.8f s\n", solutiontime/k); 
  }
  cudaFree(r);
  cudaFree(z);
  cudaFree(p);
  cudaFree(q);
}

/* *********************************************************** *
 * CG solver, solve A.x = b for x
 * *********************************************************** */
void save_field(const struct N n,
               const REAL* x,
               const char filename[]) {
  FILE* file_handle;
  unsigned long n_lin=n.x*n.y*n.z;
  unsigned long i;
  file_handle = fopen(filename,"w");
  fprintf(file_handle," # 3d scalar data file\n");
  fprintf(file_handle," # ===================\n");
  fprintf(file_handle," # Data is written as s(ix, iy, iz)\n");
  fprintf(file_handle," # with the rightmost index running fastests\n");
  fprintf(file_handle," nx = %10d\n",n.x);
  fprintf(file_handle," ny = %10d\n",n.y);
  fprintf(file_handle," nz = %10d\n",n.z);
  fprintf(file_handle," Lz = %10.6e\n",1.0);
  fprintf(file_handle," Ly = %10.6e\n",1.0);
  fprintf(file_handle," Lz = %10.6e\n",1.0);
  for(i=0;i<n_lin;i++)
    fprintf(file_handle," %12.8e\n",x[i]);
  fclose(file_handle);
}

/* *********************************************************** *
 * *********************************************************** *
 * M A I N
 * *********************************************************** *
 * *********************************************************** */
int main(int argc, char* argv[]) {
  // Ensure that both n.x and n.y can be divided by two to
  // allow red-black ordering in horizontal
  // int k;
   // for(k=0;k<30;k++){
  n.x = 100;//+64*k;
  n.y = 100;//+64*k;
  n.z = 100;
  printf(" parameters\n");
  printf(" ==========\n");
  printf(" nx        = %10d\n",n.x);
  printf(" ny        = %10d\n",n.y);
  printf(" nz        = %10d\n",n.z);
  printf(" omega2    = %12.6e\n",omega2);
  printf(" lambda2   = %12.6e\n",lambda2);
  printf(" delta     = %12.6e\n",delta);
  printf(" prec      = %d\n",use_prec);
    if(prec_options==0)
      printf(" precondtioner=Block Jacobi\n");
    if(prec_options==1)
      printf(" preconditoner=SOR\n");
#ifdef DOUBLE_PRECISION
    printf(" precision = DOUBLE\n");
#else
    printf(" precision = SINGLE\n");
#endif
#ifdef TEST
    printf(" solving TEST problem\n");
#endif
    unsigned long i;
    int ix, iy, iz;
  unsigned int n_lin = n.x*n.y*n.z;
  REAL *x;
  cudaMallocHost(&x, n_lin*sizeof(REAL));  

#ifdef TEST
  REAL *error;// = malloc(n_lin*sizeof(REAL));
  cudaMallocHost(&error, n_lin*sizeof(REAL));
  REAL L2error;
#endif
  REAL *r; //= malloc(n_lin*sizeof(REAL));
  REAL *b; //= malloc(n_lin*sizeof(REAL));
  cudaMallocHost(&r, n_lin*sizeof(REAL));
  cudaMallocHost(&b, n_lin*sizeof(REAL));
  REAL x_,y_,z_;
  // Set initial solution to 0
  for (i=0; i<n_lin; i++) {
    x[i] = 0;
    
  }
  
  

  
  //Initialise RHS
  for (ix=0; ix<n.x; ix++) {
    for (iy=0; iy<n.y; iy++) {
      for (iz=0; iz<n.z; iz++) {
		x_ = (ix+0.5)/n.x;
        y_ = (iy+0.5)/n.y;
        z_ = (iz+0.5)/n.z;
#ifdef TEST
	b[LINIDX(n,ix,iy,iz)] = frhstest(x_,y_,z_);
#else
	b[LINIDX(n,ix,iy,iz)] = frhs(x_,y_,z_);
	#endif
       
      }
    }
    }*/
  /* solve equation */
  gpu_solver(n,b,x,resreduction);
#ifdef TEST
  /* If we are solving the test problem, calculate the difference
   * between the exact solution, given by utest(), and the numerical
   * solution. The overall error is the L2 norm of this difference,
   * and it should scale like 1/n^2 where n is the number of 
   * grid points in one direction. Assuming that the numerical
   * solution is sufficiently exact, this will be a measure of the 
   * discretisation error.
   */
  // Initialise error
  for (ix=0; ix<n.x; ix++) {
    for (iy=0; iy<n.y; iy++) {
      for (iz=0; iz<n.z; iz++) {
        x_ = (ix+0.5)/n.x;
        y_ = (iy+0.5)/n.y;
        z_ = (iz+0.5)/n.z;
        error[LINIDX(n,ix,iy,iz)] = utest(x_,y_,z_);
      }
    }
  }
  // Substract numerical solution
  saxpy(n,-1.0,x,error);
  if (savefields)
    save_field(n,error,"error.dat");
  // Calculate L2norm of error
  L2error = norm2(n,error);
  L2error = 1./(n.x*n.y*n.z)*sqrt(L2error);
  printf("||error||_2       = %8.4e\n",L2error);
  printf("log_2 ||error||_2 = %8.4f\n",log(L2error)/log(2.));
#endif

  /* Save solution and resiudual */
  if (savefields) {
    save_field(n,x,"solution.dat");
    apply(n,x,r);
    saypx(n,-1.0,b,r);
    save_field(n,r,"residual.dat");
  }

  // Deallocate memory
  cudaFree(x);
  cudaFree(b);
  cudaFree(r);
#ifdef TEST
  cudaFree(error);
#endif
  // }
  return(0);
  
}
