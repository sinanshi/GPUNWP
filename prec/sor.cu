pply operator to three dimensional field vector
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
  // Temporary arrays for Thomas algorithm
  REAL* c = malloc(n.z*sizeof(REAL));
  REAL* d = malloc(n.z*sizeof(REAL));
  int iiter;
  // Independently loop over the (ix,iy) with
  // ix+iy mod 2 = 0 (color=red) and
  // ix+iy mod 2 = 1 (color=black)
  int color;
  REAL *r = malloc(n_lin*sizeof(REAL));
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
  free(c);
  free(d);
  free(r);
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
  REAL* c = malloc(n.z*sizeof(REAL));
  REAL* d = malloc(n.z*sizeof(REAL));
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
  free(c);
  free(d);
}
