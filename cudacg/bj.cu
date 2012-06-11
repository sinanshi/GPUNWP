#include<time.h>
#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<cuda.h>
#include"CG.h"



/* *********************************************************** *

* Block-Jacobi Preconditioner

* y = M^{-1}.b

* *********************************************************** */



__global__ void gpu_bj_kernel(const struct N *n,const float* b,float* y) {
  int ix, iy, iz;
  // Grid spacings in all dirctions
  float hx = 1./n->x;
  float hy = 1./n->y;
  float hz = 1./n->z;
  float hx_inv2 = 1./(hx*hx);
  float hy_inv2 = 1./(hy*hy);
  float hz_inv2 = 1./(hz*hz);
  
  //float *c,*d;
  // Temporary arrays for Thomas algorithm
  //cudaMalloc((void**)&c,n->z*sizeof(float));
  //cudaMalloc((void**)&d,n->z*sizeof(float));
  float c[1000],d[1000];
  //float* c=malloc(n.z*sizeof(float));
  //float* d=malloc(n.z*sizeof(float));
  // Loop over number of relaxation steps
  ix=blockIdx.x*BLOCK_SIZE+threadIdx.x;
  iy=blockIdx.y*BLOCK_SIZE+threadIdx.y;
  // for (ix = 0; ix<n->x; ix++) {
  // for (iy = 0; iy<n->y; iy++) {
      // Do a tridiagonal solve in the vertical direction
      // STEP 1: Calculate modified coefficients
      c[0] = (-omega2*lambda2*hz_inv2)/(delta+2.*omega2*(hx_inv2+hy_inv2+lambda2*hz_inv2));
      d[0] = b[GLINIDX(n, ix,iy,0)]/(delta+2.*omega2*(hx_inv2+hy_inv2+lambda2*hz_inv2));
      for (iz = 1; iz<n->z; iz++) {
        c[iz] = (-omega2*lambda2*hz_inv2)/( (delta+2.*omega2*(hx_inv2+hy_inv2+lambda2*hz_inv2))- (-omega2*lambda2*hz_inv2) * c[iz-1]);
        d[iz] = (b[GLINIDX(n, ix,iy,iz)]-(-omega2*lambda2*hz_inv2)*d[iz-1])/((delta+2.*omega2*(hx_inv2+hy_inv2+lambda2*hz_inv2))- (-omega2*lambda2*hz_inv2)*c[iz-1]);
      }
      // STEP 2: Back-substitution.
      y[GLINIDX(n, ix,iy,n->z-1)] = d[n->z-1];
      for (iz = n->z-2; iz>=0; iz--) {
        y[GLINIDX(n, ix,iy,iz)]=d[iz] - c[iz]*y[GLINIDX(n, ix,iy,iz+1)];
      }
      // }
      // }
}




int gpu_bj(const N *n, const REAL *dev_b, REAL *dev_y ){
 
 
  N *dev_n;
  cudaMalloc((void**)&dev_n,sizeof(N));
  dim3 dimBlock(BLOCK_SIZE,BLOCK_SIZE);
  dim3 dimGrid(n->x/BLOCK_SIZE,n->y/BLOCK_SIZE);
  gpu_bj_kernel<<<dimGrid,dimBlock>>>(dev_n,dev_b,dev_y);
  cudaFree(dev_n);
  return(0);
}


