#include<time.h>
#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<cuda.h>

/* Macro for mapping three dimensional index (ix,iy,iz) to 
 * linear index. The vertical index (z) is running fastest so 
 * that vertical columns are always kept together in memory.
 */
#define LINIDX(n, ix,iy,iz) ((n.z)*(n.y)*(ix) + (n.z)*(iy) + (iz))
#define GLINIDX(n, ix,iy,iz) ((n->z)*(n->y)*(ix) + (n->z)*(iy) + (iz))
#define len n.x*n.y*n.z

#define BLOCK_SIZE 5

/* Structure for three dimensional grid size */
struct N {
  int x;   // Number of grid points in x-direction
  int y;   // Number of grid points in y-direction
  int z;   // Number of grid points in z-direction
};

/* Number of gridpoints */
struct N n;

/* Relative residual reduction target */
//const float resreduction = 1.0e-5;


/* parameters of PDE */
const float lambda2 = 1e4;
const float omega2 = 1.0;
const float delta = 0.0;


/* *********************************************************** *

* Block-Jacobi Preconditioner

* y = M^{-1}.b

* *********************************************************** */

void prec_BJ(const struct N n,
	      const float* b,
	      float* y) {
  int ix, iy, iz;
  // Grid spacings in all dirctions
  float hx = 1./n.x;
  float hy = 1./n.y;
  float hz = 1./n.z;
  float hx_inv2 = 1./(hx*hx);
  float hy_inv2 = 1./(hy*hy);
  float hz_inv2 = 1./(hz*hz);
   float *c,*d;
  // Temporary arrays for Thomas algorithm
   cudaMallocHost((void**)&c,n.z*sizeof(float));
   cudaMallocHost((void**)&d,n.z*sizeof(float));
   // float c[100],d[100];
   //float* c=malloc(n.z*sizeof(float));
   //float* d=malloc(n.z*sizeof(float));
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





__global__ void gpu_bj(const struct N *n,
	      const float* b,
	      float* y) {
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
  float c[100],d[100];
  //float* c=malloc(n.z*sizeof(float));
  //float* d=malloc(n.z*sizeof(float));
  // Loop over number of relaxation steps
  for (ix = 0; ix<n->x; ix++) {
    for (iy = 0; iy<n->y; iy++) {
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
    }
  }
  //cudaFree(c);
  //cudaFree(d);
}




int main(){
  n.x = 20;
  n.y = 100;
  n.z = 20;
  int i;  
  
  printf(" parameters\n");
  printf(" ==========\n");
  printf(" nx        = %10d\n",n.x);
  printf(" ny        = %10d\n",n.y);
  printf(" nz        = %10d\n",n.z);
  printf(" omega2    = %12.6e\n",omega2);
  printf(" lambda2   = %12.6e\n",lambda2);
  printf(" delta     = %12.6e\n",delta);
  //int len=n.x*n.y*n.z;
  float x[len],y[len],gpu_y[len];
  for(i=0;i<len;i++){
    x[i]=1;
  }
  
  prec_BJ(n,x,y);



  N *h_n,*dev_n;
  h_n=(struct N*)malloc(sizeof(N));

  h_n->x=n.x;
  h_n->y=n.y;
  h_n->z=n.z;
  
  float *dev_b, *dev_y;


  cudaMalloc((void**)&dev_n,sizeof(N));
  cudaMalloc((void**)&dev_b,len*sizeof(float));
  cudaMalloc((void**)&dev_y,len*sizeof(float));


  cudaMemcpy(dev_n,h_n,sizeof(N),cudaMemcpyHostToDevice);
  cudaMemcpy(dev_b,x,len*sizeof(float),cudaMemcpyHostToDevice);
  dim3 dimBlock(BLOCK_SIZE,BLOCK_SIZE,BLOCK_SIZE);
  dim3 dimGrid(n.x/BLOCK_SIZE,n.y/BLOCK_SIZE,n.z/BLOCK_SIZE);
 
  gpu_bj<<<dimGrid,dimBlock>>>(dev_n,dev_b,dev_y);
  cudaMemcpy(gpu_y,dev_y,len*sizeof(float),cudaMemcpyDeviceToHost);
  


  
  for(i=0;i<len;i++){
    if(y[i]-gpu_y[i]>0.00001)
      printf("%f--%f\n",y[i],gpu_y[i]);
    else
      printf(".");
  }
  cudaFree(dev_n);
  cudaFree(dev_b);
  cudaFree(dev_y);
  


  return(0);
}


