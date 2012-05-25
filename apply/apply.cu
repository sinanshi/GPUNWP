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

void apply(const struct N n,
		   const float* x,
		   float* y) {

  int ix, iy, iz;
  // grid spacings in all directions
  float hx = 1./n.x;   
  float hy = 1./n.y;
  float hz = 1./n.z;
  float hx_inv2 = 1./(hx*hx);
  float hy_inv2 = 1./(hy*hy);
  float hz_inv2 = 1./(hz*hz);
  for (ix = 0; ix<n.x; ix++) {
    for (iy = 0; iy<n.y; iy++) {
      for (iz = 0; iz<n.z; iz++) {
        // Diagonal element
	y[LINIDX(n, ix,iy,iz)]=(delta+2.0*omega2 * (hx_inv2 + hy_inv2 + lambda2*hz_inv2))* x[LINIDX(n, ix,iy,iz)];
        // Off diagonal elements, enforce homogenous Dirichlet
        // boundary conditions 
	if (ix>0)
	  y[LINIDX(n, ix,iy,iz)]+= x[LINIDX(n, ix-1,iy,iz)]* (-omega2*hx_inv2);
	if (ix<n.x-1)
	  y[LINIDX(n, ix,iy,iz)]+= x[LINIDX(n, ix+1,iy,iz)]* (-omega2*hx_inv2);
	if (iy>0)
	  y[LINIDX(n, ix,iy,iz)]+= x[LINIDX(n, ix,iy-1,iz)]* (-omega2*hy_inv2);
	if (iy<n.y-1)
	  y[LINIDX(n, ix,iy,iz)]+= x[LINIDX(n, ix,iy+1,iz)]* (-omega2*hy_inv2);
	if (iz>0)
	  y[LINIDX(n, ix,iy,iz)]+= x[LINIDX(n, ix,iy,iz-1)]* (-omega2*lambda2*hz_inv2);
	if (iz<n.z-1)
	  y[LINIDX(n, ix,iy,iz)]+= x[LINIDX(n, ix,iy,iz+1)]* (-omega2*lambda2*hz_inv2);
      }
    }
  }
}

__global__ void gpu_apply(const N *n, const float*x, float *y){//(const N *n, const float* x, float* y){

 
  int ix, iy, iz;
  //grid spacings in all directions
  float hx = 1./n->x;   
  float hy = 1./n->y;
  float hz = 1./n->z;
  float hx_inv2 = 1./(hx*hx);
  float hy_inv2 = 1./(hy*hy);
  float hz_inv2 = 1./(hz*hz);

    
  ix=blockIdx.x*BLOCK_SIZE+threadIdx.x;
  iy=blockIdx.y*BLOCK_SIZE+threadIdx.y;
  iz=blockIdx.z*BLOCK_SIZE+threadIdx.z;

  // Diagonal element
  y[GLINIDX(n, ix,iy,iz)]=(delta+2.0*omega2 * (hx_inv2 + hy_inv2 + lambda2*hz_inv2))* x[GLINIDX(n, ix,iy,iz)];
  // Off diagonal elements, enforce homogenous Dirichlet
  // boundary conditions 
  if (ix>0)
    y[GLINIDX(n, ix,iy,iz)]+= x[GLINIDX(n, ix-1,iy,iz)]* (-omega2*hx_inv2);
  if (ix<n->x-1)
    y[GLINIDX(n, ix,iy,iz)]+= x[GLINIDX(n, ix+1,iy,iz)]* (-omega2*hx_inv2);
  if (iy>0)
    y[GLINIDX(n, ix,iy,iz)]+= x[GLINIDX(n, ix,iy-1,iz)]* (-omega2*hy_inv2);
  if (iy<n->y-1)
    y[GLINIDX(n, ix,iy,iz)]+= x[GLINIDX(n, ix,iy+1,iz)]* (-omega2*hy_inv2);
  if (iz>0)
    y[GLINIDX(n, ix,iy,iz)]+= x[GLINIDX(n, ix,iy,iz-1)]* (-omega2*lambda2*hz_inv2);
  if (iz<n->z-1)
    y[GLINIDX(n, ix,iy,iz)]+= x[GLINIDX(n, ix,iy,iz+1)]* (-omega2*lambda2*hz_inv2);
 
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
  int len=n.x*n.y*n.z;
  float x[len],y[len],gpu_y[len];
  for(i=0;i<len;i++){
    x[i]=1;
  }
  
  apply(n,x,y);
  N *h_n,*dev_n;
  h_n=(struct N*)malloc(sizeof(N));

  h_n->x=n.x;
  h_n->y=n.y;
  h_n->z=n.z;
  
  float *dev_x, *dev_y;


  cudaMalloc((void**)&dev_n,sizeof(N));
  cudaMalloc((void**)&dev_x,len*sizeof(float));
  cudaMalloc((void**)&dev_y,len*sizeof(float));


  cudaMemcpy(dev_n,h_n,sizeof(N),cudaMemcpyHostToDevice);
  cudaMemcpy(dev_x,x,len*sizeof(float),cudaMemcpyHostToDevice);
  dim3 dimBlock(BLOCK_SIZE,BLOCK_SIZE,BLOCK_SIZE);
  dim3 dimGrid(n.x/BLOCK_SIZE,n.y/BLOCK_SIZE,n.z/BLOCK_SIZE);
 
  gpu_apply<<<dimGrid,dimBlock>>>(dev_n,dev_x,dev_y);
  cudaMemcpy(gpu_y,dev_y,len*sizeof(float),cudaMemcpyDeviceToHost);
  
  









for(i=0;i<len;i++){
   if(abs(gpu_y[i]-y[i])>1)
     printf("x");
    else
     printf(".");
   //printf("%.2f----%.2f\n",gpu_y[i],y[i]);
   }

  //printf("x=%d\n",h_n->x);
  
// int count;
  //cudaDeviceProp prop;
  //cudaGetDeviceCount(&count);
  //cudaGetDeviceProperties(&prop,i);
  cudaFree(dev_n);
  cudaFree(dev_x);
  cudaFree(dev_y);
 

  return(0);
}
