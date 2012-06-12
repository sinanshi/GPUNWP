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

#define real float
/* parameters of PDE */
const real lambda2 = 1e4;
const real omega2 = 1.0;
const real delta = 0.0;


void apply(const struct N n,
		   const real* x,
		   real* y) {

  int ix, iy, iz;
  // grid spacings in all directions
  real hx = 1./n.x;   
  real hy = 1./n.y;
  real hz = 1./n.z;
  real hx_inv2 = 1./(hx*hx);
  real hy_inv2 = 1./(hy*hy);
  real hz_inv2 = 1./(hz*hz);
  //int i;
  // for(i=0;i<n.x*n.y*n.z;i++)
  // y[i]=0;
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

__global__ void gpu_apply(const N *n, const real *x, real *y){//(const N *n, const float* x, float* y){

 
  int ix, iy, iz;
  //grid spacings in all directions
  real hx = 1./n->x;   
  real hy = 1./n->y;
  real hz = 1./n->z;
  real hx_inv2 = 1./(hx*hx);
  real hy_inv2 = 1./(hy*hy);
  real hz_inv2 = 1./(hz*hz);

    
  ix=blockIdx.x*BLOCK_SIZE+threadIdx.x;
  iy=blockIdx.y*BLOCK_SIZE+threadIdx.y;
  iz=blockIdx.z*BLOCK_SIZE+threadIdx.z;



  // Diagonal element
   y[GLINIDX(n, ix,iy,iz)]=(delta+2.0*omega2 * (hx_inv2 + hy_inv2 + lambda2*hz_inv2))* x[GLINIDX(n, ix,iy,iz)];
  // Off diagonal elements, enforce homogenous Dirichlet
  // boundary conditions 
   __syncthreads();
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
  n.x = 10;
  n.y = 10;
  n.z = 10;
  int i;  
  //clock_t start1, end1;//start2,end2;  
  printf(" parameters\n");
  printf(" ==========\n");
  printf(" nx        = %10d\n",n.x);
  printf(" ny        = %10d\n",n.y);
  printf(" nz        = %10d\n",n.z);
  printf(" omega2    = %12.6e\n",omega2);
  printf(" lambda2   = %12.6e\n",lambda2);
  printf(" delta     = %12.6e\n",delta);
  int len=n.x*n.y*n.z;
  //cudaMallocHost 
  real *x,*y,*gpu_y;
  cudaMallocHost((void**)&x,len*sizeof(real));
  cudaMallocHost((void**)&y,len*sizeof(real));
  cudaMallocHost((void**)&gpu_y,len*sizeof(real));
  //float x[len],y[len],gpu_y[len];
  for(i=0;i<len;i++){
      x[i]=i;
  }
  
  apply(n,x,y);
  


  N *h_n,*dev_n;
  h_n=(struct N*)malloc(sizeof(N));

  h_n->x=n.x;
  h_n->y=n.y;
  h_n->z=n.z;
  
  real *dev_x, *dev_y;

  //start2=clock();
  
  cudaMalloc((void**)&dev_n,sizeof(N));
  cudaMalloc((void**)&dev_x,len*sizeof(real));
  cudaMalloc((void**)&dev_y,len*sizeof(real));


  int dx,dy,dz;
  dx=(int)ceil((double)n.x/BLOCK_SIZE);
  dy=(int)ceil((double)n.y/BLOCK_SIZE); 
  dz=(int)ceil((double)n.z/BLOCK_SIZE);

  cudaMemcpy(dev_n,h_n,sizeof(N),cudaMemcpyHostToDevice);
  cudaMemcpy(dev_x,x,len*sizeof(real),cudaMemcpyHostToDevice);
  dim3 dimBlock(BLOCK_SIZE,BLOCK_SIZE,BLOCK_SIZE);
  dim3 dimGrid(dx,dy,dz);//(n.x,n.y,n.z);
 

  gpu_apply<<<dimGrid,dimBlock>>>(dev_n,dev_x,dev_y);
  cudaMemcpy(gpu_y,dev_y,len*sizeof(float),cudaMemcpyDeviceToHost);
   

  int k=0;
  double er=0;
  for(i=0;i<len;i++){
    //printf("%f\n",gpu_y[i]);
    if(abs(gpu_y[i]-y[i])<1e-12){
      k++;
      er+=abs(gpu_y[i]-y[i]);
      printf("%f----%f(%d)\n",gpu_y[i],y[i],i);
    }
}
  printf(" error number=   %d(%f)\n",k,er/k);
 printf("x=%d,y=%d,z=%d\n",dx,dy,dz);  

  cudaFree(dev_n);
  cudaFree(dev_x);
  cudaFree(dev_y);
 
  



  return(0);
}
