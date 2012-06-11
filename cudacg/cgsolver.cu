#include<stdlib.h>
#include<stdio.h>
#include<string.h>





#include<cuda_runtime_api.h>
#include<cusparse_v2.h>
#include<cublas_v2.h>
#include"CG.h"



int gpu_solver(const struct N n, const REAL *b, const REAL* x, REAL resreduction){
  unsigned int maxiter=20;
  unsigned int k;
  unsigned long n_lin=n.x*n.y*n.z;
  REAL* r;
  REAL* z;
  REAL* p;
  REAL* q; 
  REAL* tq;
  const REAL negone=-1.0;
  N *dev_n,*l_n;
  cudaMallocHost(&r, n_lin*sizeof(REAL));
  cudaMallocHost(&z, n_lin*sizeof(REAL));
  cudaMallocHost(&p, n_lin*sizeof(REAL));
  cudaMallocHost(&q, n_lin*sizeof(REAL));
  cudaMallocHost(&l_n, n_lin*sizeof(N));
  cudaMallocHost(&tq,n_lin*sizeof(REAL));//for test qpply function
  REAL alpha, beta, temp;
  REAL rnorm, rnorm0, rnorm_old,rz, rznew;
 
  

  //GPU Memory Allocation
  REAL *dev_x,*dev_b,*dev_r,*dev_z,*dev_p,*dev_q;
  cudaMalloc((void**)&dev_x,n_lin*sizeof(REAL));  
  cudaMalloc((void**)&dev_b,n_lin*sizeof(REAL));
  cudaMalloc((void**)&dev_r,n_lin*sizeof(REAL));  
  cudaMalloc((void**)&dev_z,n_lin*sizeof(REAL)); 
  cudaMalloc((void**)&dev_p,n_lin*sizeof(REAL));  
  cudaMalloc((void**)&dev_q,n_lin*sizeof(REAL)); 
  cudaMalloc((void**)&dev_n,sizeof(N));



  //Memory copy

  l_n->x=n.x;
  l_n->y=n.y;
  l_n->z=n.z;

  

  cudaMemcpy(dev_n,l_n,sizeof(N),cudaMemcpyHostToDevice);
  cudaMemcpy(dev_x,x,n_lin*sizeof(REAL),cudaMemcpyHostToDevice);
  cudaMemcpy(dev_b,b,n_lin*sizeof(REAL),cudaMemcpyHostToDevice);
 
  //Initialise CUBLAS
  cublasHandle_t cublasHandle=0;
  cublasCreate(&cublasHandle);

  /*Initilise CG solver (Iteration 0)*/
  cublasScopy(cublasHandle,n_lin,dev_b,1,dev_r,1);
  gpu_apply(l_n,dev_x,dev_q);
  

  cublasSaxpy(cublasHandle,n_lin,&negone,dev_q,1,dev_r,1);//r_0=b_0-Ax_0
  if(use_prec){
    // gpu_bj(l_n,dev_r,dev_z);
  }
  else
    cublasScopy(cublasHandle,n_lin,dev_r,1,dev_z,1);
  cublasScopy(cublasHandle,n_lin,dev_z,1,dev_p,1);//r_0->p_0
  cublasSdot(cublasHandle,n_lin,dev_r,1,dev_z,1,&rz);
  cublasSnrm2(cublasHandle,n_lin,dev_r,1,&rnorm0);
  rnorm_old=rnorm0;

  printf("CG initial residual %8.4e\n",rnorm0);

  /*
   *CG Iteration
   */
  for(k=1;k<2;k++){

    gpu_apply(l_n,dev_p,dev_q);
    ///////////////////////////////////////////////////////////////////////////
    cudaMemcpy(p,dev_p,n_lin*sizeof(REAL),cudaMemcpyDeviceToHost);
    cudaMemcpy(q,dev_q,n_lin*sizeof(REAL),cudaMemcpyDeviceToHost);
    apply(n,p,tq);
    int err=0;
    int j;
    for(j=0;j<n_lin;j++){
      if(tq[j]!=q[j]){
	err++;
     	printf("%f---%f\n",tq[j],q[j]);
      }
    }
      printf("apply(%d) error=%d\n",k,err);
      /////////////////////////////////////////////////////////////////////////
    cublasSdot(cublasHandle,n_lin,dev_p,1,dev_q,1,&temp);
    alpha=rz/temp;
    float negalpha=0-alpha;
    cublasSaxpy(cublasHandle,n_lin,&alpha,dev_p,1,dev_x,1);
    cublasSaxpy(cublasHandle,n_lin,&negalpha,dev_q,1,dev_r,1);
    cublasSnrm2(cublasHandle,n_lin,dev_r,1,&rnorm);

    //    printf("iteration %d||r||=%8.3e rho_r=%6.3f, beta=%f, alpha=%f\n",k,rnorm,rnorm/rnorm_old,beta,alpha);

    if(rnorm/rnorm0<resreduction) break;

    if(use_prec){
      //gpu_bj(l_n,dev_r,dev_z);
    }
    else
      cublasScopy(cublasHandle,n_lin,dev_r,1,dev_z,1);
    cublasSdot(cublasHandle,n_lin,dev_r,1,dev_z,1,&rznew);
    beta=rznew/rz;
    cublasSaxpy(cublasHandle,n_lin,&beta,dev_z,1,dev_p,1);
    rz=rznew;
    rnorm_old=rnorm;
  }


  cublasDestroy(cublasHandle);
  cudaFree(dev_r);
  cudaFree(dev_z);
  cudaFree(dev_p);
  cudaFree(dev_q);

  return 0;

}

