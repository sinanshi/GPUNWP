#include<time.h>
#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<cuda.h>
/* Macro for mapping three dimensional index (ix,iy,iz) to 
 * linear index. The vertical index (z) is running fastest so 
 * that vertical columns are always kept together in memory.
 */

#ifdef DOUBLE_PRECISON
  #define REAL double
#else
  #define REAL float
#endif
#define Nx 128
#define Ny 128
#define Nz 128



/* Structure for three dimensional grid size */
struct N {
  int x;   // Number of grid points in x-direction
  int y;   // Number of grid points in y-direction
  int z;   // Number of grid points in z-direction
};

/* Number of gridpoints */
//struct N n;

/* Relative residual reduction target */
const REAL resreduction = 1.0e-5;

/* verbosity level */
const int verbose = 1;

/* parameters of PDE */
const REAL lambda2 = 1e4;
const REAL omega2 = 1.0;
const REAL delta = 0.0;

/* Use preconditioner? */
const int use_prec=0;

/* save field? */
const int savefields=0;

/*preconditioner options:
 *0-Block Jacobi
 *1-SOR
*/
const int prec_options=1;


/* Macro for mapping three dimensional index (ix,iy,iz) to 
 * linear index. The vertical index (z) is running fastest so 
 * that vertical columns are always kept together in memory.
 */
#define LINIDX(n, ix,iy,iz) ((n.z)*(n.y)*(ix) + (n.z)*(iy) + (iz))
#define GLINIDX(n, ix,iy,iz) ((n->z)*(n->y)*(ix) + (n->z)*(iy) + (iz))

#define BLOCK_SIZE 10


int gpu_apply(const N *n,const REAL *dev_x, REAL *dev_y);
void apply(const struct N, const REAL *x, REAL *y);

int gpu_bj(const N *n, const REAL *dev_b, REAL *dev_y );
int gpu_solver(const struct N n, const REAL *b, const REAL* x, REAL resreduction);
