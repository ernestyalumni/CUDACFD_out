/* boundary.h
 * fix boundary conditions at the "boundary strip" cf. Griebel, Dornseifer, & Neunhoeffer (1997)
 * \author Ernest Yeung  
 * \email ernestyalumni@gmail.com
 * \date 20161207
 * 
 * compilation tip: (compile separately)
 * nvcc -std=c++11 -c ./physlib/boundary.cu -o boundary.o
 * 
 */
#ifndef __BOUNDARY_H__
#define __BOUNDARY_H__

#include "../physlib/R2grid.h"       // Grid2d
#include "../physlib/dev_R2grid.h"       // Dev_Grid2d

/* --------------------------------------------------------------- */
/* Setting the boundary conditions at the boundary strip.      	   */
// on the host CPU memory
/* --------------------------------------------------------------- */
__host__ void set_BConditions_host( Grid2d & grid2d) ; 

__host__ void set_lidcavity_BConditions_host( Grid2d & ) ; 

/* --------------------------------------------------------------- */
/* Setting the boundary conditions at the boundary strip.      	   */
// on the device GPU memory
/* --------------------------------------------------------------- */

void set_BConditions( Dev_Grid2d & dev_grid2d ) ;

__host__ void set_lidcavity_BConditions( Dev_Grid2d & );

////////////////////////////////////////////////////////////////////////

__host__ void set_horiz_press_BCs( Dev_Grid2d & ) ;

////////////////////////////////////////////////////////////////////////

__host__ void set_vert_press_BCs( Dev_Grid2d & ) ;



#endif // __BOUNDARY_H__
