/* boundary.cu
 * fix boundary conditions at the "boundary strip" cf. Griebel, Dornseifer, & Neunhoeffer (1997)
 * \author Ernest Yeung  
 * \email ernestyalumni@gmail.com
 * \date 20161207
 * 
 * compilation tip: (compile separately)
 * nvcc -std=c++11 -c ./physlib/boundary.cu -o boundary.o
 * 
 */
#include "boundary.h"

/* --------------------------------------------------------------- */
/* Setting the boundary conditions at the boundary strip.      	   */
// on the host CPU memory
/* --------------------------------------------------------------- */

__host__ void set_BConditions_host( Grid2d & grid2d) {
	const int imax { grid2d.Ld[0] } ;
	const int jmax { grid2d.Ld[1] } ;

	/* western and eastern boundary, i.e. x=0, and x=L_x+1 i.e. x=imax+1 */
	
	/* no slip */
	for (auto j = 0; j<= (jmax+1);j++) { 
		/* western boundary */
		grid2d.u[grid2d.staggered_flatten( 0,j) ] = 0.0;  	   /* u = 0  	*/
		grid2d.v[grid2d.staggered_flatten( 0,j) ] = 
			(-1.0) * grid2d.v[grid2d.staggered_flatten( 1,j) ]; /* v = 0 at the boundary by averaging */

		/* eastern boundary */
		grid2d.u[grid2d.staggered_flatten( imax,j) ] = 0.0; 
		grid2d.v[grid2d.staggered_flatten( imax+1,j) ] = 
			(-1.0) * grid2d.v[grid2d.staggered_flatten( imax,j) ];
	}
	
	/* northern and southern boundary, i.e. y=0 and y=L_y+1, i.e. y=jmax+1 */ 

	/* no slip */
	for (auto i = 0; i<= (imax+1);i++) {
		/* northern boundary */
		grid2d.v[grid2d.staggered_flatten( i,jmax)] = 0.0;  /* v = 0  */
		grid2d.u[grid2d.staggered_flatten( i,jmax+1)] = 
			(-1.0)*grid2d.u[grid2d.staggered_flatten( i,jmax)]; /* u = 0 at the boundary by averaging */
	
		/* southern boundary */
		grid2d.v[grid2d.staggered_flatten( i,0) ] = 0.0;     /* v = 0 */
		grid2d.u[grid2d.staggered_flatten( i,0) ] = 
			(-1.0)*grid2d.u[grid2d.staggered_flatten( i,1) ]; /* u = 0 at the boundary by averaging */
		
	}
	return;	
} // end set_BConditions_host

/* -------------------------------------------------------------- */
/* Driven Cavity: u^x = 1.0 at the upper boundary                 */ 
// on the host CPU memory
/* -------------------------------------------------------------- */
// we set \overline{u} = 2.0, but this case be changed, i.e. how fast the lid moves to the right, or positive +x direction
__host__ void set_lidcavity_BConditions_host( Grid2d & grid2d) {
	const int imax { grid2d.Ld[0] } ;
	const int jmax { grid2d.Ld[1] } ;

	
	for (auto i = 1; i< imax+1;i++) {
		/* northern boundary */
		grid2d.u[grid2d.staggered_flatten( i,jmax+1)] = 
			2.0 - grid2d.u[grid2d.staggered_flatten( i, jmax)] ; }

	return;
}

/* --------------------------------------------------------------- */
/* Setting the boundary conditions at the boundary strip.      	   */
// on the device GPU memory
/* --------------------------------------------------------------- */

void set_BConditions( Dev_Grid2d & dev_grid2d ) {
	const int imax = dev_grid2d.Ld.x;
	const int jmax = dev_grid2d.Ld.y;
	
	/* western and eastern boundary, i.e. x=0, and x=L_x+1 i.e. x=imax+1 */

	/* no slip */
	for (auto j = 0; j<= (jmax+1);j++) { 
		/* western boundary */
		dev_grid2d.u[dev_grid2d.staggered_flatten( 0,j) ] = 0.0;  	   /* u = 0  	*/
		dev_grid2d.v[dev_grid2d.staggered_flatten( 0,j) ] = 
			(-1.0) * dev_grid2d.v[dev_grid2d.staggered_flatten( 1,j) ]; /* v = 0 at the boundary by averaging */

		/* eastern boundary */
		dev_grid2d.u[dev_grid2d.staggered_flatten( imax,j) ] = 0.0; 
		dev_grid2d.v[dev_grid2d.staggered_flatten( imax+1,j) ] = 
			(-1.0) * dev_grid2d.v[dev_grid2d.staggered_flatten( imax,j) ];
	}
	
	/* northern and southern boundary, i.e. y=0 and y=L_y+1, i.e. y=jmax+1 */ 

	/* no slip */
	for (auto i = 0; i<= (imax+1);i++) {
		/* northern boundary */
		dev_grid2d.v[dev_grid2d.staggered_flatten( i,jmax)] = 0.0;  /* v = 0  */
		dev_grid2d.u[dev_grid2d.staggered_flatten( i,jmax+1)] = 
			(-1.0)*dev_grid2d.u[dev_grid2d.staggered_flatten( i,jmax)]; /* u = 0 at the boundary by averaging */
	
		/* southern boundary */
		dev_grid2d.v[dev_grid2d.staggered_flatten( i,0) ] = 0.0;     /* v = 0 */
		dev_grid2d.u[dev_grid2d.staggered_flatten( i,0) ] = 
			(-1.0)*dev_grid2d.u[dev_grid2d.staggered_flatten( i,1) ]; /* u = 0 at the boundary by averaging */
		
	}
	return;	
} // end set_BConditions


/* -------------------------------------------------------------- */
/* Setting specific boundary conditions, depending on "problem"   */
/* -------------------------------------------------------------- */

/* -------------------------------------------------------------- */
/* Driven Cavity: u^x = 1.0 at the upper boundary                 */ 
// on the device GPU memory
/* -------------------------------------------------------------- */
// we set \overline{u} = 2.0, but this case be changed, i.e. how fast the lid moves to the right, or positive +x direction

__host__ void set_lidcavity_BConditions( Dev_Grid2d & dev_grid2d) {
	const int imax = dev_grid2d.Ld.x  ;
	const int jmax = dev_grid2d.Ld.y  ;

	for (auto i = 0; i< imax+1;i++) {
		/* northern boundary */
		dev_grid2d.u[dev_grid2d.staggered_flatten( i,jmax+1)] = 
			2.0 - dev_grid2d.u[dev_grid2d.staggered_flatten( i, jmax)] ; }

	return;
}


////////////////////////////////////////////////////////////////////////
__host__ void set_horiz_press_BCs( Dev_Grid2d & dev_grid2d) {
	const int imax = dev_grid2d.Ld.x; 
	const int jmax = dev_grid2d.Ld.y; 
	
	for (auto i = 1; i<= imax; i++) {
	/* copy values at external boundary */
	/* -------------------------------- */
	// p_i,0 = p_i,1
		dev_grid2d.p[ dev_grid2d.staggered_flatten( i, 0)] = 
			dev_grid2d.p[ dev_grid2d.staggered_flatten( i, 1)] ; 
	// p_i,jmax+1 = p_i,jmax
		dev_grid2d.p[ dev_grid2d.staggered_flatten( i, jmax+1)] = 
			dev_grid2d.p[ dev_grid2d.staggered_flatten( i, jmax)] ; 
	}
}

////////////////////////////////////////////////////////////////////////
__host__ void set_vert_press_BCs( Dev_Grid2d & dev_grid2d) {
	const int imax = dev_grid2d.Ld.x; 
	const int jmax = dev_grid2d.Ld.y; 

	for (auto j = 1; j<= jmax; j++) {
	/* copy values at external boundary */
	/* -------------------------------- */
	// p_0,j = p_1,j
		dev_grid2d.p[ dev_grid2d.staggered_flatten( 0, j)] = 
			dev_grid2d.p[ dev_grid2d.staggered_flatten( 1, j)] ; 
	// p_imax+1,j = p_imax,j
		dev_grid2d.p[ dev_grid2d.staggered_flatten( imax+1, j)] = 
			dev_grid2d.p[ dev_grid2d.staggered_flatten( imax, j)] ; 
	}
}

