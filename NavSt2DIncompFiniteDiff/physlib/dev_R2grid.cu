/* dev_R2grid.cu
 * R3 under discretization (discretize functor) to a (staggered) grid
 * Ernest Yeung  ernestyalumni@gmail.com
 * 2016115
 * 
 * compilation tip: (compile separately)
 * nvcc -std=c++11 -c ./physlib/dev_R2grid.cu -o dev_R2grid.o
 * 
 */
#include "dev_R2grid.h"

//__constant__ int dev_Ld[2];

// constructor
__host__ Dev_Grid2d::Dev_Grid2d( dim3 Ld_in) : Ld(Ld_in)
{
	staggered_Ld.x  = Ld.x+2;
	staggered_Ld.y  = Ld.y+2;

	thrust::device_vector<float> temp_p( this->staggered_SIZE(), 0.0 );
	p = temp_p;
	p_temp = temp_p;
	F = temp_p;
	G = temp_p;
	RHS= temp_p;
	
	u = temp_p;
	v = temp_p;


	p_arr = thrust::raw_pointer_cast( (this->p).data() );	
	p_temp_arr = thrust::raw_pointer_cast( (this->p_temp).data() );	
	F_arr = thrust::raw_pointer_cast( (this->F).data() );	
	G_arr = thrust::raw_pointer_cast( (this->G).data() );	
	RHS_arr = thrust::raw_pointer_cast( (this->RHS).data() );	

	u_arr = thrust::raw_pointer_cast( (this->u).data() );	
	v_arr = thrust::raw_pointer_cast( (this->v).data() );	

	
}

// destructor

__host__ Dev_Grid2d::~Dev_Grid2d() {

}


__host__ int Dev_Grid2d :: NFLAT() {
	return Ld.x*Ld.y;
}	

__host__ int Dev_Grid2d :: staggered_SIZE() {
	return (staggered_Ld.x)*(staggered_Ld.y);
}	

__host__ int Dev_Grid2d :: flatten(const int i_x, const int i_y ) {
	return i_x+i_y*Ld.x  ;
}

__host__ int Dev_Grid2d :: staggered_flatten(const int i_x, const int i_y ) {
	return i_x+i_y*(staggered_Ld.x)  ;
}

