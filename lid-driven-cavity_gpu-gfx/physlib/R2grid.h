/* R2grid.h
 * R2 under discretization (discretize functor) to a grid
 * Ernest Yeung  ernestyalumni@gmail.com
 * 20160701
 */
#ifndef __R2GRID_H__
#define __R2GRID_H__

#include <array> // std::array
#include <cmath> // expf

class Grid2d
{
	public : 
		std::array<int,2> Ld;
		std::array<float,2> ld;
		std::array<float,2> hd;
		
		// arrays for pressure and velocity
		float* F;
		float* G;
		float* u;
		float* v;

		// arrays for pressure
		float* pres_red;
		float* pres_black;
		
		Grid2d(std::array<int,2> Ld_in, std::array<float,2> ld_in);
		
		std::array<float,2> gridpt_to_space(std::array<int,2> );
		
		int NFLAT();
		int staggered_NFLAT();
		
		int flatten(const int i_x, const int i_y ) ;
		int staggered_flatten(const int i_x, const int i_y);
	
	
		~Grid2d();	
};


#endif // __R2GRID_H__
