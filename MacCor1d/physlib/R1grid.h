/* R1grid.h
 * R1 under discretization (discretize functor) to a grid
 * Ernest Yeung  ernestyalumni@gmail.com
 * 20161026
 * */
#ifndef __R1GRID_H__
#define __R1GRID_H__

#include <array> // std::array

class Grid1d
{
	public :
		std::array<int,1> Ld;
		std::array<float,1> ld;
		std::array<float,1> hd;

		// Put in the desired physical objects that live on top of manifold here.
		// this has to be a pointer to a float or "old-school" float array 
		// in order to be cuda copied back to the host
		float* f; // represents dynamic scalar quantity, u in Cebeci, Shao, Kafyeke, Laurendeau
		
		Grid1d(std::array<int,1> Ld_in, std::array<float,1> ld_in);
		
		std::array<float,1> gridpt_to_space(std::array<int,1> );
		
		int NFLAT();
		
		// destroy the physical quantities that sit on the R1 manifold
		~Grid1d();
};

#endif // __R1GRID_H__
