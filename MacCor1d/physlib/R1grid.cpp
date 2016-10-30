/* R1grid.cpp
 * R1 under discretization (discretize functor) to a grid
 * Ernest Yeung  ernestyalumni@gmail.com
 * 20161026
 * */
#include "R1grid.h"

Grid1d :: Grid1d(std::array<int,1> Ld_in, std::array<float,1> ld_in)
	: Ld(Ld_in), ld(ld_in)
{
	hd = { ld[0]/ Ld[0] }; 
	
	// Put in the desired physical objects that live on top of manifold here.
	// this has to be a pointer to a float or "old-school" float array 
	// in order to be cuda copied back to the host
	f = new float[ this->NFLAT() ];
}

std::array<float,1> Grid1d :: gridpt_to_space(std::array<int,1> index) {
	std::array<float,1> Xi { index[0]*hd[0] } ;
	return Xi;
}

int Grid1d :: NFLAT() {
	return Ld[0] ;
}

// destroy the physical quantities that sit on the R1 manifold
Grid1d::~Grid1d() {
	delete[] f ;
}

