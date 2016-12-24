/* R3grid.cpp
 * R3 under discretization (discretize functor) to a grid
 * Ernest Yeung  ernestyalumni@gmail.com
 * 20160630
 */
#include "R2grid.h"


Grid2d :: Grid2d(std::array<int,2> Ld_in, std::array<float,2> ld_in)
	: Ld(Ld_in), ld(ld_in)
{
	hd = { ld[0]/Ld[0], ld[1]/Ld[1]  };
	
	F = new float[ this->staggered_NFLAT() ];
	G = new float[ this->staggered_NFLAT() ];
	u = new float[ this->staggered_NFLAT() ];
	v = new float[ this->staggered_NFLAT() ];

	for (auto i = 0; i < this->staggered_NFLAT(); ++i ) {
		F[i] = 0.0;
		G[i] = 0.0;
		u[i] = 0.0;
		v[i] = 0.0;
	}

	int size_pres = ((Ld[0] / 2) + 2) * ( Ld[1] + 2);
	pres_red   = new float[ size_pres ];
	pres_black = new float[ size_pres ];

	for (int i = 0; i < size_pres; ++i) {
		pres_red[i] = 0.0;
		pres_black[i] = 0.0 ;
	}

}

std::array<float,2> Grid2d :: gridpt_to_space(std::array<int,2> index) {
	std::array<float,2> Xi { index[0]*hd[0], index[1]*hd[1]  } ;
	return Xi;
}

int Grid2d :: NFLAT() {
	return Ld[0]*Ld[1] ;
}	

int Grid2d :: staggered_NFLAT() {
	return (Ld[0]+2)*(Ld[1]+2) ;
}	


int Grid2d :: flatten(const int i_x, const int i_y ) {
	return i_x+i_y*Ld[0]  ;
}

int Grid2d :: staggered_flatten(const int i_x, const int i_y ) {
	return i_x+i_y*(Ld[0]+2)  ;
}


Grid2d::~Grid2d() {
	delete[] F;
	delete[] G;
	delete[] u;
	delete[] v;

	delete[] pres_red;
	delete[] pres_black;
}


