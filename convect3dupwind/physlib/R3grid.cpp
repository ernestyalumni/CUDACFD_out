/* R3grid.cpp
 * R3 under discretization (discretize functor) to a grid
 * Ernest Yeung  ernestyalumni@gmail.com
 * 20160630
 */
#include "R3grid.h"


Grid3d :: Grid3d(const int N_x, const int N_y, const int N_z, 
				const float h_x, const float h_y, const float h_z)
	: N_is {N_x,N_y,N_z}, h_is {h_x,h_y,h_z} 	
{}	


	
float Grid3d :: l_i(int i) {
		return N_is[i]*h_is[i];
	}

int* Grid3d :: is_grid_pt(int* I) {
	if (I[0] >= N_is[0] || I[1] >= N_is[1] || I[2] >= N_is[2]) {
			return nullptr; }
		return I;
	}
	
void Grid3d :: gridpt_to_space(int* Index, float* Xi) {
	auto checkingrid = this->is_grid_pt(Index);
	if (checkingrid == nullptr) {
		return; }
		
	Xi[0] = Index[0]*h_is[0]; 
	Xi[1] = Index[1]*h_is[1]; 
	Xi[2] = Index[2]*h_is[2];
	}

int Grid3d :: NFLAT() {
	int N {1};
		for (auto N_i : N_is) {
			N *= N_i; }
		return N;
	}

int Grid3d :: flatten(const int i_x, const int i_y, const int i_z) {
	return i_x+i_y*N_is[0]+i_z*N_is[0]*N_is[1];	
}

//Grid3d::~Grid3d() {}

