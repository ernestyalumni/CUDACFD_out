/* R3grid.h
 * R3 under discretization (discretize functor) to a grid
 * Ernest Yeung  ernestyalumni@gmail.com
 * 20160701
 */
#ifndef __R3GRID_H__
#define __R3GRID_H__

class Grid3d
{
	public : 
		const int   N_is[3];
		const float h_is[3];
		
		Grid3d(const int N_x = 2, const int N_y = 2, const int N_z = 2, 
				const float h_x = 0.1f, const float h_y = 0.1f, const float h_z = 0.1f);

		
		float l_i(int i) ;
		int* is_grid_pt(int* I) ;
		void gridpt_to_space(int* Index, float* Xi) ;
		
		int NFLAT();
		
		int flatten(const int i_x, const int i_y, const int i_z) ;
	
//		~Grid3d();	
};

#endif // __R3GRID_H__
