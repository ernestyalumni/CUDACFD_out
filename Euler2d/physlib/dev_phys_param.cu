/* dev_phys_param.cu
 * physical parameters on the device
 * Ernest Yeung  ernestyalumni@gmail.com
 * 20160804
 */
#include "dev_phys_param.h" 
 
__constant__ float dev_Deltat[1]; // Deltat

__constant__ float dev_gas_params[3] ; // dev_gas_params[0] = heat capacity ratio

/*
__device__ float pressure( float energy, float rho, float2 u ) {
	float pressure_val {0.f};
	float usqr { u.x*u.x+u.y+u.y };
	pressure_val = (dev_gas_params[0] - 1.f)*(energy-0.5f*rho*usqr) ;
	return pressure_val;
}
* */

__device__ float pressure( float energy, float rho, float usq ) {
	float pressure_val {0.f};
	
	pressure_val = (dev_gas_params[0] - 1.f)*(energy-0.5f*rho*usq) ;
	return pressure_val;
}
