/* dev_phys_param.h
 * physical parameters on the device
 * Ernest Yeung  ernestyalumni@gmail.com
 * 20160804
 */
#ifndef __DEV_PHYS_PARAM_H__
#define __DEV_PHYS_PARAM_H__

extern __constant__ float dev_Deltat[1]; // Deltat

extern __constant__ float dev_gas_params[3] ; // dev_gas_params[0] = heat capacity ratio, dev_gas_params[1] = C_V, dev_gas_params[2] = M

//__device__ float pressure( float energy, float rho, float2 u ) ;

__device__ float pressure( float energy, float rho, float usq) ;


#endif // __DEV_PHYS_PARAM_H__
