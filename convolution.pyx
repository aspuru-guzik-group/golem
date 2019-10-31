#!/usr/bin/env python 

import  cython 
cimport cython

import  numpy as np 
cimport numpy as np 

from libc.math cimport exp, sqrt, erf
from libc.math cimport isnan

#=========================================================================

# example for memview
# cdef double [:, :, :] tensor = np_tensor

@cython.cdivision(True)
cdef double gauss_cdf(double x, double loc, double scale):
	cdef double arg
	arg = (x - loc) / (1.4142135623730951 * scale)
	if arg > 3.:
		return 1.
	elif arg < -3.:
		return 0.
	return (1. + erf( arg )) * 0.5

@cython.cdivision(True)
cdef double uniform_cdf(double x, double loc, double scale):
	cdef double a = loc - 0.5 * scale
	cdef double b = loc + 0.5 * scale 
	if x < a:
		return 0.
	elif x > b:
		return 1.
	return (x - a) / (b - a)

#=========================================================================

cdef class Convolution:

	
	cdef np.ndarray np_X
	cdef np.ndarray np_dists
	cdef double     np_beta
	
	cdef np.ndarray np_preds
	cdef np.ndarray np_bounds

	cdef int num_samples, num_dims, num_tiles


	def __init__(self, X, preds, bounds, dists, beta):
	
		self.np_X        = X
		self.np_beta     = beta
		self.np_dists    = dists
		self.np_preds    = preds
		self.np_bounds   = bounds

		self.num_samples = X.shape[0]
		self.num_dims    = X.shape[1]
		self.num_tiles   = preds.shape[0]



	@cython.boundscheck(False)
	cdef double [:] _convolute(self):

		cdef int num_dim, num_tile, num_sample
		
		cdef double dist_type, dist_param
		cdef double low, high

		cdef double xi
		cdef double joint_prob

		cdef double [:]       preds  = self.np_preds
		cdef double [:, :]    X      = self.np_X
		cdef double [:, :]    dists  = self.np_dists
		cdef double [:, :, :] bounds = self.np_bounds

		cdef double [:] newy = np.empty(self.num_samples)
		
		cdef double yi_reweighted
		cdef double yi_reweighted_squared 

		cdef double cache

		for num_sample in range(self.num_samples):

			yi_reweighted         = 0.
			yi_reweighted_squared = 0.

			for num_tile in range(self.num_tiles):

				joint_prob = 1.

				for num_dim in range(self.num_dims):

					xi         = X[num_sample, num_dim]
					dist_type  = dists[num_dim, 0]
					dist_param = dists[num_dim, 1]

					if dist_type == 0.:

						low  = bounds[num_tile, num_dim, 0]
						high = bounds[num_tile, num_dim, 1]
						if isnan(low):
							joint_prob *= gauss_cdf(high, xi, dist_param)
						elif isnan(high):
							joint_prob *= 1.0 - gauss_cdf(low, xi, dist_param)
						else:
							joint_prob *= gauss_cdf(high, xi, dist_param) - gauss_cdf(low, xi, dist_param)	

					elif dist_type == 1.:

						low  = bounds[num_tile, num_dim, 0]
						high = bounds[num_tile, num_dim, 1]
						if isnan(low):
							joint_prob *= uniform_cdf(high, xi, dist_param)
						elif isnan(high):
							joint_prob *= 1.0 - uniform_cdf(low, xi,dist_param)
						else:
							joint_prob *= uniform_cdf(high, xi, dist_param) - uniform_cdf(low, xi, dist_param)	
							
				cache                  = joint_prob * preds[num_tile]
				yi_reweighted         += cache
				yi_reweighted_squared += cache * preds[num_tile]

			newy[num_sample] = yi_reweighted - self.np_beta * sqrt(yi_reweighted_squared - yi_reweighted**2)		

		return newy
	



	@cython.boundscheck(False)
	cdef double [:] __OLD___convolute(self):

		cdef int num_dim, num_tile, num_sample
		
		cdef double dist_type, dist_param
		cdef double low, high

		cdef double xi
		cdef double total_prob
		cdef double variance

		cdef double [:]       preds  = self.np_preds
		cdef double [:, :]    X      = self.np_X
		cdef double [:, :]    dists  = self.np_dists
		cdef double [:, :, :] bounds = self.np_bounds

		cdef double [:, :] probs = np.empty((self.num_dims, self.num_tiles))

		cdef double [:] y_reweighted = np.empty(self.num_samples, dtype = np.float64)
		cdef double [:] y_reweighted_squared = np.empty(self.num_samples, dtype = np.float64)
#		y_reweighted = []
#		y_reweighted_squared = []

		cdef double joint_prob
		cdef double [:] joint_probs = np.empty(self.num_samples)
		cdef double [:] newy = np.empty(self.num_samples)
		
		cdef double yi_reweighted = 0.
		cdef double yi_reweighted_squared = 0.




		# for each observation
		for num_sample in range(self.num_samples):

			for num_dim in range(self.num_dims):

				xi = X[num_sample, num_dim]

				dist_type  = dists[num_dim, 0]
				dist_param = dists[num_dim, 1]
				if dist_type == 0.:
					# gauss
					
					for num_tile in range(self.num_tiles):
						low  = bounds[num_tile, num_dim, 0]
						high = bounds[num_tile, num_dim, 1]
						if isnan(low):
							probs[num_dim, num_tile] = gauss_cdf(high, xi, dist_param)
						elif isnan(high):
							probs[num_dim, num_tile] = 1.0 - gauss_cdf(low, xi,dist_param)
						else:
							probs[num_dim, num_tile] = gauss_cdf(high, xi, dist_param) - gauss_cdf(low, xi, dist_param)			

				elif dist_type == 1.:
					# uniform

					for num_tile in range(self.num_tiles):
						low  = bounds[num_tile, num_dim, 0]
						high = bounds[num_tile, num_dim, 1]
						if isnan(low):
							probs[num_dim, num_tile] = uniform_cdf(high, xi, dist_param)
						elif isnan(high):
							probs[num_dim, num_tile] = 1.0 - uniform_cdf(low, xi, dist_param)
						else:
							probs[num_dim, num_tile] = uniform_cdf(high, xi, dist_param) - uniform_cdf(low, xi, dist_param)			

			for num_tile in range(self.num_tiles):
				joint_prob = 1.
				for num_dim in range(self.num_dims):
					joint_prob *= probs[num_dim, num_tile]
				joint_probs[num_tile] = joint_prob
	


			yi_reweighted = 0.
			yi_reweighted_squared = 0.
			for num_tile in range(self.num_tiles):
				yi_reweighted += joint_probs[num_tile] * preds[num_tile]
				yi_reweighted_squared += joint_probs[num_tile] * preds[num_tile] * preds[num_tile]


			y_reweighted[num_sample] = yi_reweighted
			y_reweighted_squared[num_sample] = yi_reweighted_squared

		for num_sample in range(self.num_samples):
			variance = y_reweighted_squared[num_sample] - y_reweighted[num_sample]**2
			newy[num_sample] = y_reweighted[num_sample] - self.np_beta * sqrt(variance)		

		return newy


	cpdef convolute(self):
		return np.array(self._convolute())

