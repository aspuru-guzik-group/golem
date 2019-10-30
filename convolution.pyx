#!/usr/bin/env python 

import  cython 
cimport cython

import  numpy as np 
cimport numpy as np 

from libc.math cimport exp, sqrt, erf
from libc.math cimport isnan

#=========================================================================

from scipy import stats

#=========================================================================

# example for memview
# cdef double [:, :, :] tensor = np_tensor

cdef double gauss_cdf(double x, double loc, double scale):
	return (1. + erf( (x - loc) / (1.4142135623730951 * scale) )) * 0.5

cdef double uniform_cdf(double x, double loc, double scale):
	cdef double a = loc - 0.5 * scale
	cdef double b = loc + 0.5 * scale 
	if x < a:
		return 0.
	elif x > b:
		return 1.
	else:
		return (x - a) / (b - a)

#=========================================================================

cdef class Convolution:

	
	cdef np.ndarray X
	cdef np.ndarray dists
	cdef double     beta
	
	cdef np.ndarray preds
	cdef np.ndarray bounds

	cdef np.ndarray probabilities

	cdef int num_samples, num_dims, num_tiles


	def __init__(self, X, preds, bounds, distributions, beta):
#		self.X             = X
#		self.bboxes        = bboxes
#		self.distributions = distributions
#		self.beta          = beta

#		print('INITIALIZING')
	
		self.X    = X
		self.beta = beta
		self.dists = distributions
		self.preds = preds
		self.bounds = bounds

		print('0')

		self.num_samples = self.X.shape[0]
		self.num_dims    = self.X.shape[1]
		self.num_tiles   = self.preds.shape[0]
	
		print('1') 

		self.probabilities = np.zeros(shape = (self.num_dims, self.num_tiles))

		print('2')


		print('INITIALIZED')


	cdef double [:] _convolute(self):

#		print('convolute')

		cdef double [:, :] X_mem     = self.X
		cdef double [:, :] dists_mem = self.dists
		
		cdef double [:, :] probs_mem = self.probabilities

		cdef double [:] preds_mem        = self.preds
		cdef double [:, :, :] bounds_mem = self.bounds

		cdef double xi

		cdef double [:] y_reweighted = np.empty(self.num_samples)
		cdef double [:] y_reweighted_squared = np.empty(self.num_samples)
#		y_reweighted = []
#		y_reweighted_squared = []

		cdef double joint_prob
		cdef double [:] joint_probs = np.empty(self.num_samples)
		cdef double [:] newy = np.empty(self.num_samples)
		
		cdef double yi_reweighted = 0.
		cdef double yi_reweighted_squared = 0.


		cdef double total_prob = 0.

		# for each observation
		for num_sample in range(self.num_samples):

#			print(num_sample)
#		for x in self.X:

			# store probability in all dimensions, which later will be multiplied together
#			probabilities = np.zeros(shape=(np.shape(self.X)[1], len(self.bboxes)))
			# predicted response values for all tiles
#			y_pred = np.array([bbox['value'] for bbox in self.bboxes])

			# ------------------------------
			# go through all dimensions of x
			# ------------------------------
			for num_dim in range(self.num_dims):

				xi = X_mem[num_sample, num_dim]

				dist_type  = dists_mem[num_dim, 0]
				dist_param = dists_mem[num_dim, 1]
				if dist_type == 0.:
					# gauss
					
					for num_tile in range(self.num_tiles):
						low, high = bounds_mem[num_tile, num_dim, 0], bounds_mem[num_tile, num_dim, 1]
						if isnan(low):
							prob = gauss_cdf(high, xi, dist_param)
						elif isnan(high):
							prob = 1.0 - gauss_cdf(low, xi,dist_param)
						else:
							prob = gauss_cdf(high, xi, dist_param) - gauss_cdf(low, xi, dist_param)

						if isnan(prob):
							print('PROB', prob)	

						probs_mem[num_dim, num_tile] = prob						

				elif dist_type == 1.:
					# uniform

					for num_tile in range(self.num_tiles):
						low, high = bounds_mem[num_tile, num_dim, 0], bounds_mem[num_tile, num_dim, 1]
						if isnan(low):
							prob = uniform_cdf(high, xi, dist_param)
						elif isnan(high):
							prob = 1.0 - uniform_cdf(low, xi, dist_param)
						else:
							prob = uniform_cdf(high, xi, dist_param) - uniform_cdf(low, xi, dist_param)
			
						probs_mem[num_dim, num_tile] = prob						

			for num_tile in range(self.num_tiles):
				joint_prob = 1.
				for num_dim in range(self.num_dims):
					joint_prob *= probs_mem[num_dim, num_tile]
				joint_probs[num_tile] = joint_prob

			
#			total_prob = 0.
#			for num_tile in range(self.num_tiles):
#				total_prob += joint_probs[num_tile]
#			print('TOTAL PROB', total_prob)
#			assert( np.abs(1 - total_prob)  < 1e-4 )			


			yi_reweighted = 0.
			yi_reweighted_squared = 0.
			for num_tile in range(self.num_tiles):
				yi_reweighted += joint_probs[num_tile] * preds_mem[num_tile]
				yi_reweighted_squared += joint_probs[num_tile] * preds_mem[num_tile] * preds_mem[num_tile]


			y_reweighted[num_sample] = yi_reweighted
			y_reweighted_squared[num_sample] = yi_reweighted_squared

#			yi_reweighted = np.dot(joint_probabilities, y_pred)
#			yi_reweighted_squared = np.dot(joint_probabilities, y_pred ** 2)

			# append reweighted y_i values to the list of all reweighted y
#			y_reweighted.append(yi_reweighted)
#			y_reweighted_squared.append(yi_reweighted_squared)

		# now we have all the reweighted y values - E[f(x)] and E[f(x)^2]
		# if beta > 0 we are penalising also the variance
		cdef double variance

		for num_sample in range(self.num_samples):
			variance = y_reweighted_squared[num_sample] - y_reweighted[num_sample]**2
			newy[num_sample] = y_reweighted[num_sample] - self.beta * sqrt(variance)		

#		if self.beta > 0:
#			variance = np.array(y_reweighted_squared) - np.array(y_reweighted) ** 2
#			newy = np.array(y_reweighted) - self.beta * np.sqrt(variance)
#		else:
#			newy = np.array(y_reweighted)

		print("done")
		return newy

	cpdef convolute(self):
		return np.array(self._convolute())

