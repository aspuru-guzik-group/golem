#!/usr/bin/env python 

import  cython 
cimport cython

import  numpy as np 
cimport numpy as np 

from libc.math cimport exp

#=========================================================================

from scipy import stats

#=========================================================================

class Convolution:

	

	def __init__(self, X, bboxes, distributions, beta):
		self.X             = X
		self.bboxes        = bboxes
		self.distributions = distributions
		self.beta          = beta


	def convolute(self):

		print('Performing convolution...')

		y_reweighted = []
		y_reweighted_squared = []

		# for each observation
		for x in self.X:

			# store probability in all dimensions, which later will be multiplied together
			probabilities = np.zeros(shape=(np.shape(self.X)[1], len(self.bboxes)))
			# predicted response values for all tiles
			y_pred = np.array([bbox['value'] for bbox in self.bboxes])

			# ------------------------------
			# go through all dimensions of x
			# ------------------------------
			for i, xi in enumerate(x):
				# define distribution centered at the sample
				distribution_i = self.distributions[i]['distribution']
				params = self.distributions[i]['params']

				# fix location of uniform
				if distribution_i == stats.uniform:
					d = distribution_i(**params, loc=xi-params['scale']/2)
				else:
					d = distribution_i(**params, loc=xi)

				# determine probability/weight of each tile in the rectangular tesselation
				for j, bbox in enumerate(self.bboxes):

					high = bbox[i]['high']
					low = bbox[i]['low']

					if low is None:
						probability = d.cdf(high)
					elif high is None:
						probability = 1.0 - d.cdf(low)
					else:
						probability = d.cdf(high) - d.cdf(low)

					# store
					probabilities[i, j] = probability

			# now multiply the marginal probabilities together to get the joint
			joint_probabilities = np.prod(probabilities, axis=0)

			# check we have gone through the whole distribution (i.e. area sums up to 1)
			assert np.isclose(np.sum(joint_probabilities), 1.0)

			# multiply the probabilities by the value of the tiles to get the reweighted
			# value for y (response) corresponding to the x just parsed
			assert len(y_pred) == len(joint_probabilities)
			yi_reweighted = np.dot(joint_probabilities, y_pred)
			yi_reweighted_squared = np.dot(joint_probabilities, y_pred ** 2)

			# append reweighted y_i values to the list of all reweighted y
			y_reweighted.append(yi_reweighted)
			y_reweighted_squared.append(yi_reweighted_squared)

		# now we have all the reweighted y values - E[f(x)] and E[f(x)^2]
		# if beta > 0 we are penalising also the variance
		if self.beta > 0:
			variance = np.array(y_reweighted_squared) - np.array(y_reweighted) ** 2
			newy = np.array(y_reweighted) - self.beta * np.sqrt(variance)
		else:
			newy = np.array(y_reweighted)

		print("done")
		return newy
