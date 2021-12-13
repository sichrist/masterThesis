import numpy as np
from Boids import *
from Boids_smooth import *
from WeightedDistances import *



class Simulator(object):
	"""docstring for Simulator"""
	def __init__(self, data, SSF=False,ID=0):
		super(Simulator, self).__init__()
		# Simulate single Fish?
		self.SSF  = SSF
		# ID of Fish 2 simulate
		self.ID   = ID
		# Data of agents
		self.data = data


		
		