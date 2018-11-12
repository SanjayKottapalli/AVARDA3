# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 10:33:36 2017

@authors: Sanjay Kottapalli, Tiezheng Yuan
"""
import params
import phip_analysis
import os
import timeit

if __name__=="__main__":
	''' Main method for executing the AVARDA pipeline.
	All necessary input goes into the variables file.
	'''
	time1 = timeit.default_timer()
	path = os.getcwd().replace('\\' , '/').replace('bin','input') + '/'
	var_file = path+'variables_virus.txt'
	print("Variables file: " + var_file)
	par = params.file_IO(var_file, '=').file_to_dict()
	par = params.param_dict(par).adjust_par()
	print("Alignment file: " + par['file_aln'])
	print("Z-score file: " + par['zscore_file'])
	print("---------------------------------------------------------------")
	phip = phip_analysis.phip(par)
	time2 = timeit.default_timer()
	print("Total time: " + str(time2-time1))

# End
