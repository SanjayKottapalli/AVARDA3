# -*- coding: utf-8 -*-
"""
Created on Mon May 28 21:22:29 2018

@author: Sanjay
"""
import params
from flex_array import gen_ind_hits
import flex_array
import pandas as pd
import numpy as np
import timeit
from itertools import combinations_with_replacement as combo
from joblib import Parallel, delayed
#import sys

def dependent_peptides(): #function to get dependency dict
	dependent_pep = {}
	file_dependent = '../ref_seq/'+'virus_dependent_peptides_trunc_v2_20180522.csv'
	f = open(file_dependent, 'r')
	for line in f:
		line = line.strip().split(',')
		pep1 = str(line[0])
		pep2 = str(line[1])
		if pep1 in dependent_pep:
			dependent_pep[pep1].append(pep2)
		else:
			dependent_pep[pep1] = [pep2]
	f.close()
	return dependent_pep

def calc_pair(pair, virus_aln, virus_xr, dep_pep, first_round_prob, peptide_lib, iter_number, total_iter): #function to calculate unique probabilities
	# pair is (virus_i, virus_j); virus_aln is all evidence peptides, virus_xr is all x-reactive peptides
	i = pair[0]; j = pair[1]; d1 = {}; d2 = {};
	# define sets of total evidence peptides
	i_index = set(virus_aln[i].index); j_index = set(virus_aln[j].index)
	# as above, but for xr peptides
	xri_index = set(virus_xr[i].index); xrj_index = set(virus_xr[j].index)
	# define set of peptides which are xr to both viruses
	xr_shared_index = set(xri_index).intersection(xrj_index)
	
	i_all = i_index.union(xri_index); j_all = j_index.union(xrj_index) # define sets of all peptides aligning to i or j
	shared_all = i_all.intersection(j_all) # all shared, including xr and evidence
	i_shared = i_index.intersection(shared_all); j_shared = j_index.intersection(shared_all)
    # to find all shared peptides that are evidence for at least one of the viruses, subtract peptides that are xr to both
	shared_evidence = shared_all - xr_shared_index
	
	# don't perform unnecessary calculations if there are no shared evidence peptides (equivalent to ranking prob)
	if len(shared_evidence) == 0:
		d1[(i,j)] = first_round_prob[j]
		d1[(j,i)] = first_round_prob[i]
		#d2[(i,j)] = 0.0
		#d2[(j,i)] = 0.0
		print('shared 0')
	else:
		unique_j = j_index - shared_all; unique_j = virus_aln[j].loc[unique_j] #unique evidence for j
		unique_i = i_index - shared_all; unique_i = virus_aln[i].loc[unique_i] #unique evidence for i
#		unique_xri = xri_index - shared_all; unique_xrj = xrj_index - shared_all #unique xr
		
		# parallelized filtering of uniques
		#arrays = [unique_j, unique_i]
		#filter_unique_j, filter_unique_i = Parallel(n_jobs=-1)(delayed(gen_ind_hits)(array, dep_pep) for array in arrays)
		#del arrays # memory conservation
		# Numerator: filtered uniques
		# Denominator terms: library - shared peptides which are evidence for either or both viruses - unique xr to j - dependent unique evidence (thrown out)
		d1[(i,j)] = len(unique_j)/(len(peptide_lib)-len(i_shared))
		# above, but for virus i
		d1[(j,i)] = len(unique_i)/(len(peptide_lib)-len(j_shared))
		#d2[(i,j)] = len(filter_shared)/(len(peptide_lib)-len(unique_i)-len(unique_j)-(len(shared)-len(filter_shared)))
		#d2[(j,i)] = d2[(i,j)]
	print("Pair "+str(iter_number)+" of "+str(total_iter)+" completed.", flush=True)
	
	return d1, d2

if __name__ == "__main__": # main method
	# define the peptide library
	initial_time = timeit.default_timer()
	viral_peptidome = open('../ref_seq/'+'20180211_VirScan_annot.txt', 'r')
	peptide_lib = []
	next(viral_peptidome)
	for line in viral_peptidome:
		items = line.split('\t')
		peptide_lib.append(str(items[0]))
	viral_peptidome.close()
	peptide_lib = peptide_lib[:-1]
	
	############ read aln matrix, populate with ppos values
	f = params.file_IO('../ref_seq/pep_against_human_viruses.tblastn.species.noseg.WS3.max_target_seqs100000.180625 (1).m8', '\t')
	orig_aln = f.flat_file_to_df([0,1,15])
	f = params.file_IO('../ref_seq/new_pep_against_human_viruses.tblastn.species.noseg.WS3.max_target_seqs100000.180625 (1).m8', '\t')
	new_aln = f.flat_file_to_df([0,1,15])
	
	orig_aln.index = [i.split('_')[1] for i in orig_aln.index]
	aln_df = pd.concat([orig_aln, new_aln])
	aln_df.fillna(0, inplace=True)
	#################
	
	# filter out subset viruses and those that have less than 10 peptides >=80% pos
	aln_df = aln_df.reindex(peptide_lib).fillna(0)
	print("filled NA", flush=True)
	binary_b = aln_df[aln_df>=80].fillna(0)
	binary_b = pd.DataFrame(index=binary_b.index, columns=binary_b.columns, data=binary_b.values, dtype=bool)
	binary_b = flex_array.array(binary_b).filter_aln(ref_seq='../ref_seq/')
	aln_df = aln_df.loc[:,binary_b.columns] #use the resulting columns from filtering the binary matrix
	binary_b = pd.DataFrame(index=aln_df.index, columns=aln_df.columns, data=aln_df.values, dtype=bool)
	print("filtered", flush=True)
	
	# dictionary of list of alignments for each virus. keys: virus strings, values: Series objects containing only nonzero alignments
	virus_aln = {i:aln_df.loc[binary_b[i], i] for i in binary_b.columns} 
	# above, but with only the xr peptides
	virus_xr = {i:virus_aln[i][virus_aln[i]<80] for i in binary_b.columns} 
	# only 'evidence' peptides (reassigned variable name)
	virus_aln = {i:virus_aln[i][virus_aln[i]>=80] for i in binary_b.columns}
	dep_pep = dependent_peptides()
	
	# Calculating ranking probabilities for each virus
	virus_tot_filter = [virus_aln[i] for i in virus_aln] # dictionary of filtered evidence peptides
	virus_sums = pd.Series(index=binary_b.columns, data=[len(i) for i in virus_tot_filter]) # filtered evidence numbers for each virus
	
	virus_sums = virus_sums[virus_sums>10]
	virus_aln = {i:virus_aln[i] for i in virus_sums.index}
	virus_xr = {i:virus_xr[i] for i in virus_sums.index}

	first_round_prob = pd.Series(index=virus_sums.index)
	viruses = list(virus_sums.index)
	for i in first_round_prob.index:
		print("Virus " + str(viruses.index(i)))
		# Numerator: number of filtered evidence peptides for each virus
		# Denominator terms: library - cross reactives to virus i - dependent evidence peptides (thrown out)
		#first_round_prob[i] = virus_sums[i]/(len(peptide_lib)-len(virus_xr[i])-(len(virus_aln[i])-virus_sums[i]))
		first_round_prob[i] = virus_sums[i]/(len(peptide_lib)-(len(virus_aln[i])-virus_sums[i]))
	first_round_prob.to_csv("total_probabilities_20181110.csv", header=False, index=True) #print to file
	print("First probability file generated.", flush=True)
	
	pair_time = timeit.default_timer()
	virus_pairs = list(combo(viruses, 2)) # list of all unique pairs of viruses under consideration (10 viruses for testing)
	
	print("Starting unique probability calculations.", flush=True)
	print("Progress: ", flush=True)
	# start parallel loop for calculating unique probabilities (see function)s
	results = Parallel(n_jobs=-1)(delayed(calc_pair)(pair, virus_aln, virus_xr, dep_pep, first_round_prob, peptide_lib, it_num, len(virus_pairs)) for pair,it_num  in zip(virus_pairs, list(range(1,len(virus_pairs)+1))))
	m1, m2 = zip(*results)
	del results
	m1 = {k:v for d in m1 for k, v in d.items()} # housekeeping for unpacking results
	#m2 = {k:v for d in m2 for k, v in d.items()}
	print("Done with calculations.", flush=True)
	time_end = timeit.default_timer()
	print('Time to finish pairs: '+str(time_end-pair_time))
	
	# operations to unpack the parallelized results into a viruses x viruses 2D matrix
	second_round_prob = pd.Series(m1)
	second_round_prob = second_round_prob.unstack().reindex()
	
	second_round_prob.to_csv("unique_probabilities_20181110.csv", header=True, index=True) #print to file


