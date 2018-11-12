# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 10:47:13 2017

@authors: Sanjay Kottapalli, Tiezheng Yuan
"""
import os
import pandas as pd
#import numpy as np
from datetime import datetime as date
from itertools import combinations_with_replacement as combo
from joblib import Parallel, delayed
import flex_array

class file_IO:
	'''Class for parsing file input'''
	def __init__(self, infile, sep = '='):
		self.infile = infile
		self.sep = sep
		
	def file_to_dict(self):
		mydict = {}
		f = open(self.infile,'r')
		for line in f:
			line=line.strip()
			if 0 < line.find(self.sep) < len(line):
				key, value = line.split(self.sep)
				mydict[key] = value
		f.close()
		return mydict
	
	def dict_to_file(self, mydict):
		f = open(self.infile, 'w')
		for i in mydict:
			f.write(str(i)+self.sep+str(mydict[i])+'\n')
		f.close()
		return None
	
	def flat_file_to_df(self, df_index=[0,1,2]):
		aln_dict = {}
		#get index of rows, columns, and values
		row_index, col_index, value_index = df_index 
		#read file into nested dict
		f = open(self.infile, 'r')
		for line in f:
			line = line.strip()
			items = line.split(self.sep)
			row, col, value = items[row_index], items[col_index], items[value_index]
			if not col in aln_dict: aln_dict[col] = {}
			#assign value
			aln_dict[col][row] = float(value)
		f.close()
		#convert dict2 to df and fill NA with 0
		mydf = pd.DataFrame(aln_dict).fillna(0)
		return mydf
	
class param_dict:
	''''''
	def __init__(self, mydict):
		self.par = mydict
		
	def adjust_par(self):
		if 'organism' in self.par['file_aln']:
			self.par['organism'] = True
		else:
			self.par['organism'] = False
		self.par['dir_home'] = os.getcwd().replace('\\' , '/').replace('bin','')
		#if not self.par['dir_home'][-1] == '/':
		#	self.par['dir_home'] += '/'
		self.par['dir_input'] = self.par['dir_home'] + 'input/'
		self.par['dir_result'] = self.par['dir_home'] + 'results/'
		try:
			os.makedirs(self.par['dir_result'], exist_ok=False)
		except:
			pass
		self.par['dir_ref_seq'] = self.par['dir_home'] + 'ref_seq/'
		print("Home directory: " + self.par['dir_home'])
		print("Result directory: " + self.par['dir_result'])
		
		d = date.today()
		d = d.strftime("%d-%B-%Y_%H-%M")
		file_head = self.par['zscore_file'].split('.')[0]
		self.par['sub_dir'] = self.par['dir_result'] + file_head + ' ' + d + '/'
		os.makedirs(self.par['sub_dir'], exist_ok=True)
		
		self.par['graph_dir'] = self.par['sub_dir'] + 'sample_networks/'
		os.makedirs(self.par['graph_dir'], exist_ok=True)
		
		self.par['file_annotation'] = self.par['dir_ref_seq'] + self.par['file_annotation']
		self.par['file_aln'] = self.par['dir_ref_seq'] + self.par['file_aln']
		self.par['zscore_file'] = self.par['dir_input'] + self.par['zscore_file']
		
		if self.par['use_filter'].lower() == 'yes':
			self.par['use_filter'] = True
		elif self.par['use_filter'].lower() == 'no':
			self.par['use_filter'] = False
		
		#default parameters
		self.par['Z_threshold']=int(self.par['Z_threshold']) if 'Z_threshold' in self.par else 10
		self.par['p_threshold']=float(self.par['p_threshold']) if 'p_threshold' in self.par else 0.01
		self.par['x_threshold']=int(self.par['x_threshold']) if 'x_threshold' in self.par else 2
		self.par['bh_threshold']=float(self.par['bh_threshold']) if 'bh_threshold' in self.par else 0.05
		#
		return self.par
	
	def probability_ref(self):
		# Writes to file probability tables later used in binomial assessments 
		viral_peptidome = open(self.par['file_annotation'], 'r')
		peptide_lib = []
		next(viral_peptidome)
		for line in viral_peptidome:
			items = line.split('\t')
			peptide_lib.append(str(items[0]))
		viral_peptidome.close()
		peptide_lib = peptide_lib[:-1]
		#peptide_lib.sort(key=int)
		
		binary_b = flex_array.sparse_aln_df(self.par['file_aln'])
		binary_b = binary_b.reindex(peptide_lib).fillna(0)
		binary_b = flex_array.array(binary_b).filter_aln(ref_seq=self.par['dir_ref_seq'])
		binary_b = pd.DataFrame(index=binary_b.index, columns=binary_b.columns, data=binary_b.values, dtype=bool)
		
		virus_aln = {i:binary_b.loc[binary_b[i], i] for i in binary_b.columns} #list of alignments to feed into filter
		dep_pep = self.dependent_peptides()
		virus_tot_filter = [flex_array.gen_ind_hits(virus_aln[i], dep_pep) for i in virus_aln]
		#virus_aln = dict(zip(list(binary_b.columns), virus_aln))
		
		virus_sums = pd.Series(index=binary_b.columns, data=[len(i) for i in virus_tot_filter])#binary_b.apply(np.count_nonzero, axis=0)
		first_round_prob = pd.Series(index=binary_b.columns)
		viruses = list(binary_b.columns)
		for i in first_round_prob.index:
			print("Virus " + str(viruses.index(i)))
			first_round_prob[i] = virus_sums[i]/(len(peptide_lib)-(len(virus_aln[i])-virus_sums[i]))
		first_round_prob.to_csv(self.par['dir_ref_seq']+"total_probabilities_20180524.csv", header=False, index=True)
		print("First probability file generated.")
		
		'''
		virus_shared = pd.DataFrame(index=viruses, columns=viruses)
		virus_unique = pd.DataFrame(index=viruses, columns=viruses)
		for i in viruses:
			for j in viruses:
				shared = virus_aln[i]*virus_aln[j]; shared.fillna(0.0, inplace=True);
				shared = shared[shared>0]#; shared = list(shared.index); shared = [str(i) for i in shared]
				#shared = ';'.join(shared)
				
				virus_shared.loc[i,j] = len(shared)#(flex_array.gen_ind_hits(shared, dep_pep))
				
		for i in virus_shared.columns:
			virus_unique[i] = virus_sums[i] - virus_shared[i]
			'''
		
		second_round_prob = pd.DataFrame(index=viruses, columns=viruses)
		third_round_prob = pd.DataFrame(index=viruses, columns=viruses)
		#virus_pairs = []
		#total = 139656
		#count = 0
		
		virus_pairs = list(combo(viruses, 2))
		
		def calc_pair(pair):
			i = pair[0]; j = pair[1]; d1 = {}; d2 = {};
			i_index = set(virus_aln[i].index); j_index = set(virus_aln[j].index)
			shared_index = set(i_index).intersection(j_index); shared = virus_aln[i].loc[list(shared_index)]
			if len(shared) == 0:
				d1[(i,j)] = first_round_prob[j]
				d1[(j,i)] = first_round_prob[i]
				d2[(i,j)] = 0.0
				d2[(j,i)] = 0.0
			else:
				unique_j = j_index - shared_index; unique_j = virus_aln[j].loc[unique_j]
				filter_unique_j = flex_array.gen_ind_hits(unique_j, dep_pep)
				d1[(i,j)] = len(filter_unique_j)/(len(peptide_lib)-len(shared)-(len(unique_j)-len(filter_unique_j)))
				unique_i = i_index - shared_index; unique_i = virus_aln[i].loc[unique_i]
				filter_unique_i = flex_array.gen_ind_hits(unique_i, dep_pep)
				d1[(j,i)] = len(filter_unique_i)/(len(peptide_lib)-len(shared)-(len(unique_i)-len(filter_unique_i)))
				filter_shared = flex_array.gen_ind_hits(shared, dep_pep)
				d2[(i,j)] = len(filter_shared)/(len(peptide_lib)-len(unique_i)-len(unique_j)-(len(shared)-len(filter_shared)))
				d2[(j,i)] = d2[(i,j)]
		
			return d1, d2
		
		results = Parallel(n_jobs=-1, verbose=100000)(delayed(calc_pair)(pair) for pair in virus_pairs)
		m1, m2 = zip(*results)
		
#		for i in index:
#			for j in range(index.index(i),len(second_round_prob.columns)):
#				#pair = set([i, j])
#				#if pair not in virus_pairs:
#					# print progress
#				j = second_round_prob.columns[j]
#				count += 1
#				print("Proportion of pairs evaluated: " + str(count/total))
#				# add pair to list of sets
#				#virus_pairs.append(pair)
#				'''
#				#do uniques
#				shared = virus_aln[i] & virus_aln[j]; shared.fillna(False, inplace=True); shared = shared[shared];
#				unique_j = virus_aln[j] ^ shared; unique_j.fillna(True, inplace=True); unique_j = unique_j[unique_j];
#				filter_unique_j = flex_array.gen_ind_hits(unique_j, dep_pep)
#				second_round_prob.loc[i,j] = len(filter_unique_j)/(len(peptide_lib)-len(shared)-(len(unique_j)-len(filter_unique_j)))
#				
#				unique_i = virus_aln[i] ^ shared; unique_i.fillna(True, inplace=True); unique_i = unique_i[unique_i];
#				filter_unique_i = flex_array.gen_ind_hits(unique_i, dep_pep)
#				second_round_prob.loc[j,i] = len(filter_unique_i)/(len(peptide_lib)-len(shared)-(len(unique_i)-len(filter_unique_i)))
#				# now do shared probabilities
#				filter_shared = flex_array.gen_ind_hits(shared, dep_pep)
#				third_round_prob.loc[i,j] = len(filter_shared)/(len(peptide_lib)-len(unique_i)-len(unique_j)-(len(shared)-len(filter_shared)))
#				third_round_prob.loc[j,i] = third_round_prob.loc[i,j]
#				'''
#				# find shared
#				i_index = set(virus_aln[i].index); j_index = set(virus_aln[j].index)
#				shared_index = set(i_index).intersection(j_index); shared = virus_aln[i].loc[list(shared_index)]
#				# set values if shared is 0
#				if len(shared) == 0:
#					second_round_prob.loc[i,j] = first_round_prob[j]
#					second_round_prob.loc[j,i] = first_round_prob[i]
#					third_round_prob.loc[i,j] = 0.0
#					third_round_prob.loc[j,i] = 0.0
#				else:
#					# unique at j
#					unique_j = j_index - shared_index; unique_j = virus_aln[j].loc[unique_j]
#					filter_unique_j = flex_array.gen_ind_hits(unique_j, dep_pep)
#					second_round_prob.loc[i,j] = len(filter_unique_j)/(len(peptide_lib)-len(shared)-(len(unique_j)-len(filter_unique_j)))
#					# unique at i
#					unique_i = i_index - shared_index; unique_i = virus_aln[i].loc[unique_i]
#					filter_unique_i = flex_array.gen_ind_hits(unique_i, dep_pep)
#					second_round_prob.loc[j,i] = len(filter_unique_i)/(len(peptide_lib)-len(shared)-(len(unique_i)-len(filter_unique_i)))
#					# shared prob
#					filter_shared = flex_array.gen_ind_hits(shared, dep_pep)
#					third_round_prob.loc[i,j] = len(filter_shared)/(len(peptide_lib)-len(unique_i)-len(unique_j)-(len(shared)-len(filter_shared)))
#					third_round_prob.loc[j,i] = third_round_prob.loc[i,j]
#		
						
				
		second_round_prob.to_csv(self.par['dir_ref_seq']+"unique_probabilities_20180524.csv", header=True, index=True)
		print("Second probability file generated.")
		
		third_round_prob.to_csv(self.par['dir_ref_seq']+"shared_probabilities_20180524.csv", header=True, index=True)
		print("Third (and last) probability file generated.")
		
		'''
				a = binary_b[i]; b = binary_b[j]
				virus_intersections.loc[i,j] = np.dot(a,b)
		
		virus_unique = pd.DataFrame(index=viruses, columns=viruses)
		for i in virus_intersections.columns:
			virus_unique[i] = virus_sums[i] - virus_intersections[i]
		
		second_round_prob = pd.DataFrame(index=viruses, columns=viruses)
		for i in virus_intersections.index:
			for j in virus_intersections.columns:
				second_round_prob.loc[i,j] = virus_unique.loc[i,j]/(len(peptide_lib)-virus_intersections.loc[i,j])
		second_round_prob.to_csv(self.par['dir_ref_seq']+"unique_probabilities.csv", header=True, index=True)
		print("Second probability file generated.")
		
		third_round_prob = pd.DataFrame(index=viruses, columns=viruses)
		for i in virus_intersections.index:
			for j in virus_intersections.columns:
				third_round_prob.loc[i,j] = virus_intersections.loc[i,j]/(len(peptide_lib)-virus_unique.loc[i,j]-virus_unique.loc[j,i])
		third_round_prob.to_csv(self.par['dir_ref_seq']+"shared_probabilities.csv", header=True, index=True)
		print("Third (and last) probability file generated.")
		'''
		
		return None
		
	def dependent_peptides(self):
		dependent_pep = {}
		file_dependent = self.par['dir_ref_seq']+'virus_dependent_peptides_trunc_v2_20180522.csv'
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

# End