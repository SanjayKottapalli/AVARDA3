# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 15:41:05 2017

@authors: Sanjay Kottapalli, Tiezheng Yuan
"""
import params
import pandas as pd
import numpy as np
import timeit
import networkx as nx
from scipy.stats import binom_test
from joblib import Parallel, delayed


def standard_df(infile):
	sep = ',' if infile.endswith('csv') else '\t'
	stand_df = pd.read_csv(infile, header=0, index_col=0, sep=sep, low_memory=False)
	stand_df.index=[str(i) for i in stand_df.index]
	stand_df.columns=[str(i) for i in stand_df.columns]
	stand_df.fillna(0, inplace=True)
	return stand_df

def binary_aln_df(infile):
	print('Reading alignment file: ' + infile)
	file_sep = '\t' if infile.endswith('txt') else ','
	aln_df = params.file_IO(infile, file_sep).flat_file_to_df([0,1,11])
	#convert to binary matrix
	binary_b=pd.DataFrame(np.where(aln_df > 0, 1, 0))
	binary_b.index=[str(i).split('_')[1] for i in aln_df.index]
	binary_b.columns=[str(i).replace(',', ';') for i in aln_df.columns]
	binary_b.fillna(0, inplace=True)
	return binary_b

def full_aln_df(infile):
	print('Reading alignment file: ' + infile)
	file_sep = '\t' if infile.endswith('txt') else ','
	aln_df = params.file_IO(infile, file_sep).flat_file_to_df([0,1,11])
	#aln_df.index=[str(i).split('_')[1] for i in aln_df.index]
	aln_df.columns=[str(i).replace(',', ';') for i in aln_df.columns]
	aln_df.fillna(0, inplace=True)
	return aln_df

def sparse_aln_df(infile):
    print('Reading alignment file: ' + infile)
    aln_df = pd.read_pickle(infile)
    #aln_df = aln_df.to_dense()
    binary_b = pd.DataFrame(np.where(aln_df > 0, 1, 0), index=[str(i) for i in aln_df.index], columns=aln_df.columns)
    binary_b.fillna(0, inplace=True)
    return binary_b

'''
def indep_subgraph(peptide_hits, overlap_dict):
	#Take only key-value pairs from overlap_dict for the sample hits
	sub_dict = {i: overlap_dict[i] for i in peptide_hits if i in overlap_dict}
	#Generated a subgraph of relationships between all sample hits
	G = nx.Graph(sub_dict)
	for i in list(G.nodes()):
		if i not in peptide_hits:
			G.remove_node(i)
	#Add peptides that have no connections to others
	for i in peptide_hits:
		if i not in list(G.nodes()):
			G.add_node(i)
	#Find independent set (10 iterations)
	isolates = nx.isolates(G)
	max_list = []
	for i in range(10):
		max1 = nx.maximal_independent_set(G, isolates)
		max_list.append(max1)
	list_length = [len(i) for i in max_list]
	index = list_length.index(max(list_length))
	max1 = max_list[index]
		
	return max1
'''

def binom_reassign(zb_df, hits, sample_name, dep_pep, ref_seq, p_threshold=0.01, hits_threshold=2, organism=False):
	time1 = timeit.default_timer()
	#Read probability files
	if organism:
		org=ref_seq+'org_'
	else:
		org=ref_seq
	
	first_round_prob = pd.read_csv(org+"total_probabilities_20181110.csv", index_col=0, header=None, squeeze=True)
	second_round_prob = pd.read_csv(org+"unique_probabilities_20181110.csv", header=0, index_col=0)
	#third_round_prob = pd.read_csv(org+"shared_probabilities_20180809.csv", header=0, index_col=0)
	
	zb_df = zb_df.loc[:,first_round_prob.index]
	
	#Function for defining significance
	def is_sig(p, n, x):
		p_value = binom_test(p=p, n=n, x=x, alternative='greater')
		if p_value < p_threshold and x > hits_threshold:
			return True
		else:
			return False
	
	zb_df_orig = zb_df.copy()
	zb_df = zb_df.astype(bool)
	
	#for use in reassignments:
	#virus_evi = {i:zb_df_orig.loc[zb_df[i], i] for i in zb_df.columns}
	virus_evi = {i:zb_df_orig[zb_df[i]] for i in zb_df.columns}
	virus_xr = {i:virus_evi[i][virus_evi[i]<80] for i in zb_df.columns}
	virus_evi = {i:virus_evi[i][virus_evi[i]>=80] for i in zb_df.columns}
	
	#virus_aln = [zb_df.loc[zb_df[i], i] for i in zb_df.columns]
	virus_tot_filter = {i:gen_ind_hits(hits.loc[virus_evi[i][i].dropna().index], dep_pep) for i in virus_evi.keys()}
	#virus_aln = dict(zip(list(binary_b.columns), virus_aln))
	
	#Series of p-values for each virus
	p_series = pd.Series(index=list(zb_df.columns))
	n_series = pd.Series(index=list(zb_df.columns))
	filter_series = pd.Series(index=list(zb_df.columns))
	ranked_hits = {i:len(virus_tot_filter[i]) for i in zb_df.columns}
	ranked_hits = pd.Series(ranked_hits)
	#ranked_hits = pd.Series(index=list(zb_df.columns), data=[len(i) for i in virus_tot_filter])
	n_rank = {i:len(gen_ind_hits(hits.loc[list(set(hits.index)-(set(virus_evi[i][i].dropna().index)-set(virus_tot_filter[i].index)))], dep_pep)) for i in zb_df.columns}
	n_rank = pd.Series(n_rank)
	#Number of hits to each virus
	#ranked_hits = zb_df.apply(np.count_nonzero, axis=0)
	
	orig_viruses = zb_df.columns
	zb_df = zb_df[ranked_hits[ranked_hits>0].index]
	
	
	#Calculate p-values for each virus based on the number of total hits
	#n_rank = len(gen_ind_hits(hits, dep_pep))
	n_hits = hits.copy()
	
	virus_pvalues_1 = pd.Series(index=list(zb_df.columns))
	for i in virus_pvalues_1.index:
		#print('Virus:\t'+i+'\tNull p:\t'+str(first_round_prob[i])+'\tN:\t'+str(n_rank)+'\tX:\t'+str(ranked_hits[i]),flush=True)
		
		virus_pvalues_1[i] = binom_test(p=first_round_prob[i], n=n_rank[i], x=ranked_hits[i], alternative='greater')
	
	#Pre-greedy p-values for output
	orig_pseries = virus_pvalues_1.copy()
	
	#Sort viruses by initial p-value
	virus_pvalues_1.sort_values(inplace=True, ascending=True)
	#Sort virus hits series by p-value
	ranked_hits = ranked_hits[virus_pvalues_1.index]
	specie = list(ranked_hits.index)
	print("Number of species: "+str(len(specie)), flush=True)
	#first_sp = specie[0]
	#print("UNIQUE NUMBERS : ")
	#s_num = len(specie)
	#test_num = 0
	
	zb_df_orig = zb_df_orig[zb_df.columns]
	
	while len(ranked_hits)>0 and ranked_hits.iloc[0]>0:
		#Comparing top hit to everything else (reassignments only)
		zb_df = zb_df.astype(bool) #cast to bool for faster computation
		#if specie[0] == first_sp:
		#test_num += 1
		print("Virus: "+str(specie[0]), flush=True)#+str(test_num)+" of "+str(s_num), flush=True)
		for i in specie[1:]:
			#Sync these variables with zb_df
			#virus_evi = {i:zb_df_orig.loc[zb_df[i], i] for i in zb_df.columns}
			virus_evi = {j:zb_df_orig[zb_df[j]][j] for j in zb_df.columns}
			virus_xr = {k:virus_evi[k][virus_evi[k]<80] for k in zb_df.columns}
			virus_evi = {l:virus_evi[l][virus_evi[l]>=80] for l in zb_df.columns}
			########
			high_index = set(virus_evi[specie[0]].dropna().index); i_index = set(virus_evi[i].dropna().index)
			xrhigh_index = set(virus_xr[specie[0]].dropna().index); xri_index = set(virus_xr[i].dropna().index)
			xr_shared_index = set(xrhigh_index).intersection(xri_index)
			high_all = high_index.union(xrhigh_index); i_all = i_index.union(xri_index)
			shared_all = high_all.intersection(i_all)
			
			high_shared = high_index.intersection(shared_all); i_shared = i_index.intersection(shared_all)
			
			shared_evidence = shared_all - xr_shared_index
			
			########
			#Top hit virus (binary vector)
			#highrank = zb_df[specie[0]]
			#i_rank = zb_df[i]
			#element-wise multiplication to get the vector of shared peptides
			#shared_peps = np.multiply(i_rank, highrank)
			#shared_num = np.count_nonzero(shared_peps)
			'''
			print(shared_peps)
			print(shared_num)
			print(highrank)
			print(i_rank)
			print('---------------------------------------')
			'''
			#print(str(len(shared_evidence)), flush=True)
			#Only do reassignment/sim tags if there is overlap between the two viruses
			if len(shared_evidence) > 0:
				#print(specie[0], i, flush=True)
				unique_high = high_index - shared_all; unique_high = hits.loc[unique_high]#unique_high = virus_evi[specie[0]][specie[0]].dropna().loc[unique_high] #unique evidence for j
				unique_i = i_index - shared_all; unique_i = hits.loc[unique_i]
				
				#arrays = (unique_high, unique_i)
				#try:
				#filter_unique_high, filter_unique_i = Parallel(n_jobs=-1)(delayed(gen_ind_hits)(array, dep_pep) for array in arrays)
				if len(unique_high) != 0:
					filter_unique_high = gen_ind_hits(unique_high, dep_pep)
					filter_unique_high = set(filter_unique_high.index)
				else:
					filter_unique_high = set()
					
				if len(unique_i) != 0:
					filter_unique_i = gen_ind_hits(unique_i, dep_pep)
					filter_unique_i = set(filter_unique_i.index)
				else:
					filter_unique_i = set()
				#filter_unique_high = set(filter_unique_high.index); filter_unique_i = set(filter_unique_i.index)
				#except:
					#filter_unique_high = set(); filter_unique_i = set();
				#del arrays
				top_filter_num = len(filter_unique_high); i_filter_num = len(filter_unique_i)
				
				#shared_peps = shared_peps[shared_peps>0]
				unique_high = set(unique_high.index)
				n_high = n_hits.loc[list(set(n_hits.index)-high_shared-(unique_high-filter_unique_high))]
				#n_high = len(gen_ind_hits(n_high, dep_pep))
				unique_i = set(unique_i.index)
				n_i = n_hits.loc[list(set(n_hits.index)-i_shared-(unique_i-filter_unique_i))]
				#n_i = len(gen_ind_hits(n_i, dep_pep))
				
				n_high, n_i = Parallel(n_jobs=-1)(delayed(gen_ind_hits)(array, dep_pep) for array in [n_high, n_i])
				n_high = len(n_high); n_i = len(n_i)
				
				#high_num = np.count_nonzero(highrank)
				#i_num = np.count_nonzero(i_rank)
				#
				#shared_peps = shared_peps[shared_peps>0]
				'''
				unique_top = highrank^shared_peps; unique_top = unique_top[unique_top>0]#print('unique top: ',sum(unique_top)); unique_top = unique_top[unique_top>0]
				unique_i = i_rank^shared_peps; unique_i = unique_i[unique_i>0]
				filter_unique_top = gen_ind_hits(hits.loc[unique_top.index], dep_pep)
				filter_unique_i = gen_ind_hits(hits.loc[unique_i.index], dep_pep)
				top_filter_num = len(filter_unique_top)
				i_filter_num = len(filter_unique_i)
				'''
				'''
				print(unique_top)
				print('-------')
				print(unique_i)
				print('-------')
				print(filter_unique_top)
				print('-------')
				print(filter_unique_i)
				print('-------')
				print(top_filter_num)
				print('-------')
				print(i_filter_num)
				print('-------')
				'''
				#
				#if 'alphaherpesvirus_1' in specie[0]:
					#print(str((specie[0],i,'n_rank: ',n_rank,'shared: ',shared_num,'top hit unique: ',len(unique_top), 'top hit filtered unique: ',top_filter_num,'unique i: ',len(unique_i),'i unique filtered: ',i_filter_num))+' top unique: '+str(binom_test(p=second_round_prob.loc[i,specie[0]], n=n_rank-shared_num-(len(unique_top)-top_filter_num), x=top_filter_num, alternative='greater'))+' i unique: ' + str(binom_test(p=second_round_prob.loc[specie[0],i], n=n_rank-shared_num-(len(unique_i)-i_filter_num), x=i_filter_num, alternative='greater')))
				#print(str((specie[0],i,'n_rank: ',n_rank,'shared: ',shared_num,'top hit unique: ',len(unique_top), 'top hit filtered unique: ',top_filter_num,'unique i: ',len(unique_i),'i unique filtered: ',i_filter_num))+' top unique: '+str(binom_test(p=second_round_prob.loc[i,specie[0]], n=n_rank-shared_num-(len(unique_top)-top_filter_num), x=top_filter_num, alternative='greater'))+' i unique: ' + str(binom_test(p=second_round_prob.loc[specie[0],i], n=n_rank-shared_num-(len(unique_i)-i_filter_num), x=i_filter_num, alternative='greater')))
				#If only one passes the threshold, reassign peptides accordingly
				#print('High virus: '+specie[0]+' Unique high p-value: ' +str(binom_test(top_filter_num, n_high, second_round_prob.loc[i,specie[0]])), flush=True)
				#print('i virus: '+i+' Unique i p-value: ' +str(binom_test(i_filter_num, n_i, second_round_prob.loc[specie[0],i])), flush=True)

				#print('Before, Top virus ',specie[0],' i Virus: ',i,'evi high: ',sum(zb_df.loc[:,specie[0]]),' evi i: ',sum(zb_df.loc[:,i]), flush=True)
				
				if is_sig(second_round_prob.loc[i,specie[0]], n_high, top_filter_num) and not is_sig(second_round_prob.loc[specie[0],i], n_i, i_filter_num):#highrank_pvalue < p_threshold and i_pvalue >= p_threshold and high_num-shared_num > hits_threshold and i_num-shared_num > hits_threshold:
					#Subtract shared peptides from the insignificant virus
					#print(sum(zb_df.loc[:,i]))
					#zb_df.loc[:,i] = i_rank^shared_peps
					zb_df.loc[list(i_shared),i] = False
					zb_df_orig.loc[list(i_shared),i] = 0.0
					#print("Virus pair: ",specie[0],',',i,"N_rank: ",str(n_rank)," N_pair: ",str(n_pair),'shared: ',shared_num,'top hit filtered unique: ',top_filter_num,'i unique filtered: ',i_filter_num,", Reassigned to highrank")
					#print(sum(zb_df.loc[:,i]))
				elif not is_sig(second_round_prob.loc[i,specie[0]], n_high, top_filter_num) and is_sig(second_round_prob.loc[specie[0],i], n_i, i_filter_num):#highrank_pvalue >= p_threshold and i_pvalue < p_threshold and high_num-shared_num > hits_threshold and i_num-shared_num > hits_threshold:
					#Same as above
					#print(sum(zb_df.loc[:,specie[0]]))
					#zb_df.loc[:,specie[0]] = highrank^shared_peps
					zb_df.loc[list(high_shared),specie[0]] = False
					zb_df_orig.loc[list(high_shared),specie[0]] = 0.0
					#print("Virus pair: ",specie[0],',',i,"N_rank: ",str(n_rank)," N_pair: ",str(n_pair),'shared: ',shared_num,'top hit filtered unique: ',top_filter_num,'i unique filtered: ',i_filter_num,", Reassigned to i_rank")
					#print(sum(zb_df.loc[:,specie[0]]))
				#print('After, Top virus ',specie[0],' i Virus: ',i,'evi high: ',sum(zb_df.loc[:,specie[0]]),' evi i: ',sum(zb_df.loc[:,i]), flush=True)
				#print('max i ppos ', max(zb_df_orig[zb_df[i]][i]),flush=True)
				zb_df = zb_df.astype(bool) #recast in case of subtraction back to int
		
		#Add p-value to series after any potential reassignments
		top_hit = specie[0]
		
		virus_evi = {m:zb_df_orig[zb_df[m]][m] for m in zb_df.columns}
		virus_evi = {n:virus_evi[n][virus_evi[n]>=80] for n in zb_df.columns}
		
		filter_top = virus_evi[top_hit].dropna()#zb_df.loc[zb_df[top_hit], top_hit]
		filter_top_index = set(filter_top.index).intersection(set(n_hits.index))
		filter_top = gen_ind_hits(n_hits.loc[filter_top_index], dep_pep)
		
		n_rank = len(gen_ind_hits(n_hits.loc[list(set(n_hits.index)-(set(virus_evi[top_hit].dropna().index)-set(filter_top.index)))], dep_pep))
		
		p_series[top_hit] = binom_test(p=first_round_prob[top_hit], n=n_rank, x=len(filter_top), alternative='greater')
		
		#print('')
		
		n_series[top_hit] = n_rank
		#print('Top virus: ',top_hit.split(',s__')[-1],' filtered evidence: ',list(filter_top.index),flush=True)
		filter_series[top_hit] = len(filter_top)
		
		zb_df = zb_df.astype(int)
		#Figure out how many peptides were globally unique to highrank
		high_peptides = np.where(zb_df[specie[0]] == 1)[0]
		#high_hits = zb_df[zb_df[specie[0]]==1].apply(np.count_nonzero, axis=1)
		#high_unique = len(high_hits[high_hits==1])
		high_unique = (zb_df.iloc[high_peptides,:].apply(np.count_nonzero, axis=1))[zb_df.iloc[high_peptides,:].apply(np.count_nonzero, axis=1)==1]
		high_unique = high_unique.index
		#Now remove the highest hit since it will not be involved in subsequent comparisons
		#print('length of specie: '+str(len(specie)), flush=True)
		specie.remove(specie[0])
		#print('length of specie: '+str(len(specie)), flush=True)

		#Re-rank virus hits by total binomial
		'''
		virus_aln = [zb_df.loc[zb_df[i], i] for i in zb_df.columns]
		virus_tot_filter = [gen_ind_hits(hits.loc[i.index], dep_pep) for i in virus_aln]
		ranked_hits = pd.Series(index=list(zb_df.columns), data=[len(i) for i in virus_tot_filter])			#ranked_hits = ranked_hits[specie] (unnecessary)
		virus_pvalues_1 = pd.Series(index=specie)
		'''
		#Adjust the n used for binomial tests
		n_hits = n_hits.loc[list(set(n_hits.index)-set(high_unique))]
		#n_rank = len(gen_ind_hits(n_hits, dep_pep))
		
		#virus_tot_filter = {i:gen_ind_hits(n_hits.loc[virus_evi[i][i].dropna().index], dep_pep) for i in specie}
		#virus_tot_filter = {i:n_hits.loc[virus_evi[i][i].dropna().index] for i in specie}
		virus_tot_filter = {o:set(n_hits.index).intersection(set(virus_evi[o].dropna().index)) for o in specie}
		ranked_hits = {p:len(virus_tot_filter[p]) for p in specie}
		ranked_hits = pd.Series(ranked_hits)
		#print('length of specie: '+str(len(specie)), flush=True)
		#print("max ranked hits: "+str(max(ranked_hits)), flush=True)
		
		#ranked_hits = pd.Series(index=list(zb_df.columns), data=[len(i) for i in virus_tot_filter])
		#n_rank = {i:len(gen_ind_hits(n_hits.loc[list(set(n_hits.index)-(set(virus_evi[i][i].dropna().index)-set(virus_tot_filter[i].index)))], dep_pep)) for i in specie}
		#n_rank = {i:len(n_hits.loc[list(set(n_hits.index)-(set(virus_evi[i][i].dropna().index)-set(virus_tot_filter[i].index)))]) for i in specie}
		n_rank = {q:len(n_hits.index) for q in specie}
		n_rank = pd.Series(n_rank)
		
		#ranked_hits = ranked_hits[ranked_hits>0].dropna()
		
		#Sort viruses by p-value
		virus_pvalues_1 = pd.Series(index=specie)
		for r in specie:
			virus_pvalues_1[r] = binom_test(p=first_round_prob[r], n=n_rank[r], x=ranked_hits[r], alternative='greater')
		virus_pvalues_1.sort_values(inplace=True, ascending=True)
		#print('length of specie: '+str(len(specie)), flush=True)
		
		specie = list(virus_pvalues_1.index)
		#print('length of specie: '+str(len(specie)), flush=True)
		#ranked_hits = ranked_hits[ranked_hits>0]
		ranked_hits = ranked_hits[specie].dropna()
		
		#print("max ranked hits: "+str(max(ranked_hits)), flush=True)
		
		zb_df = zb_df.astype(bool)
	
	#Calculate sim tags after performing all peptide reassignments (REPLACE THIS)
	'''
	virus_aln = [zb_df.loc[zb_df[i], i] for i in zb_df.columns]
	virus_tot_filter = [gen_ind_hits(hits.loc[i.index], dep_pep) for i in virus_aln]
	ranked_hits = pd.Series(index=list(zb_df.columns), data=[len(i) for i in virus_tot_filter])
	#ranked_hits = zb_df.apply(np.count_nonzero, axis=0)
	
	n_rank = len(gen_ind_hits(hits, dep_pep))
	n_hits = hits.copy()
	
	virus_pvalues_1 = pd.Series(index=list(zb_df.columns))
	for i in virus_pvalues_1.index:
		virus_pvalues_1[i] = binom_test(p=first_round_prob[i], n=n_rank, x=ranked_hits[i], alternative='greater')
	virus_pvalues_1.sort_values(inplace=True, ascending=True)
	ranked_hits = ranked_hits[virus_pvalues_1.index]
	'''
	#Sim tags for each virus, list of species to be examined
	specie=list(ranked_hits.index)
	sim_tag = pd.Series(float(0), index=specie)
	tag = 0
			
	#print("SHARED NUMBERS : ")
	
	zb_df = zb_df.astype(bool)
	'''
	while len(ranked_hits)>0 and ranked_hits.iloc[0]>0:
		for i in specie[1:]:
			highrank = zb_df[specie[0]]
			i_rank = zb_df[i]
			shared_peps = np.multiply(i_rank, highrank)
			shared_num = np.count_nonzero(shared_peps)
			#Only do sim tags if there is overlap between the two viruses
			if shared_num > 0:
				i_rank = i_rank[i_rank>0]
				highrank = highrank[highrank>0]
				shared_peps = shared_peps[shared_peps>0]
				pair_union = set(i_rank.index).union(highrank.index)
				unique_pair = pair_union-set(shared_peps.index)
				n_pair = n_hits.loc[list(set(n_hits.index)-unique_pair)]
				n_pair = len(gen_ind_hits(n_pair, dep_pep))
				#high_num = np.count_nonzero(highrank)
				
				#print("high_num ", high_num)
				#i_num = np.count_nonzero(i_rank)
				
				#print("i_num ", i_num)
				#
				
				filter_shared_top = gen_ind_hits(hits.loc[shared_peps.index], dep_pep)
				shared_filter_num = len(filter_shared_top)
				
				unique_top = highrank^shared_peps; unique_top = unique_top[unique_top>0]; #print('unique top: ',sum(unique_top));
				unique_i = i_rank^shared_peps; unique_i = unique_i[unique_i>0]
				filter_unique_top = gen_ind_hits(hits.loc[unique_top.index], dep_pep)
				filter_unique_i = gen_ind_hits(hits.loc[unique_i.index], dep_pep)
				top_filter_num = len(filter_unique_top)
				i_filter_num = len(filter_unique_i)
				#
				#print(str((specie[0],i,'n_rank: ',n_rank,'shared: ',shared_num,'top hit unique: ',len(unique_top), 'top hit filtered unique: ',top_filter_num,'unique i: ',len(unique_i),'i unique filtered: ',i_filter_num))+' top unique: '+str(binom_test(p=second_round_prob.loc[i,specie[0]], n=n_rank-shared_num-(len(unique_top)-top_filter_num), x=top_filter_num, alternative='greater'))+' i unique: ' + str(binom_test(p=second_round_prob.loc[specie[0],i], n=n_rank-shared_num-(len(unique_i)-i_filter_num), x=i_filter_num, alternative='greater')))
				
				#If neither passes (using shared test, symmetric probability table) and neither can stand on their own:
				if is_sig(third_round_prob.loc[i,specie[0]], n_pair, shared_filter_num) and not is_sig(second_round_prob.loc[i,specie[0]], n_pair, top_filter_num) and not is_sig(second_round_prob.loc[specie[0],i], n_pair, i_filter_num):
					#If neither of them already has a sim tag
					if np.array_equal(sim_tag[[specie[0],i]], [0,0]):
						tag += 1
						#Add a tag to the list for both viruses
						sim_tag.loc[[specie[0],i]] = 0.001*tag
						#print((specie[0],i,n_rank))
					#If either already has a tag, assign that tag to the other
					elif sim_tag[specie[0]] != 0 and sim_tag[i] == 0:
						sim_tag.loc[i] = sim_tag[specie[0]]
					elif sim_tag[i] != 0 and sim_tag[specie[0]] == 0:
						sim_tag.loc[specie[0]] = sim_tag[i]
				#zb_df = zb_df.astype(bool)
		
		#Figure out how many peptides were globally unique to highrank
		
		high_peptides = np.where(zb_df[specie[0]] == 1)[0]
		high_unique = (zb_df.iloc[high_peptides,:].apply(np.count_nonzero, axis=1))[zb_df.iloc[high_peptides,:].apply(np.count_nonzero, axis=1)==1]
		high_unique = high_unique.index
		#high_unique = len(np.where(zb_df.iloc[high_peptides,:].apply(np.count_nonzero, axis=1) == 1)[0])
		
		#high_hits = zb_df[zb_df[specie[0]]==1].apply(np.count_nonzero, axis=1)
		#high_unique = np.count_nonzero(high_hits)#len(high_hits[high_hits==1])
		#Now remove the highest hit since it will not be involved in subsequent comparisons
		specie.remove(specie[0])
		#Re-rank virus hits by total binomial (UNNECESSARY)
		#ranked_hits = zb_df.apply(np.count_nonzero, axis=0)
		#ranked_hits = ranked_hits[specie]
		#virus_pvalues_1 = pd.Series(index=specie)
		n_hits = n_hits.loc[list(set(n_hits.index)-set(high_unique))]
		n_rank = len(gen_ind_hits(n_hits, dep_pep))
		
		#for i in specie:
		#	virus_pvalues_1[i] = binom_test(p=first_round_prob[i], n=n_rank, x=ranked_hits[i], alternative='greater')
		#Sort viruses by p-value
		
		virus_pvalues_1 = virus_pvalues_1[specie]
		virus_pvalues_1.sort_values(inplace=True, ascending=True)
		#Sort virus hits series by p-value
		specie = list(virus_pvalues_1.index)
		ranked_hits = ranked_hits[specie]
	'''
	zb_df = zb_df.astype(int)
	#Generate unique values matrix
	glob_unique = zb_df.copy()
	for i in glob_unique.index:
		if sum(glob_unique.loc[i]) > 1:
			glob_unique.loc[i] -= glob_unique.loc[i]
	
	#Generate adjusted p-values for the p-value output Series (using the R package)
	#stats = importr('stats')
	#p_adjusted = stats.p_adjust(FloatVector(p_series.values), method='BH')​
	zb_df = zb_df.reindex(columns=orig_viruses, fill_value=0)
	glob_unique = glob_unique.reindex(columns=orig_viruses, fill_value=0)
	sim_tag = sim_tag.reindex(orig_viruses, fill_value=0)
	n_series = n_series.reindex(orig_viruses, fill_value=0)
	p_series = p_series.reindex(orig_viruses, fill_value=0)
	orig_pseries = orig_pseries.reindex(orig_viruses, fill_value=0)
	filter_series = filter_series.reindex(orig_viruses, fill_value=0)
	
	print("Done with reassignments! Time: ", timeit.default_timer()-time1)
	return zb_df, glob_unique, sim_tag, p_series, orig_pseries, filter_series, sample_name, n_series#, p_adjust_series
	

def gen_ind_hits(hits, overlap_dict, graph_dir='', sample_number=None):
	hits_series = pd.Series(hits).astype(float)
	#Start time, for timing purposes
	#start_time = timeit.default_timer()
	#Take only key-value pairs from overlap_dict for the sample hits
	hits_dict = hits_series.to_dict()
	peptide_hits = [str(i) for i in hits_dict.keys()]
	sub_dict = {i: overlap_dict[i] for i in peptide_hits if i in overlap_dict}
	#Generated a subgraph of relationships between all sample hits
	G = nx.Graph(sub_dict)
	for i in list(G.nodes()):
		if i not in peptide_hits:
			G.remove_node(i)
	#Add peptides that have no connections to others
	for i in peptide_hits:
		if i not in list(G.nodes()):
			G.add_node(i)
	#Add z-scores as attributes to the graph
	#num_hits = len(hits_series)
	zscore = hits_series.to_dict()
	zscore = {i:float(zscore[i]) for i in zscore}
	nx.set_node_attributes(G, name='Z-Score', values=zscore)
	zscore = nx.get_node_attributes(G, name='Z-Score')
	
	edge = nx.get_edge_attributes(G, name='weight')
	edge = {i:float(edge[i]) for i in edge}
	nx.set_edge_attributes(G, name='weight', values=edge)
	
	#Write graphml file to appropriate directory
	#nx.write_graphml(G, graph_dir+sample_name.replace('.','-')+'.graphml')
	#nx.write_graphml(G, graph_dir+'sample_' + str(sample_number) + '.graphml')
	
	#Reducing graph to max_degree 2
	if len(G.edges()) != 0:
		degree = dict(G.degree(nbunch=G.nodes()))
		degrees = [i for i in degree.values()]
		max_degree = np.max(degrees)
		while max_degree > 2:
			degree = dict(G.degree(nbunch=G.nodes()))
			vertices = [i for i in degree.keys()]
			degrees = [i for i in degree.values()]
			vertices = np.array(vertices); degrees = np.array(degrees);
			max_degree = np.max(degrees)
			#Remove nodes of highest degree (look for lowest z-score if tie)
			if max_degree > 2:
				max_vertex_indices = np.where(degrees==max_degree)
				max_vertices = vertices[max_vertex_indices]
				max_degree_scores = [zscore[i] for i in max_vertices]
				min_z = min(max_degree_scores)
				for i in max_vertices:
					if zscore[i] == min_z:
						G.remove_node(i)
						break #so that multiple nodes are not removed
			
	#Eliminates one vertex from each cycle (lowest z-score) to convert them into paths
	len_cycles = 1
	#While loop to make sure that there are no cycles left
	while len_cycles != 0:
		cycles = []
		for i in list(G.nodes()):
			try:
				cycles.append(nx.find_cycle(G, source=i))
			except:
				pass
		len_cycles = len(cycles)
		if len_cycles != 0:
			cycles = [np.array(i) for i in cycles]
			cycles = [np.ndarray.flatten(i) for i in cycles]
			cycles = [np.unique(i) for i in cycles]
			cycles = pd.DataFrame(cycles).drop_duplicates().values.tolist()
			for i in range(len(cycles)):
				cycles[i] = [str(j) for j in cycles[i] if str(j) in G.nodes()]
			node_zscores = nx.get_node_attributes(G, name='Z-Score')
			for i in cycles:
				cycle_scores=[node_zscores[k] for k in i]
				min_score = min(cycle_scores)
				for j in i:
					if node_zscores[j] == min_score:
						G.remove_node(j)
						break #otherwise it will eliminate two nodes in one cycle which both have the same z-score
	
	#Code for deleting vertices from paths based on even or odd length
	node_zscores = nx.get_node_attributes(G, name='Z-Score')
	degree = dict(G.degree(nbunch=G.nodes()))
	if len(G.edges()) != 0:
		degrees = [i for i in degree.values()]
		max_degree = max(degrees)
		while(max_degree > 0):
			components = list(nx.connected_component_subgraphs(G))
			for i in components:
				#even paths
				if len(i.nodes())%2 == 0:
					path_scores = [node_zscores[k] for k in i]
					min_score = min(path_scores)
					for j in i.nodes():
						if node_zscores[j] == min_score:
							G.remove_node(j)
							break #same as above
				#odd paths
				elif len(i.nodes())%2 == 1 and len(i.nodes()) != 1:
					endpoints=[]
					for j in i.nodes():
						if degree[j] == 1:
							endpoints.append(j)
					if len(endpoints)==2:
						path = nx.shortest_path(G, source=endpoints[0], target=endpoints[1])
						middle_indices = np.arange(1,len(path)-1,2)
						for i in middle_indices:
							G.remove_node(path[i])
			degree = dict(G.degree(nbunch=G.nodes()))
			degrees = [i for i in degree.values()]
			max_degree = max(degrees)
	
	#num_nodes = len(G.nodes())
	#proportion = np.divide(float(num_nodes), float(num_hits))
	#print("Proportion of hits kept: " + str(proportion))
	
	#Creating pd.Series object from the nodes of the graph
	ind_hits_dict = {i: node_zscores[i] for i in G.nodes()}
	ind_hits_series = pd.Series(ind_hits_dict)
	#print("Number of peptides kept: " + str(len(ind_hits_series)))  
	'''
	end_time = timeit.default_timer()
	sample_time = end_time - start_time
	print("Time it took to remove overlaps: " + str(sample_time))
	'''
	#print("Number of edges in G: " + str(len(G.edges()))) (should be 0, always)
	
	return ind_hits_series

class array:
	def __init__(self, array):
		self.array = array
	
	def filter_aln(self, ref_seq='', min_alignments=10):
		time1 = timeit.default_timer()
		virus_sums = self.array.apply(np.sum, axis=0)
		#viruses = list(self.array.columns)
		'''
		virus_intersections = pd.DataFrame(index=viruses, columns=viruses)
		for i in viruses:
			a = self.array[i]
			for j in viruses:
				b = self.array[j]
				virus_intersections.loc[i,j] = np.sum(np.multiply(a,b))
		
		virus_intersections = pd.DataFrame(index=viruses, columns=viruses, data=virus_intersections.values, dtype=np.float64)
		
		virus_intersections.to_csv(ref_seq+"virus_intersections_20181016.csv", header=True, index=True)
		print('Wrote intersections to file', flush=True)
		'''
		virus_intersections = pd.read_csv(ref_seq+"virus_intersections_20181016.csv", header=0, index_col=0)
		
		shared_prob = virus_intersections.divide(virus_sums, axis='columns')
		np.fill_diagonal(shared_prob.values, 0)
		
		#Create directed graph representation of virus dependencies
		shared_df = pd.DataFrame(columns=['child', 'parent'])
		child = []; parent = [];
		for i in shared_prob.columns:
			if 1.0 in shared_prob[i].values:
				parents = np.array(shared_prob.index)[np.where(shared_prob[i].values == 1.0)[0]]
				for j in parents:
					child.append(i)
					parent.append(j)
		shared_df['child'] = child
		shared_df['parent'] = parent
		G = nx.from_pandas_edgelist(shared_df, source='child', target='parent', create_using=nx.DiGraph())
		virus_lengths = {i:len(i) for i in G.nodes()}
		nx.set_node_attributes(G, name='Length', values=virus_lengths)
		#Remove complete subset viruses
		in_degree = pd.Series(dict(G.in_degree()))
		remove_viruses = list(in_degree.index[np.where(in_degree == 0)[0]])
		G.remove_nodes_from(remove_viruses)
		
		#Find cycles in G, remove all but shortest string virus in each cycle
		len_cycles = 1
		while len_cycles != 0:
			cycles = []
			for i in list(G.nodes()):
				try:
					cycles.append(nx.find_cycle(G, source=i))
				except:
					pass
			len_cycles = len(cycles)
			if len_cycles != 0:
				cycles = [np.array(i) for i in cycles]
				cycles = [np.ndarray.flatten(i) for i in cycles]
				cycles = [np.unique(i) for i in cycles]
				cycles = pd.DataFrame(cycles).drop_duplicates().values.tolist()
				lengths = nx.get_node_attributes(G, name='Length')
				for i in cycles:
					cycle_lengths = {k:lengths[k] for k in i}
					min_length = min(cycle_lengths)
					for j in i:
						if cycle_lengths[j] != min_length:
							remove_viruses.append(j)
							G.remove_node(j)
		
		#Remove viruses with most outgoing edges (denoting subset) iteratively
		if len(G.edges()) != 0:
			out_degree = pd.Series(dict(G.out_degree()))
			while max(out_degree) != 0:
				out_degree = pd.Series(dict(G.out_degree()))
				if max(out_degree) != 0:
					max_virus = out_degree.index[list(out_degree).index(max(out_degree))]
					#print(max_virus)
					remove_viruses.append(max_virus)
					G.remove_node(max_virus)
		'''
		f = open('removed_viruses.txt', 'w')
		for i in remove_viruses:
			f.write(i+'\n')
		f.close()
		'''
		self.array.drop(remove_viruses, axis=1, inplace=True)
		
		#Drop the appropriate viruses from the alignment matrix
		virus_sums = self.array.apply(np.sum, axis=0, raw=True, reduce=True)
		virus_sums_index = list(virus_sums.index)
		for i in range(len(virus_sums)):
			if virus_sums.iloc[i] <= min_alignments:
				self.array.drop(virus_sums_index[i], axis=1, inplace=True)
		
		time2 = timeit.default_timer()
		print("Time to filter the alignment matrix: " + str(time2-time1))
		
		return self.array
	
	def names_string(self, cutoff=0.001):
		hits = pd.Series(self.array)
		hits = hits[hits>=cutoff]
		names_str = ';'.join(list(hits.index))
		return names_str
	'''
	def gen_ind_hits(self, overlap_dict, graph_dir='', sample_number=None):
		hits_series = pd.Series(self.array).astype(float)
		#Start time, for timing purposes
		start_time = timeit.default_timer()
		#Take only key-value pairs from overlap_dict for the sample hits
		hits_dict = hits_series.to_dict()
		peptide_hits = [str(i) for i in hits_dict.keys()]
		sub_dict = {i: overlap_dict[i] for i in peptide_hits if i in overlap_dict}
		#Generated a subgraph of relationships between all sample hits
		G = nx.Graph(sub_dict)
		for i in list(G.nodes()):
			if i not in peptide_hits:
				G.remove_node(i)
		#Add peptides that have no connections to others
		for i in peptide_hits:
			if i not in list(G.nodes()):
				G.add_node(i)
		#Add z-scores as attributes to the graph
		num_hits = len(hits_series)
		zscore = hits_series.to_dict()
		zscore = {i:float(zscore[i]) for i in zscore}
		nx.set_node_attributes(G, name='Z-Score', values=zscore)
		zscore = nx.get_node_attributes(G, name='Z-Score')
		
		edge = nx.get_edge_attributes(G, name='weight')
		edge = {i:float(edge[i]) for i in edge}
		nx.set_edge_attributes(G, name='weight', values=edge)
		
		#Write graphml file to appropriate directory
		#nx.write_graphml(G, graph_dir+sample_name.replace('.','-')+'.graphml')
		nx.write_graphml(G, graph_dir+'sample_' + str(sample_number) + '.graphml')
		
		#Reducing graph to max_degree 2
		if len(G.edges()) != 0:
			degree = dict(G.degree(nbunch=G.nodes()))
			degrees = [i for i in degree.values()]
			max_degree = np.max(degrees)
			while max_degree > 2:
				degree = dict(G.degree(nbunch=G.nodes()))
				vertices = [i for i in degree.keys()]
				degrees = [i for i in degree.values()]
				vertices = np.array(vertices); degrees = np.array(degrees);
				max_degree = np.max(degrees)
				#Remove nodes of highest degree (look for lowest z-score if tie)
				if max_degree > 2:
					max_vertex_indices = np.where(degrees==max_degree)
					max_vertices = vertices[max_vertex_indices]
					max_degree_scores = [zscore[i] for i in max_vertices]
					min_z = min(max_degree_scores)
					for i in max_vertices:
						if zscore[i] == min_z:
							G.remove_node(i)
							break #so that multiple nodes are not removed
				
		#Eliminates one vertex from each cycle (lowest z-score) to convert them into paths
		len_cycles = 1
		#While loop to make sure that there are no cycles left
		while len_cycles != 0:
			cycles = []
			for i in list(G.nodes()):
				try:
					cycles.append(nx.find_cycle(G, source=i))
				except:
					pass
			len_cycles = len(cycles)
			if len_cycles != 0:
				cycles = [np.array(i) for i in cycles]
				cycles = [np.ndarray.flatten(i) for i in cycles]
				cycles = [np.unique(i) for i in cycles]
				cycles = pd.DataFrame(cycles).drop_duplicates().values.tolist()
				for i in range(len(cycles)):
					cycles[i] = [str(j) for j in cycles[i] if str(j) in G.nodes()]
				node_zscores = nx.get_node_attributes(G, name='Z-Score')
				for i in cycles:
					cycle_scores=[node_zscores[k] for k in i]
					min_score = min(cycle_scores)
					for j in i:
						if node_zscores[j] == min_score:
							G.remove_node(j)
							break #otherwise it will eliminate two nodes in one cycle which both have the same z-score
		
		#Code for deleting vertices from paths based on even or odd length
		node_zscores = nx.get_node_attributes(G, name='Z-Score')
		degree = dict(G.degree(nbunch=G.nodes()))
		if len(G.edges()) != 0:
			degrees = [i for i in degree.values()]
			max_degree = max(degrees)
			while(max_degree > 0):
				components = list(nx.connected_component_subgraphs(G))
				for i in components:
					#even paths
					if len(i.nodes())%2 == 0:
						path_scores = [node_zscores[k] for k in i]
						min_score = min(path_scores)
						for j in i.nodes():
							if node_zscores[j] == min_score:
								G.remove_node(j)
								break #same as above
					#odd paths
					elif len(i.nodes())%2 == 1 and len(i.nodes()) != 1:
						endpoints=[]
						for j in i.nodes():
							if degree[j] == 1:
								endpoints.append(j)
						if len(endpoints)==2:
							path = nx.shortest_path(G, source=endpoints[0], target=endpoints[1])
							middle_indices = np.arange(1,len(path)-1,2)
							for i in middle_indices:
								G.remove_node(path[i])
				degree = dict(G.degree(nbunch=G.nodes()))
				degrees = [i for i in degree.values()]
				max_degree = max(degrees)
		
		num_nodes = len(G.nodes())
		proportion = np.divide(float(num_nodes), float(num_hits))
		print("Proportion of hits kept: " + str(proportion))
		
		#Creating pd.Series object from the nodes of the graph
		ind_hits_dict = {i: node_zscores[i] for i in G.nodes()}
		ind_hits_series = pd.Series(ind_hits_dict)
		#print("Number of peptides kept: " + str(len(ind_hits_series)))  
		
		end_time = timeit.default_timer()
		sample_time = end_time - start_time
		print("Time it took to remove overlaps: " + str(sample_time))
		
		#print("Number of edges in G: " + str(len(G.edges()))) (should be 0, always)
		
		return ind_hits_series
	'''
#End