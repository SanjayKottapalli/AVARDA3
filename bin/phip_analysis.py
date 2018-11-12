# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 15:39:38 2017

@authors: Sanjay Kottapalli, Tiezheng Yuan
"""
import flex_array
import params
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from statsmodels.sandbox.stats.multicomp import multipletests

class phip:
	def __init__(self, par):
		self.par = par
		self.dependent_pep = params.param_dict(par).dependent_peptides()
		self.run_analysis()
	'''
	def dependent_peptides(self):
		self.dependent_pep = {}
		file_dependent = self.par['dir_ref_seq']+'virus_dependent_peptides_trunc.csv'
		f = open(file_dependent, 'r')
		for line in f:
			line = line.strip().split(',')
			pep1 = str(line[0])
			pep2 = str(line[1])
			if pep1 in self.dependent_pep:
				self.dependent_pep[pep1].append(pep2)
			else:
				self.dependent_pep[pep1] = [pep2]
		f.close()
		return self.dependent_pep
	'''
	def run_analysis(self):
		zdf = flex_array.standard_df(self.par['zscore_file'])

		f = params.file_IO('../ref_seq/pep_against_human_viruses.tblastn.species.noseg.WS3.max_target_seqs100000.180625 (1).m8', '\t')
		orig_aln = f.flat_file_to_df([0,1,15])
		f = params.file_IO('../ref_seq/new_pep_against_human_viruses.tblastn.species.noseg.WS3.max_target_seqs100000.180625 (1).m8', '\t')
		new_aln = f.flat_file_to_df([0,1,15])
		
		orig_aln.index = [i.split('_')[1] for i in orig_aln.index]
		aln_df = pd.concat([orig_aln, new_aln])
		aln_df.fillna(0, inplace=True)
		
		binary_b = aln_df[aln_df>=80].fillna(0)
		binary_b = pd.DataFrame(index=binary_b.index, columns=binary_b.columns, data=binary_b.values, dtype=bool)
		binary_b = flex_array.array(binary_b).filter_aln(ref_seq=self.par['dir_ref_seq'])
		binary_b=binary_b.reindex(zdf.index).fillna(0)
		aln_df = aln_df.loc[:,binary_b.columns]
		
		#binary_b = flex_array.sparse_aln_df(self.par['file_aln'])
		#binary_b = flex_array.array(binary_b).filter_aln(ref_seq=self.par['dir_ref_seq'])
	
		sum_df = pd.DataFrame(0, index=list(binary_b), columns=list(zdf))
		glob_unique = pd.DataFrame(0, index=list(binary_b), columns=list(zdf))
		pep_df = pd.DataFrame(np.nan, index=list(binary_b), columns=list(zdf))
		p_df = pd.DataFrame(index=list(binary_b), columns=list(zdf))
		n_df = pd.DataFrame(index=list(binary_b), columns=list(zdf))
		padjust_df = pd.DataFrame(index=list(binary_b), columns=list(zdf))
		orig_p = pd.DataFrame(index=list(binary_b), columns=list(zdf))
		filter_df = pd.DataFrame(index=list(binary_b), columns=list(zdf))
		
		hits_series = pd.Series(index=list(zdf))
		nonoverlap_hits_series = pd.Series(index=list(zdf))
		samples = list(zdf.columns)
		
		nonoverlap_dict = {}
		
		parallel_dict1 = {}
		parallel_dict2 = {}
		
		for sample_name, column in zdf.iteritems():
			hits = column[column>=self.par['Z_threshold']].copy()
			if self.par['use_filter']:
				nonoverlap_hits = flex_array.gen_ind_hits(hits, self.dependent_pep, 
											  self.par['graph_dir'], samples.index(sample_name))
				input_num = len(nonoverlap_hits)
			elif not self.par['use_filter']:
				nonoverlap_hits = hits.copy()
				input_num = len(flex_array.gen_ind_hits(hits, self.dependent_pep, 
											  self.par['graph_dir'], samples.index(sample_name)))
			hits_series[sample_name] = len(hits)
			nonoverlap_hits_series[sample_name] = input_num
			nonoverlap_dict[sample_name] = list(nonoverlap_hits.index)
			print("%s:\thits=%s, nonoverlapped=%s" %(sample_name, len(hits), input_num))
			
			if input_num > 0:
				zb_df=aln_df.loc[nonoverlap_hits.index]
				parallel_dict1[sample_name] = zb_df
				parallel_dict2[sample_name] = nonoverlap_hits
				'''
				collapse_zb, glob_array, sim_tag, p_series, orig_pseries, filter_series = flex_array.array(zb_df).binom_reassign(
						nonoverlap_hits, self.dependent_pep, self.par['dir_ref_seq'], self.par['p_threshold'], self.par['x_threshold'], self.par['organism'])
				sum_df[sample_name]=collapse_zb.apply(sum, axis=0) + sim_tag
				glob_unique[sample_name] = glob_array.apply(sum, axis=0) + sim_tag
				pep_df[sample_name]=collapse_zb.apply(lambda x: flex_array.array(x).names_string(0.001),axis=0)
				p_df[sample_name]=p_series
				orig_p[sample_name]=orig_pseries
				filter_df[sample_name]=filter_series
				'''
		#parallel_dict1 = pd.Series(parallel_dict1)
		#parallel_dict2 = pd.Series(parallel_dict2)
		#parallel_dict2 = parallel_dict.loc[parallel_dict1.index]
		list1 = list(parallel_dict1.keys()) #sample names
		list2 = list(parallel_dict1.values()) #zb_df
		list3 = [parallel_dict2[i] for i in list1] #hits series
		zipped = zip(list2, list3, list1)
		
		results = Parallel(n_jobs=-1)(delayed(flex_array.binom_reassign)(zb_df, nonoverlap_hits, sample_name, self.dependent_pep, self.par['dir_ref_seq'], 
					 self.par['p_threshold'], self.par['x_threshold'], self.par['organism']) for zb_df, nonoverlap_hits, sample_name in zipped)
			
		r1, r2, r3, r4, r5, r6, r7, r8 = zip(*results)
		for i in range(len(r7)):
			sample_name=r7[i]; collapse_zb=r1[i]; glob_array=r2[i]; sim_tag=r3[i]; p_series=r4[i]; orig_pseries=r5[i]; filter_series=r6[i]; n_series=r8[i]
			sum_df[sample_name]=collapse_zb.apply(sum, axis=0) + sim_tag
			glob_unique[sample_name] = glob_array.apply(sum, axis=0) + sim_tag
			pep_df[sample_name]=collapse_zb.apply(lambda x: flex_array.array(x).names_string(0.001),axis=0)
			n_df[sample_name]=n_series
			p_df[sample_name]=p_series
			orig_p[sample_name]=orig_pseries
			filter_df[sample_name]=filter_series
		
		file_head = self.par['sub_dir'] + self.par['zscore_file'].split('/')[-1].split('.')[0] #Removes file path and extension
		if self.par['organism']:
			file_head += '_organism_'
		else:
			file_head += '_species_'
			
		#Write log file
		params.file_IO(self.par['sub_dir']+'parameters.log', sep='=').dict_to_file(self.par)
		
		#Write analysis files
		sum_df.to_csv(file_head+'total-counts.txt', sep='\t', header=True, index_label='Specie')
		glob_unique.to_csv(file_head+'unique-counts.txt', sep='\t', header=True, index_label='Specie')
		pep_df.to_csv(file_head+'peptides.txt', sep='\t', header=True, index_label='Specie')
		p_df.to_csv(file_head+'p-values.txt', sep='\t', header=True, index_label='Specie')
		orig_p.to_csv(file_head+'orig-p-values.txt', sep='\t', header=True, index_label='Specie')
		filter_df.to_csv(file_head+'virus-filter.txt', sep='\t', header=True, index_label='Specie')
		
		for i in p_df.columns:
			pvals=np.array(p_df[i].values)
			if not pd.isnull(pvals).all():
				mask = [j for j in np.where(np.isfinite(pvals))[0]]
				pval_corrected = np.empty(pvals.shape)
				pval_corrected.fill(np.nan)
				pval_corrected[mask] = multipletests(pvals[mask], method='fdr_bh')[1]
				padjust_df[i] = pval_corrected
		padjust_df.to_csv(file_head+'p-adjusted.txt', sep='\t', header=True, index_label='Specie')
		
		#Write independent peptides file
		f = open(self.par['sub_dir']+'independent_peptides.txt', 'w')
		for i in samples:
			f.write(i)
			for j in nonoverlap_dict[i]:
				f.write('\t' + str(j))
			f.write('\n')
		f.close()

		#Write summary file
		f = open(file_head+'results_summary.txt', 'w')
		f.write("Sample name\tVirus\tBH p-value\tRaw p-value\tOrig p-value\tAssigned counts\tFiltered Assigned Counts\t")
		f.write("Assigned peptides\tTotal significant peptides\tRanking N\tTotal sample hits\tTotal filtered sample hits\n")
		for i in samples:
			BH = padjust_df[i]
			BH = BH[BH < self.par['bh_threshold']]
			p_value = p_df[i]
			n_value = n_df[i]
			n_value = n_value[BH.index]
			p_value = p_value[BH.index]
			filter_value = filter_df[i]
			filter_value = filter_value[BH.index]
			orig_pvalue = orig_p[i]
			orig_pvalue = orig_pvalue[BH.index]
			counts = sum_df[i]
			counts = counts[BH.index]
			peptides = pep_df[i]
			peptides = peptides[BH.index]
			
			for j in BH.index:
					if filter_value[j] > self.par['x_threshold']:
						f.write(i+'\t')
						f.write(j+'\t'+str(BH[j])+'\t')
						f.write(str(p_value[j])+'\t'+str(orig_pvalue[j])+'\t')
						f.write(str(counts[j])+'\t'+str(filter_value[j])+'\t'+str(peptides[j])+'\t')
						#write number of peptides
						pep_set = set()
						for k in BH.index:
							pep_list = peptides[k].split(';')
							pep_set = pep_set.union(set(pep_list))
						f.write(str(len(pep_set))+'\t')
						f.write(str(n_value[j])+'\t')
						f.write(str(hits_series[i])+'\t'+str(nonoverlap_hits_series[i])+'\n')
		f.close()
		print("End of run.")
		return None
		#End
		