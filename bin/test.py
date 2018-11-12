# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 13:47:31 2018

@author: Sanjay
"""

import params
import pandas as pd

f = params.file_IO('pep_against_human_viruses.tblastn.species.noseg.WS3.max_target_seqs100000.180625 (1).m8', '\t')
orig_aln = f.flat_file_to_df([0,1,15])
f = params.file_IO('new_pep_against_human_viruses.tblastn.species.noseg.WS3.max_target_seqs100000.180625 (1).m8', '\t')
new_aln = f.flat_file_to_df([0,1,15])

orig_aln.index = [i.split('_')[1] for i in orig_aln.index]
aln_df = pd.concat([orig_aln, new_aln])
aln_df.fillna(0, inplace=True)
