#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 10:48:36 2024

@author: kate
"""
from Bio import SeqIO
import numpy as np
import matplotlib.pyplot as plt
import math
import time
import argparse
import pathlib
import random


#add in all options
parser = argparse.ArgumentParser(
    prog='combinatorics',
    description='Simulation of recombination of IGHV gene')

parser.add_argument("-vf", "--vfile",nargs="?",default="input/all/IGHV_all.txt",
                    help="input fasta file of v-region")
parser.add_argument("-df","--dfile",nargs="?",default="input/all/IGHD_all.txt",
                    help="input fasta file of d-region")
parser.add_argument("-jf","--jfile",nargs="?",default="input/all/IGHJ_all.txt",
                    help="input fasta file of j-region")
parser.add_argument("-vtf","--vtruncfivep", nargs="?",type=float, default=0,
                    help="poisson probability for v five prime truncation")
parser.add_argument("-vtt","--vtruncthreep", nargs="?",type=float, default=1.2,
                    help="poisson probability for v three prime truncation")
parser.add_argument("-dtf","--dtruncfivep", nargs="?",type=float, default=4.4,
                    help="poisson probability for d five prime truncation")
parser.add_argument("-dtt","--dtruncthreep", nargs="?",type=float, default=5,
                    help="poisson probability for d three prime truncation")
parser.add_argument("-jtf","--jtruncfivep", nargs="?",type=float, default=4.7,
                    help="poisson probability for j five prime truncation")
parser.add_argument("-jtt","--jtruncthreep", nargs="?",type=float, default=0,
                    help="poisson probability for j three prime truncation")
parser.add_argument("-pid","--percentId",nargs="?",type=float, default=0.9,
                    help="identity percentage for v-region hypermutation")
parser.add_argument("-s","--seed",type=int,default=np.random.default_rng(),
                    help="input seed for rng")
parser.add_argument("-pp","--pprob",type=int,default=1,
                    help="poisson probability for adding p nucleotides")
parser.add_argument("-np","--nprob",type=int,default=10,
                    help="poisson probability for adding n nucleotides")
parser.add_argument("-rl","--readlength",type=int,default=100,
                    help="length to cut reads")
parser.add_argument("-rb","--readbuffer",type=int,default=20,
                    help="length of buffer around junction (minimum bp to keep on either side of junction while cutting)")
parser.add_argument("-o","--output",nargs="?",default="output",
                    help="output directory")
parser.add_argument("-n","--num_permutations",nargs="?",default="all",
                    help="specify the number of permutations to allow, if different from all")
parser.add_argument("-os","--output_separate",default="True",
                    help="extra output with v,d,j separate")
parser.add_argument("-oc","--output_cut",default="True",
                    help="extra output with v,d,j separate truncated")


args = parser.parse_args()

#input files
VFILE=args.vfile
DFILE=args.dfile
JFILE=args.jfile

#truncation options (probability)
V_TRUNC={"fp":args.vtruncfivep,
         "tp":args.vtruncthreep}
D_TRUNC={"fp":args.dtruncfivep,
         "tp":args.dtruncthreep}
J_TRUNC={"fp":args.jtruncfivep,
         "tp":args.jtruncthreep}

#option for percent identity
P_ID=args.percentId

#option to input seed 
rng = args.seed

#option for p and n nucleotides
P_PROB_POISSON=args.pprob
N_PROB_POISSON=args.nprob

#option for cutting sequences in NGS-length pieces
READ_LENGTH=args.readlength
READ_BUFFER=args.readbuffer

#option for optional outputs (os and oc)
OUTPUT_S=bool(args.output_separate)
OUTPUT_C=bool(args.output_cut)

#option for output file
OUTPUT_DIR=args.output
OUT_FILE=OUTPUT_DIR+"/combinatorics_output.txt"
if OUTPUT_S:
    OSV_FILE=OUTPUT_DIR+"/full_recombined_v.txt"
    OSD_FILE=OUTPUT_DIR+"/full_recombined_d.txt"
    OSJ_FILE=OUTPUT_DIR+"/full_recombined_j.txt"
if OUTPUT_C:
    OCV_FILE=OUTPUT_DIR+"/cut_recombined_v.txt"
    OCD_FILE=OUTPUT_DIR+"/cut_recombined_d.txt"
    OCJ_FILE=OUTPUT_DIR+"/cut_recombined_j.txt"

#create output paths if they don't exist
pathlib.Path(OUTPUT_DIR+"/logs").mkdir(parents=True, exist_ok=True)
pathlib.Path(OUTPUT_DIR+"/plots").mkdir(parents=True, exist_ok=True)

#option for number of permutations
if (args.num_permutations) == "all":
    NUM_PERMUTATIONS=args.num_permutations
else:
    NUM_PERMUTATIONS=int(args.num_permutations)

#functions
#find_all(str,str)->(list)
def find_all(string, substring):
    start=0
    indexes=[]
    while True:
        start = string.find(substring, start)
        indexes.append(start)
        if start == -1: 
            indexes=indexes[:len(indexes)-1]
            return indexes
        start += len(substring) # use start += 1 to find overlapping matches
    
#extract_cdr_coords(seqIO)->(list)
def extract_cdr_coords(v_fasta):
    cdr_coords={}
    for regions in v_fasta:
        if "CDR" in regions.description:
            pipe_indexes=(find_all(regions.description,"|"))
            
            #extract the name and allele of the gene (first pipe to second pipe)
            name=(regions.description[pipe_indexes[0]+1:pipe_indexes[1]])
            
            name+="-"+regions.description[pipe_indexes[3]+1:pipe_indexes[3]+5]
            
            #extract the span of cdr region (5th pipe to first period)
            span=(regions.description[pipe_indexes[4]+1:pipe_indexes[5]])
            
            #from span : extract the start of cdr region (start of span to first period)
            start=span[:span.find(".")]
            
            cdr_coords[name]=start
    return cdr_coords

#extract_permutations(list,list,list)->(list,list,list,list)
def extract_permutations(v_fasta,d_fasta,j_fasta):
    v_genes=[]
    d_genes=[]
    j_genes=[] 
    sequence_names=[]
    for v in v_fasta:
        curr_v_gene=v.description[(v.description).index("IGH"):(v.description).index("*")+3]
        for d in d_fasta:
            curr_d_gene=d.description[(d.description).index("IGH"):(d.description).index("*")+3]
            for j in j_fasta:
                curr_j_gene=j.description[(j.description).index("IGH"):(j.description).index("*")+3]
                #in which all first indexes match together, second, etc.
                sequence_names.append(curr_v_gene+"_"+curr_d_gene+"_"+curr_j_gene)
                v_genes.append(v.seq)
                d_genes.append(d.seq)
                j_genes.append(j.seq)
    return sequence_names,v_genes,d_genes,j_genes

#delete extra permutations for a smaller dataset
#delete_permutations(list,list,list,list,int/string)->(list,list,list,list)
def delete_permutations(sequence_names,v_genes,d_genes,j_genes,NUM_PERMUTATIONS):
    
    #lists for permutations to keep
    new_sequence_names=[]
    new_v_genes=[]
    new_d_genes=[]
    new_j_genes=[]
    
    #if we want to keep all, just return the whole lists back
    if NUM_PERMUTATIONS=="all":
        return sequence_names,v_genes,d_genes,j_genes
    
    #else, randomly sample the number of permutations we want to keep
    else:
        keep_indexes=random.sample((range(0,len(sequence_names))),NUM_PERMUTATIONS)
        
        #add all indexes to keep in the new lists, and then return them
        for indexes in keep_indexes:
            new_sequence_names.append(sequence_names[indexes])
            new_v_genes.append(v_genes[indexes])
            new_d_genes.append(d_genes[indexes])
            new_j_genes.append(j_genes[indexes])
    return new_sequence_names,new_v_genes,new_d_genes,new_j_genes
    
#poisson_prob(dict)->(dict)
def poisson_prob(lam):
    return rng.poisson(lam)

#truncate(Bio.SeqRecord,dict)->seqIO
def truncate(sequence,trunc):
    #determine how much we're truncating using poisson distribution
    n_nuc={}
    n_nuc["fp"]=poisson_prob(trunc["fp"])
    n_nuc["tp"]=poisson_prob(trunc["tp"])
    
    #five prime truncation
    sequence=sequence[n_nuc["fp"]:]
    #three prime truncation
    sequence=sequence[:(len(sequence)-n_nuc["tp"])]
    
    return sequence,n_nuc["fp"],n_nuc["tp"]
    
#check if the sequence is still productive
#find reading frame (isolate ATG)
#is_productive(str)->(bool)
def is_productive(sequence):
    STOP_CODONS=["tta","tag","tga"]
    
    #checks all codons and returns false if it's a STOP codon
    for i in range(0,len(sequence)-2,3):
        if sequence[i:i+3] in STOP_CODONS:
            return False
    return True

#hypermutation of v sequence
#hypermutation(str,float)->(str,list,list,list)
def hypermutation(sequence,p_id):
    log_mut_pos=[]
    log_mut_base=[]
    log_mut_base_curr=[]
    #check for STOP after each mutation, and reverse one step if there is one
    #can't mutate same base twice
    
    #didn't do normal distribution but could
    #not inserting a seed?
    
    sequence_list=[char for char in sequence]
    
    bases=["a","c","t","g"]
    
    n_mutations=np.random.randint(0,round((1-p_id)*len(sequence_list))+1)
    
    #starts at 3 so that we don't modify the first atg (don't care about others)
    unmut_pos=list(range(3,len(sequence_list)))
    i=0
    while(i<n_mutations):
        new_seq=sequence_list
        
        #pick random index from list containing all unmutated positions
        mut_index=np.random.randint(low=0,high=len(unmut_pos))
        mut_pos=unmut_pos[mut_index]
        
        #check if position is an n, if so start loop again
        if sequence_list[mut_pos]=="n":
            continue
        
        #pick random base 
        #start from 1 so that it doesn't get mutated into the same base
        #max 3+1
        mut_base_add=np.random.randint(low=1,high=4)
        mut_base_curr=sequence_list[mut_pos]
        mut_base=bases[(bases.index(mut_base_curr)+mut_base_add)%4]
        
        #change base
        new_seq[mut_pos]=mut_base
        
        #only accepts mutation if resulting sequence is still productive
        if is_productive(new_seq):
            i+=1 #only increments if the new seq is accepted
            sequence_list=new_seq
            unmut_pos.pop(mut_index)
            log_mut_pos.append(mut_pos)
            log_mut_base_curr.append(mut_base_curr)
            log_mut_base.append(mut_base)
            

    return "".join(sequence_list), log_mut_pos, log_mut_base_curr, log_mut_base
        
#add_nuc(str,int)->(str,list,list,list)
def add_nuc(d_seq,add_prob):
    log_bases_added=[]
    log_num_bases_added=[]
    bases=["a","c","t","g"]
    fp_add_seq=""
    tp_add_seq=""
    n_add={}
    
    n_add["fp"]=poisson_prob(add_prob)
    n_add["tp"]=poisson_prob(add_prob)
    
    log_num_bases_added.append(n_add["fp"])
    log_num_bases_added.append(n_add["tp"])
    
    for i in range(n_add["fp"]):
        base_added=bases[np.random.randint(low=0,high=4)]
        fp_add_seq+=base_added
        log_bases_added.append(base_added)
    
    for i in range(n_add["tp"]):
        base_added=bases[np.random.randint(low=0,high=4)]
        tp_add_seq+=base_added
        log_bases_added.append(base_added)
        
    d_seq=tp_add_seq+d_seq+fp_add_seq
    return d_seq,log_bases_added,log_num_bases_added
 
#cut_at_junction(int,str)->(str,int,int)   
def cut_at_junction(junction,sequence):
    #set lowcut and highcut and check after if it goes over the bounds of the gene
    low_cut=junction-(READ_LENGTH-READ_BUFFER)
    high_cut=junction-READ_BUFFER
    
    if low_cut<0:
        low_cut=0
    elif high_cut>len(sequence)-READ_LENGTH:
        high_cut=len(sequence)-READ_LENGTH
    
    #in case the setting of low_cut and high_cut to 0 or seq length 
    #goes over the other one
    if high_cut<low_cut:
        return 0
    
    #cut with a random number in between lowcut and highcut
    cut_fp=np.random.randint(low_cut,high_cut+1)
    cut_tp=cut_fp+READ_LENGTH
    cut_sequence=sequence[cut_fp:cut_tp]
    
    return cut_sequence,cut_fp,len(cut_sequence)

#make_barplot(list,str)->void
def make_barplot(log,title):
    #only use number of bins = number of unique items if i
    #it's smaller than 11 (too many bins) or bigger than 1 (too little bins)
    unique_logs,unique_counts=np.unique(log,return_counts=True)
    with open(OUTPUT_DIR+"/logs/"+title+".txt","w+") as f:
        for logs,counts in zip(unique_logs, unique_counts):
            f.write(str(logs))
            f.write(",")
            f.write(str(counts))
            f.write("\n")
    if len(set(log))==1:
        plt.hist(log,bins=5,color="lightgrey",histtype="bar",fill=True,ec="black")
    elif len(set(log))<20:
        plt.bar(*np.unique(log, return_counts=True),color="lightgrey",ec="black")
    else:
        plt.hist(log,bins=range(min(log),max(log),5),color="lightgrey",histtype="bar",fill=True,ec="black")
    plt.title(title)
    plt.ylabel('Frequency')
    plt.savefig(OUTPUT_DIR+"/plots/"+title+".svg",format='svg', dpi=1200)
    plt.close()

#main
def main():
    start_time=time.time()
    
    #input
    v_fasta = list(SeqIO.parse(VFILE, "fasta"))
    d_fasta = list(SeqIO.parse(DFILE, "fasta"))
    j_fasta = list(SeqIO.parse(JFILE, "fasta"))
    
    #1. extract cdr region coordinates (if any)
    cdr_coords=extract_cdr_coords(v_fasta)

    #2. get all permutations
    sequence_names,v_genes,d_genes,j_genes=extract_permutations(v_fasta, d_fasta, j_fasta)
    
    #2.5 only keep a set number of permutations (num_permutations) chosen randomly
    #done to generate a test set (smaller)
    sequence_names,v_genes,d_genes,j_genes=delete_permutations(sequence_names,v_genes,d_genes,j_genes,NUM_PERMUTATIONS)
    
    
    #3. truncation of v genes
    log_fp=[]
    log_tp=[]
    for counter, v_seq in enumerate(v_genes):
        v_genes[counter], fp_trunc, tp_trunc=truncate(v_seq,V_TRUNC)
        log_fp.append(fp_trunc)
        log_tp.append(tp_trunc)
    make_barplot(log_fp,"v five prime truncation")
    make_barplot(log_tp,"v three prime truncation")
    
    #3. truncation of d genes
    log_fp=[]
    log_tp=[]
    for counter, d_seq in enumerate(d_genes):  
        d_genes[counter], fp_trunc, tp_trunc=truncate(d_seq,D_TRUNC)
        log_fp.append(fp_trunc)
        log_tp.append(tp_trunc)
    make_barplot(log_fp,"d five prime truncation")
    make_barplot(log_tp,"d three prime truncation")
    
    #3. truncation of j genes
    log_fp=[]
    log_tp=[]
    for counter, j_seq in enumerate(j_genes):
        j_genes[counter],fp_trunc, tp_trunc=truncate(j_seq,J_TRUNC)
        log_fp.append(fp_trunc)
        log_tp.append(tp_trunc)
    make_barplot(log_fp,"j five prime truncation")
    make_barplot(log_tp,"j three prime truncation")
    
    #4. hypermutation of v gene
    log_mut_pos=[]
    log_mut_base_curr=[]
    log_mut_base=[]
    for counter,v_seq in enumerate(v_genes):
        v_genes[counter],mut_pos, mut_base_curr, mut_base=hypermutation(v_seq,P_ID)
        log_mut_pos+=(mut_pos)
        log_mut_base_curr+=(mut_base_curr)
        log_mut_base+=(mut_base)
    make_barplot(log_mut_pos,"mutated positions")
    make_barplot(log_mut_base_curr,"mutated bases")
    make_barplot(log_mut_base,"mutated to what base")
    
    
    #5. add p and n nucleotides around d sequence
    #adds p then n nucleotides to d sequence (five prime then three prime)
    #can add 0 nucleotides
    log_p_bases_added=[]
    log_n_bases_added=[]
    log_p_num_bases_added=[]
    log_n_num_bases_added=[]
    for counter, d_seq in enumerate(d_genes):
        d_genes[counter],p_bases_added,p_num_bases_added=add_nuc(d_seq, P_PROB_POISSON)
        d_seq=d_genes[counter]
        d_genes[counter],n_bases_added,n_num_bases_added=add_nuc(d_seq,N_PROB_POISSON)
        log_p_bases_added+=(p_bases_added)
        log_n_bases_added+=(n_bases_added)
        log_p_num_bases_added+=(p_num_bases_added)
        log_n_num_bases_added+=(n_num_bases_added)
        
    make_barplot(log_p_bases_added,"p bases added")
    make_barplot(log_n_bases_added,"n bases added")
    make_barplot(log_p_num_bases_added,"p number of bases added")
    make_barplot(log_n_num_bases_added,"n number of bases added")
    
    #5.5 Print an optional output : v,d,j sequences without being cut
    if OUTPUT_S:
        with open(OSV_FILE, "w") as f:
            for index,sequences in enumerate(v_genes):
                f.write(">"+sequence_names[index]+"_v_full_TP"+str(index)+"\n")
                f.write(str(sequences)+"\n")
        with open(OSD_FILE,"w") as f:
            for index,sequences in enumerate(d_genes):
                f.write(">"+sequence_names[index]+"_d_full_TP"+str(index)+"\n")
                f.write(str(sequences)+"\n")
        with open(OSJ_FILE,"w") as f:
            for index,sequences in enumerate(j_genes):
                f.write(">"+sequence_names[index]+"_j_full_TP"+str(index)+"\n")
                f.write(str(sequences)+"\n") 
    
    #6. Combine all sequences into one gene (keeping coords for all junctions)
    recomb_seq=[]
    vd_junction=[]
    dj_junction=[]
    length_genes=[]
    for v,d,j in zip(v_genes,d_genes,j_genes):
        whole_gene=v+d+j
        recomb_seq.append(whole_gene)
        vd_junction.append(len(v))
        dj_junction.append(len(v)+len(d))
        length_genes.append(len(whole_gene))
    
    #7. Cut sequences into READ_LENGTH sizes (emulating NGS)
    log_cut_fp=[]
    log_len_cut_seq=[]
    cut_sequences_vd=[]
    cut_start_index_vd=[]
    for index,junctions in enumerate(vd_junction):
        curr_seq=recomb_seq[index]
        cut_seq,fp_ext,len_cut_seq=cut_at_junction(junctions,curr_seq)
        cut_sequences_vd.append(cut_seq)
        cut_start_index_vd.append(fp_ext)
        log_cut_fp.append(fp_ext)
        log_len_cut_seq.append(len_cut_seq)

    cut_sequences_dj=[]
    cut_start_index_dj=[]
    for index,junctions in enumerate(dj_junction):
        curr_seq=recomb_seq[index]
        cut_seq,fp_ext,len_cut_seq=cut_at_junction(junctions,curr_seq)
        cut_sequences_dj.append(cut_seq)
        cut_start_index_dj.append(fp_ext)
        log_cut_fp.append(fp_ext)
        log_len_cut_seq.append(len_cut_seq)
    
    make_barplot(log_cut_fp,"cut fp")
    make_barplot(log_len_cut_seq,"length cut sequences")
    
    #8.5 Print an optional output : separate v,d,j sequences that are cut
    if OUTPUT_C:
        #for sequences that were cut around the vd junction (cut_sequences_vd)
        for i, read in enumerate(cut_sequences_vd):
            is_j_seq=False
            cut_v_seq=""
            cut_d_seq=""
            cut_j_seq=""
            with open(OCV_FILE,"a") as f:
                f.write(">"+sequence_names[i]+"_vd_v_TP"+str(i)+"\n")
                cut_v_seq=str(read[:vd_junction[i]-cut_start_index_vd[i]])
                f.write(cut_v_seq+"\n")
            with open(OCD_FILE,"a") as f:
                f.write(">"+sequence_names[i]+"_vd_d_TP"+str(i)+"\n")
                #if dj junction isn't in read
                if len(read) <= dj_junction[i]-cut_start_index_vd[i]:
                    cut_d_seq=str(read[vd_junction[i]-cut_start_index_vd[i]:])
                    
                #if dj junction is in read
                else:
                    cut_d_seq=str(read[vd_junction[i]-cut_start_index_vd[i]:dj_junction[i]-cut_start_index_vd[i]])
                    is_j_seq=True
                f.write(cut_d_seq+"\n")
            if is_j_seq:
                with open(OCJ_FILE,"a") as f:
                    f.write(">"+sequence_names[i]+"_vd_j_TP"+str(i)+"\n")
                    cut_j_seq=str(read[dj_junction[i]-cut_start_index_vd[i]:])
                    f.write(cut_j_seq+"\n")
            if str(cut_sequences_vd[i])!=str(cut_v_seq)+str(cut_d_seq)+str(cut_j_seq):
                print("oops1")
                
        #for sequences that were cut around the dj junction (cut_sequences_dj)            
        for i, read in enumerate(cut_sequences_dj):
            is_v_seq=False
            cut_v_seq=""
            cut_d_seq=""
            cut_j_seq=""
            with open(OCJ_FILE,"a") as f:
                f.write(">"+sequence_names[i]+"_dj_j_TP"+str(i)+"\n")
                cut_j_seq=str(read[dj_junction[i]-cut_start_index_dj[i]:])
                f.write(cut_j_seq+"\n")
            with open(OCD_FILE,"a") as f:
                f.write(">"+sequence_names[i]+"_dj_d_TP"+str(i)+"\n")
                #if vd junction isn't in read
                if vd_junction[i]<cut_start_index_dj[i]:
                    cut_d_seq=str(read[:dj_junction[i]-cut_start_index_dj[i]])
                #if vd junction is in read
                else:
                    cut_d_seq=str(read[vd_junction[i]-cut_start_index_dj[i]:dj_junction[i]-cut_start_index_dj[i]])
                    is_v_seq=True
                f.write(cut_d_seq+"\n")
            if is_v_seq:
                with open(OCV_FILE,"a") as f:
                    f.write(">"+sequence_names[i]+"_dj_v_TP"+str(i)+"\n")
                    cut_v_seq=str(read[:vd_junction[i]-cut_start_index_dj[i]])
                    f.write(cut_v_seq+"\n")

    
    #8. put all cut sequences into fasta file
    with open(OUT_FILE, 'w') as f:
        for index,sequences in enumerate(cut_sequences_vd):
            f.write(">"+sequence_names[index]+"_vd_TP"+str(index)+"\n")
            f.write(str(cut_sequences_vd[index])+"\n")
        for index,sequences in enumerate(cut_sequences_dj):
            f.write(">"+sequence_names[index]+"_dj_TP"+str(index)+"\n")
            f.write(str(cut_sequences_dj[index])+"\n") 
            
            
    end_time=time.time()  
    print("Time taken : ", end_time-start_time)
        
if __name__=="__main__":
    main()
    
    
    
    
    
    
    
    