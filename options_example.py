'''
Here's an example of an options object (basically a glorified dict, probably should be derive from dict)
to hold a bunch of options for code that required them. This defaults everything that can possibly be
defaulted to reasonable values and hides them (would be helpful for EARSHOT).  Long-term, I was going to
include some checking that would prevent you from setting options that are incompatible - also likely
helpful for EARSHOT (did not get around to this). (KB,9/25/20)
'''

class CEPParameters(object):
    '''
    Container class to hold the myriad set of parameters for the processing pipeline,
    and to do checking for inconsistent options.
    '''
    def __init__(self,main_directory,alignment_file,canon_sequence):
        '''
        Only three absoultely necessary parameters are:
            main_directory : base project directory
            alignment_file : master alignment file (FASTA or STOCKHOLM)
            canon_sequence : canonical sequence for position numbering

        Default parameters and their values:
            pdb_file          : None (no accuracy will be calculated)
            fig_directory     : 'figures/'
            file_indicator    : 'cep'
            resampling_method : 'splithalf' (split-half resampling)
            num_partitions    : 100 (100 half-splits)
            pc_type           : 'inc' (pseudocount value scales with n. of seqs.)
            pc_lambda         : 1.0 (only used if pc_type = 'fix')
            pc_mix            : mixture parameter for Preal/Ppseudo (only for pc_type = 'inc')
            gap               : 1.0 (score all columns, no matter how many gaps)
            swt_method        : 'henikoff'
            cutoff            : groups of sequences with fractional_similarity > cutoff share unit weight
                    (not used for henikoff weighting or no weighting)
            pruning           : 'mst' (how to prune dense graphs for consensus graph calculation)
            number            : 100 (the n used for 'topn' or 'bottomn' pruning)
            acc_method        : 'avgdist' (use the original scaled average distance definition
                    of Brown and Brown)
            sim_method        : 'spearman' (how to quantify similarity between split graphs)
        '''
        # parameters passed in
        self.main_directory = main_directory
        self.alignment_file = alignment_file
        self.canon_sequence = canon_sequence
        # defaults for others
        self.fig_directory = 'figures/'
        self.pdb_file = None
        self.file_indicator = 'cep'
        self.resampling_method = 'splithalf'
        self.num_partitions = 100
        self.pc_type = 'inc'
        self.pc_lambda = 1.0
        self.pc_mix = 0.1
        self.gap = 1.0
        self.swt_method = 'henikoff'
        self.cutoff = 0.68
        self.pruning = 'mst'
        self.number = 100
        self.acc_method = 'avgdist'
        self.sim_method = 'spearman'


    def set_parameters(self,**kwargs):
        '''
        Parameters can be set one at a time directly, but this wrapper just cleans
        that up by allowing multiple kwargs to be passed in at one time.
        '''
        for k in kwargs:
            setattr(self,k,kwargs[k])


    def __str__(self):
        '''
        Pretty-print string representation of the options
        '''
        str_rep = 'Correlated Substitutions Pipeline Options:\n'
        for i in inspect.getmembers(self):
            if not i[0].startswith('_'):
                if not inspect.ismethod(i[1]):
                    str_rep += '\n\t'+i[0]+' : '+str(i[1])
        return str_rep
