import attr,os

def file_not_found(instance,attribute,value):
    if not os.path.exists(value):
        raise ValueError('{} does not exist!'.format(attribute.name))


# if we wanted to be REALLY fussy, we could freeze the object in the decorator (@attr.s(frozen=True))
#   so that parameters cannot be changed after initialization (seems very non-pythonic though)
@attr.s
class PatternParameters(object):
    '''
    Class for holding and checking parameters relevant to pattern generation and properties.
    These parameters are used broadly across the EARSHOT module.
    '''
    lexicon = attr.ib(kw_only=True,validator=[attr.validators.instance_of(str),file_not_found])
    wav_path = attr.ib(kw_only=True,validator=[attr.validators.instance_of(str),file_not_found])
    pattern_path = attr.ib(kw_only=True,validator=[attr.validators.instance_of(str),file_not_found])
    metadata_file = attr.ib(kw_only=True,default='METADATA.PICKLE')
    acoustic_mode = attr.ib(kw_only=True,default='spectrogram',validator=attr.validators.in_(['spectrogram','mel']))
    spectrogram_dict = attr.ib(kw_only=True,default={'samp_rate':22050,'dimension':256,'frame_len':10,'frame_shift':10},validator=attr.validators.instance_of(dict))
    mel_dict = attr.ib(kw_only=True,default={'samp_rate':22050,'spec_dimension':1025,'dimension':80,'frame_len':10,'frame_shift':10,'max_abs':4},validator=attr.validators.instance_of(dict))
    semantic_mode = attr.ib(kw_only=True,default='SRV',validator=attr.validators.in_(['SRV','VEC']))
    semantic_dict = attr.ib(kw_only=True,default={'embedding_dim':300,'num_nonzero':30,'vec_file':None},validator=attr.validators.instance_of(dict))

@attr.s
class ModelParameters(object):
    '''
    Class for holding parameters relevant to building the EARShot network.
    '''
    hidden = attr.ib(kw_only=True,default={'type':'LSTM','size':512},validator=attr.validators.instance_of(dict))
    train_loss = attr.ib(kw_only=True,default='CE',validator=attr.validators.in_(['CE','MSE']))
    optimizer = attr.ib(kw_only=True,default={'ADAM':{'beta_1':0.9,'beta_2':0.999,'epsilon':1e-07}},validator=attr.validators.instance_of(dict))

@attr.s
class TrainingParameters(object):
    pass

@attr.s
class AnalyzerParameters(object):
    '''
    Class for holding/checking/verifying parameters relevant to analysis of EARShot model
    output.
    '''
    model_output_path = attr.ib(kw_only=True,validator=[attr.validators.instance_of(str),file_not_found])
    abs_acc_crit = attr.ib(kw_only=True,default=0.7)
    rel_acc_crit = attr.ib(kw_only=True,default=0.05)
    td_acc_crit = attr.ib(kw_only=True,default=(10,0.05),validator=attr.validators.instance_of(tuple))
    step_cut = attr.ib(kw_only=True,default=True,validator=attr.validators.instance_of(bool))
    batch_step = attr.ib(kw_only=True,default=200)
