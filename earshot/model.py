from keras import Sequential
from keras.layers import Dense, Masking, LSTM


class Earshot(object):
    '''
    Class for EARShot model creation/compilation using keras.  Just a stub right
    now with some suggestions/pseudocode as to how to get this working.
    '''
    def __init__(self,pattern_parameters,model_parameters):
        self.model_parameters = model_parameters
        self.pattern_parameters = pattern_parameters
        self.earshot_model = Sequential()
        # this would assume we pad the spectrograms with zeros; we don't have to have
        #   done the padding at this point (we could do it as we stream the data),
        #   but we need to know how long the longest pattern is.
        #
        #   max_seq_len = max sequence length (time)
        #   spec_chan = # channels in the spectrogram (feature) (we typically use 256)

        #  dummy values to check that the mask propagates
        max_seq_len = 50
        spec_chan = 256

        #   tricky part with masking is making sure the mask propagates through
        #   the network; this becomes more problematic with deep networks
        #   using layers that change sizes of things, in which case you have to
        #   make sure the mask gets propagated manually - EARShot probably doesn't
        #   have this issue, since (without the prenet) it's just Input->LSTM->Output
        self.earshot_model.add(Masking(mask_value=0., input_shape=(max_seq_len,spec_chan)))
        if self.model_parameters.model_hidden['type'] == 'LSTM':
            self.earshot_model.add(LSTM(self.model_parameters.model_hidden['size'],return_sequences=True))
        self.earshot_model.add(Dense(self.pattern_parameters.semantic_dict['embedding_dim']))
        self.earshot_model.compile(loss='mean_squared_error',optimizer='adam')
