from numpy import ones

def pad(data_2d, ref_shape, pad_val=0.0):
    '''
    Pads a 2D matrix "data" with the pad_value to make it have shape ref_shape.
    All the padded values occur "off the end" of the data array, so that the
    original array occupies the upper left of the resulting matrix.
    '''
    padded_data = pad_val*ones(ref_shape)
    padded_data[:data_2d.shape[0],:data_2d.shape[1]] = data_2d
    return padded_data
