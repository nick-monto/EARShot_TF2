import unittest,glob,os
import _pickle as pickle, numpy as np
from earshot.model import EARSHOT
from earshot.parameters import ModelParameters
from earshot.data import pad

class TestModel(unittest.TestCase):

    def setUp(self):
        # set up some simple training data - a lot of this is clunky and
        #   hard-coded here just to get the model to work
        self.acoustics = []
        self.semantics = []
        files = glob.glob('./tests/test-input/*.PICKLE')
        for f in files:
            d = pickle.load(open(f,'rb'))
            self.acoustics.append(d['Acoustic'])
            self.semantics.append(d['Semantic'])
        self.semantics = np.stack(self.semantics)
        # do the padding
        padded_acoustics = [pad(x,(54,256)) for x in self.acoustics]
        p = ModelParameters()
        self.es = EARSHOT(300,p)
        # now compile
        self.es.compile(loss=self.es.loss,optimizer="adam")

    def test_model_compile(self):
        # model should not be none after all that
        self.assertIsNotNone(self.es.compile)

    def test_model_size(self):
        # basic model should have a mask, one hidden layer, and a dense output
        self.assertEqual(len(self.es.layers),3)


    # this test should be removed.  Unfortunately, if we use the delayed build
    #   paradigm in keras, we can't introspect into the model anymore.
    '''
    def test_model_mask(self):
        # compile the model
        self.es.compile(loss=self.es.loss,optimizer="adam")
        # mask is defined in the masking layer, so it should have no input
        #   mask but should create the output mask
        self.assertIsNone(self.es.layers[1].input_mask)
        self.assertIsNotNone(self.es.layers[1].output_mask)
        # mask should be propagated through the input and output of all
        #   subsequent layers
        for i in range(2,len(self.es.layers)):
            self.assertIsNotNone(self.es.layers[i].input_mask)
            self.assertIsNotNone(self.es.layers[i].output_mask)
    '''
