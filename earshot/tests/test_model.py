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


    def test_learning_schedule(self):
        pass

    def test_model_compile(self):
        # model should not be none after all that
        self.assertIsNotNone(self.es.compile)


    def test_model_size(self):
        # basic model should have a mask, one hidden layer, and a dense output
        self.assertEqual(len(self.es.layers),3)


    def test_model_mask(self):
        # return a shape-specified version of the model for introspection
        # mask is defined in the masking layer, so it should have no input
        #   mask but should create the output mask
        x = self.es.model((54,256))
        self.assertIsNone(x.layers[1].input_mask)
        self.assertIsNotNone(x.layers[1].output_mask)
        # mask should be propagated through the input and output of all
        #   subsequent layers
        for i in range(2,len(x.layers)):
            self.assertIsNotNone(x.layers[i].input_mask)
            self.assertIsNotNone(x.layers[i].output_mask)
