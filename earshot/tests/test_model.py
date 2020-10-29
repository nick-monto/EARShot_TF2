import unittest,glob,os
from earshot.model import Earshot

class TestModel(unittest.TestCase):

    def setUp(self):
        # model with dummy sizing
        self.es = Earshot((10,10),10,10)

    def test_model_mask(self):
        # mask is defined in the masking layer, so it should have no input
        #   mask but should create the output mask
        self.assertIsNone(self.es.layers[1].input_mask)
        self.assertIsNotNone(self.es.layers[1].output_mask)
        # mask should be propagated through the input and output of all
        #   subsequent layers
        for i in range(2,len(self.es.layers)):
            self.assertIsNotNone(self.es.layers[i].input_mask)
            self.assertIsNotNone(self.es.layers[i].output_mask)

if __name__ == '__main__':
    unittest.main()
