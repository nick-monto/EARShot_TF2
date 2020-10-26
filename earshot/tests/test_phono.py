import unittest,glob,os
from earshot.phonology import *

class TestPhonology(unittest.TestCase):

    def setUp(self):
        # phonological transcriptions of a few words with known relationships
        # neighbors and rhymes but not cohorts
        self.fine = 'F.AY.N'.split('.')
        self.sine = 'S.AY.N'.split('.')
        # unrelated to the two above
        self.wife = 'W.AY.F'.split('.')
        # cohorts but not neigbhors
        self.stout = 'S.T.AW.T'.split('.')
        self.strand = 'S.T.R.AE.N.D'.split('.')
        # neighbors but nothing else
        self.tape = 'T.EY.P'.split('.')
        self.type = 'T.AY.P'.split('.')

    def test_phono_funcs(self):
        # 'fine' and 'sine' are rhymes and neighbors but not cohorts
        self.assertTrue(are_rhymes(self.fine,self.sine))
        self.assertTrue(are_neighbors(self.fine,self.sine))
        self.assertFalse(are_cohorts(self.fine,self.sine))
        # 'wife' and 'sine' are unrelated
        self.assertFalse(are_rhymes(self.wife,self.sine))
        self.assertFalse(are_neighbors(self.wife,self.sine))
        self.assertFalse(are_cohorts(self.wife,self.sine))
        # 'stout' and 'strand' are cohorts, not rhymes, not neighbors
        self.assertFalse(are_rhymes(self.stout,self.strand))
        self.assertFalse(are_neighbors(self.stout,self.strand))
        self.assertTrue(are_cohorts(self.stout,self.strand))
        # 'tape' and 'type' are neighbors but nothing else
        self.assertFalse(are_rhymes(self.tape,self.type))
        self.assertTrue(are_neighbors(self.tape,self.type))
        self.assertFalse(are_cohorts(self.tape,self.type))
        # finally, if you send in the same word twice all should fail
        self.assertFalse(are_rhymes(self.tape,self.tape))
        self.assertFalse(are_neighbors(self.tape,self.tape))
        self.assertFalse(are_cohorts(self.tape,self.tape))

if __name__ == '__main__':
    unittest.main()
