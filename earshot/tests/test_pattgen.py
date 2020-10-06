import unittest,glob,os
from earshot.parameters import PatternParameters
from earshot.patterns import PatternGenerator

class TestPatternGenerator(unittest.TestCase):

    def setUp(self):
        # set up the parameters object and create the pattern generator
        self.p = PatternParameters(lexicon='tests/test-pron.txt',wav_path='tests/wavs',pattern_path='tests/test-pat')
        self.pg = PatternGenerator(self.p)

    def test_pattern_generator(self):
        self.pg.generate_patterns()
        # this should create five pickles in the tests/test-pat directory
        created_files = glob.glob('tests/test-pat/*.PICKLE')
        self.assertEqual(len(created_files),5)
        self.pg.generate_metadata()
        # adds an additional file called 'METADATA.PICKLE' to tests/test-pat directory
        self.assertTrue(os.path.exists('tests/test-pat/METADATA.PICKLE'))


if __name__ == '__main__':
    unittest.main()
