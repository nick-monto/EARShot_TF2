import unittest,glob,os
from earshot.parameters import PatternParameters
from earshot.patterns import PatternGenerator

class TestPatternGenerator(unittest.TestCase):

    def setUp(self):
        # if the tests/test-pat directory (dumping ground for test pattern) isn't present, it needs to be created
        if not os.path.exists('tests/test-pat'):
            os.mkdir('tests/test-pat')
        # set up the parameters object and create the pattern generator
        self.p = PatternParameters(lexicon='tests/test-pron.txt',wav_path='tests/wavs',pattern_path='tests/test-pat')
        self.pg = PatternGenerator(self.p)
        # if there are already pickles in the test-pat directory, get rid of them so the test
        #   can run successfully
        files_in_testpat = glob.glob('tests/test-pat/*.PICKLE')
        for f in files_in_testpat:
            os.remove(f)

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
