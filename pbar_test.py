import unittest
import time

import pbar

class BBoxTest(unittest.TestCase):
    def setUp(self):
        self.progress = pbar.ProgressBar(150)
        
    def tearDown(self):
        pass
    
    def test_next(self):
        for i in range(150):
            time.sleep(0.02)
            self.progress.next()

    def test_update(self):
        for i in range(150):
            time.sleep(0.04)
            self.progress.update(i+1)            

#=============================================================================

if __name__ == '__main__':
    unittest.main()
    
