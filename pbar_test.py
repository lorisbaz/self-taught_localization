import unittest
import time

from vlg.util.pbar import *

class ProgressDotsTest(unittest.TestCase):
    def setUp(self):
        self.progress = ProgressBar.create(ProgressBarDotsParams(150))
        
    def tearDown(self):
        pass
    
    def test_next(self):
        for i in range(150):
            time.sleep(0.01)
            self.progress.next()
        self.progress.finish()

    def test_update(self):
        for i in range(150):
            time.sleep(0.01)
            self.progress.update(i+1)
        self.progress.finish()
                   
#=============================================================================

class ProgressBarPlusTest(unittest.TestCase):
    def setUp(self):
        self.progress = ProgressBar.create(ProgressBarPlusParams(150))
        
    def tearDown(self):
        pass
    
    def test_next(self):
        for i in range(150):
            time.sleep(0.01)
            self.progress.next()
        self.progress.finish()

    def test_update(self):
        for i in range(150):
            time.sleep(0.02)
            self.progress.update(i+1)
        self.progress.finish()         

#=============================================================================

if __name__ == '__main__':
    unittest.main()
    
