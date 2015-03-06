"""
Unit tests for the module vlg.util.parfun.
"""

import os
import time
import unittest

import parfun
import pbar

def sample_function(a, b):
    time.sleep(0.5)
    return a*b

#=============================================================================

class TestParFunDummy(unittest.TestCase):
    def setUp(self):
        self.parfun = parfun.ParFunDummy(sample_function)

    def tearDown(self):
        self.parfun = None

    def test_run(self):
        self.parfun.add_task(2, 3)
        self.parfun.add_task(2, 2)
        self.parfun.add_task(4, 5)
        out = self.parfun.run()
        self.assertEqual(out[0], 6)
        self.assertEqual(out[1], 4)
        self.assertEqual(out[2], 20)

#=============================================================================

class TestParFunProcesses(unittest.TestCase):
    def setUp(self):
        # We ask for 2 cores
        progress_bar_params = pbar.ProgressBarDotsParams()
        self.parfun = parfun.ParFunProcesses(\
            sample_function, num_processes=2,
            progress_bar_params=progress_bar_params)

    def tearDown(self):
        self.parfun = None

    def test_run(self):
        self.parfun.add_task(2, 3)
        self.parfun.add_task(2, 2)
        self.parfun.add_task(4, 5)
        out = self.parfun.run()
        self.assertEqual(out[0], 6)
        self.assertEqual(out[1], 4)
        self.assertEqual(out[2], 20)

#=============================================================================

@unittest.skipIf(not parfun.ParFunAnthill.is_current_host_supported(),
                'Skipping TestParFunAnthill because the host is not supported')
class TestParFunAnthill(unittest.TestCase):
    def setUp(self):
        progress_bar_params = pbar.ProgressBarPlusParams()
        self.parfun = parfun.ParFunAnthill( \
            sample_function, \
            progress_bar_params=progress_bar_params, max_tasks=2)

    def tearDown(self):
        self.parfun = None

    def test_run1(self):
        self.parfun.add_task(2, 3)
        self.parfun.add_task(2, 2)
        self.parfun.add_task(4, 5)
        out = self.parfun.run()
        self.assertEqual(out[0], 6)
        self.assertEqual(out[1], 4)
        self.assertEqual(out[2], 20)

    def test_run2(self):
        # We launch the paralellization as before
        self.parfun.add_task(2, 3)
        self.parfun.add_task(2, 2)
        self.parfun.add_task(4, 5)
        out = self.parfun.run()
        self.assertEqual(out[0], 6)
        self.assertEqual(out[1], 4)
        self.assertEqual(out[2], 20)
        # Let's suppose test_run1 failed because of some error in the cluster.
        # We can re-launch the same computation while computing only the
        # remaining results by invoking parfun exactly as before BUT
        # specifying the job_name, by either using the method set_jobname:
        self.parfun.set_jobname(self.parfun.get_jobname())
        # .... or by creating a new object:
        self.parfun = parfun.ParFunAnthill(sample_function, \
                                        job_name = self.parfun.get_jobname(), \
                                        max_tasks=2) # IMPORTANT: cannot change
        self.parfun.add_task(2, 3)
        self.parfun.add_task(2, 2)
        self.parfun.add_task(4, 5)
        out = self.parfun.run()
        self.assertEqual(out[0], 6)
        self.assertEqual(out[1], 4)
        self.assertEqual(out[2], 20)

#=============================================================================

if __name__ == '__main__':
    unittest.main()
