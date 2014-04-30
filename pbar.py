import sys
import time

import progressbar

class ProgressBar:
    """ Simple text-based ProgressBar """

    def __init__(self, max_val):
        raise NotImplementedError()

    def set_max_val(self, max_val=100):
        self.max_val = max_val
    
    def next(self, n_steps=1):
        raise NotImplementedError()
    
    def update(self, val):
        raise NotImplementedError()

    def finish(self):
        raise NotImplementedError()

    @staticmethod
    def create(name, max_val=100):
        if name == 'ProgressBarDots':
            return ProgressBarDots(max_val)
        elif name == 'ProgressBarPlus':
            return ProgressBarPlus(max_val)
        else:
            raise ValueError('name not recognized')

#=============================================================================

class ProgressBarDots(ProgressBar):
    """ Simple text-based ProgressBar """

    def __init__(self, max_val):
        self.max_val = max_val
        self.iter = 0

    def next(self, n_steps=1):
        self.iter += 1
        self.progress.update(self.iter)
    
    def update(self, val):
        sys.stdout.write('.')

    def finish(self):
        print ''

#=============================================================================
                        
# NOTE:
# This implementation uses the library 'progressbar':
#   pip install progressbar
# For the list of widgets, look:
# http://code.google.com/p/python-progressbar/source/browse/progressbar/widgets.py
class ProgressBarPlus(ProgressBar):
    """ Simple text-based ProgressBar """

    def __init__(self, max_val):
        widgets = [progressbar.Percentage(), ' ', \
                   progressbar.Bar(), ' ', \
                   progressbar.Counter(), ' ', \
                   progressbar.Timer(), ' ', \
                   progressbar.ETA()]        
        self.max_val = max_val
        self.iter = 0
        self.progress = progressbar.ProgressBar( widgets=widgets, \
                            maxval=max_val)
        self.progress.start()

    def next(self, n_steps=1):
        self.iter += 1
        self.progress.update(self.iter)
    
    def update(self, val):
        self.progress.update(val)

    def finish(self):
        self.progress.finish()
        
