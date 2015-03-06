import sys
import time

import progressbar

class ProgressBarParams:
    def __init__(self):
        raise NotImplementedError()

class ProgressBar:
    """ Base class for text-based ProgressBars """

    def __init__(self, max_val):
        raise NotImplementedError()

    def start(self):
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
    def create(params):
        """
        Factory
        """
        if isinstance(params, ProgressBarVoidParams):
            return ProgressBarVoid(params)
        elif isinstance(params, ProgressBarDotsParams):
            return ProgressBarDots(params)
        elif isinstance(params, ProgressBarPlusParams):
            return ProgressBarPlus(params)
        else:
            raise ValueError('name not recognized')

#=============================================================================

class ProgressBarVoidParams(ProgressBarParams):
    def __init__(self):
        pass

class ProgressBarVoid(ProgressBar):
    """ This ProgressBar does simply NOTHING """

    def __init__(self, params):
        assert isinstance(params, ProgressBarVoidParams)
        pass

    def start(self):
        pass

    def set_max_val(self, max_val=100):
        pass

    def next(self, n_steps=1):
        pass

    def update(self, val):
        pass

    def finish(self):
        pass

#=============================================================================

class ProgressBarDotsParams(ProgressBarParams):
    def __init__(self, max_val=100):
        self.max_val = max_val

class ProgressBarDots(ProgressBar):
    """ This progress bar simply prints dots ....... """

    def __init__(self, params):
        assert isinstance(params, ProgressBarDotsParams)
        self.max_val = params.max_val
        self.iter = 0

    def start(self):
        self.iter = 0

    def next(self, n_steps=1):
        self.iter += 1
        if (self.iter % 10) == 0:
            sys.stdout.write('{0}/{1}'.format(self.iter, self.max_val))
        else:
            sys.stdout.write('.')
        sys.stdout.flush()

    def update(self, val):
        n = val - self.iter
        for i in range(n):
            self.next()

    def finish(self):
        sys.stdout.write('\n')

#=============================================================================

# NOTE:
# This implementation uses the library 'progressbar':
#   pip install progressbar
# For the list of widgets, look:
# http://code.google.com/p/python-progressbar/source/browse/progressbar/widgets.py

class ProgressBarPlusParams(ProgressBarParams):
    def __init__(self, max_val=100):
        self.max_val = max_val

class ProgressBarPlus(ProgressBar):
    """ Simple text-based ProgressBar """

    def __init__(self, params):
        assert isinstance(params, ProgressBarPlusParams)
        self.widgets = [progressbar.Percentage(), ' ', \
                   progressbar.Bar(), ' ', \
                   progressbar.Counter(), ' ', \
                   progressbar.Timer(), ' ', \
                   progressbar.ETA()]
        self.max_val = params.max_val
        self.iter = 0
        self.progress = None

    def start(self):
        self.iter = 0
        self.progress = None

    def next(self, n_steps=1):
        self.iter += 1
        self.update(self.iter)

    def update(self, val):
        if self.progress == None:
            self.progress = progressbar.ProgressBar(widgets=self.widgets, \
                            maxval=self.max_val)
            self.progress.start()
        self.progress.update(val)

    def finish(self):
		if self.progress:
			self.progress.finish()
