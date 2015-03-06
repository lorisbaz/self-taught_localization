"""
This module implements an abstraction for the parallization of an
independent function. In particular, given a function f and a collection
of input parameters set {i1, i2, .. iN} we calculate and return the output
{f(i1), f(i2, ... f(iN))}.
A particular implementation will parallelize this computation somehow.

See parfun_test.py for example usages.

Author: Alessandro Bergamo
"""

import inspect
import multiprocessing
import os
import pickle
import subprocess
import time
import sys

import pbar

class ParFunParams:
    """
    Abstract Base class for the Parameters
    (to be used when using the factory)
    """
    def __init__(self):
        pass

class ParFun:
    """Abstract Base class"""

    def __init__(self):
        """An implemented constructor has to take at least a function"""
        raise NotImplementedError()

    def set_fun(self, fun):
        """ Protected method, useful for the subclasses """
        self.fun = fun

    def add_task(self, *args):
        """Add a task, which is defined by a set of parameters"""
        raise NotImplementedError()

    def run(self):
        """Run all the accumulated tasks, and returns their output as a list.
        The i-th elements of the returned list represents the output of the
        i-th function added in time.
        """
        raise NotImplementedError()

    @staticmethod
    def create(params):
        """
        Factory method
        """
        if isinstance(params, ParFunDummyParams):
            return ParFunDummy(params)
        elif isinstance(params, ParFunProcessesParams):
            return ParFunProcesses(params)
        elif isinstance(params, ParFunAnthillParams):
            return ParFunAnthill(params)
        else:
            raise ValueError('Instance of params not recognized')

#=============================================================================

class ParFunDummyParams(ParFunParams):
    def __init__(self, fun=None, progress_bar_params=None):
        self.fun = fun
        self.progress_bar_params = progress_bar_params

class ParFunDummy(ParFun):
    """
    Dummy implementation which implements a simple for-loop on a single core,
    without any parallelization.
    """
    def __init__(self, fun, progress_bar_params=None):
        """
        INPUT:
          fun: function to execute
               OR
               ParFunDummyParams object
        """
        if isinstance(fun, ParFunDummyParams):
            params = fun
            self.fun = params.fun
            self.progress_bar_params = params.progress_bar_params
        else:
            self.fun = fun
            self.progress_bar_params = progress_bar_params
        if self.progress_bar_params == None:
            self.progress_bar_params = pbar.ProgressBarDotsParams()
        self.progress = pbar.ProgressBar.create(self.progress_bar_params)
        self.args = []

    def add_task(self, *args):
        self.args.append(args)

    def run(self):
        out = []
        self.progress.set_max_val(len(self.args))
        for args in self.args:
            out.append( self.fun(*args) )
            self.progress.next()
        self.progress.finish()
        return out

#=============================================================================

class ParFunProcessesParams(ParFunParams):
    def __init__(self, fun=None, num_processes = 4, progress_bar_params = None):
        self.fun = fun
        self.num_processes = num_processes
        self.progress_bar_params = progress_bar_params

class ParFunProcesses(ParFun):
    """
    Parallelization on multiple cores, using the multiprocessing module.

    NOTE: This class has been tested successfully only with function defined in
    global space of the module.
    """
    def __init__(self, fun, num_processes = 4, progress_bar_params = None):
        """
        INPUT:
          fun: function to execute
               OR
               ParFunProcessesParams object
          num_processes: number of cores to use
          progress_bar_params:
               (optional) instance of vlg.util.pbar.ProgressBarParams
        """
        if isinstance(fun, ParFunProcessesParams):
            params = fun
            self.fun = params.fun
            self.num_processes = params.num_processes
            self.progress_bar_params = params.progress_bar_params
        else:
            self.fun = fun
            self.num_processes = num_processes
            self.progress_bar_params = progress_bar_params
        if self.progress_bar_params == None:
            self.progress_bar_params = pbar.ProgressBarDotsParams()
        self.args = []
        self.progress = pbar.ProgressBar.create(self.progress_bar_params)

    def add_task(self, *args):
        self.args.append(args)

    def run(self):
        manager = multiprocessing.Manager()
        queue = manager.Queue()
        pool = multiprocessing.Pool(processes=self.num_processes)
        num_tasks = len(self.args)
        args_for_wrapper = []
        for args in self.args:
            args_for_wrapper.append( (self.fun, args, queue) )
        ar = pool.map_async(_par_fun_processes_map_wrapper, args_for_wrapper)
        num_completed_tasks = 0
        self.progress.set_max_val(num_tasks)
        for i in range(num_tasks):
            queue.get()
            num_completed_tasks += 1
            self.progress.update(num_completed_tasks)
        self.progress.finish()
        assert num_completed_tasks == num_tasks
        pool.close()
        pool.join()
        out = ar.get()
        return out

def _par_fun_processes_map_wrapper(spec):
    fun, args, queue = spec
    out = fun(*args)
    queue.put(1)
    return out

#=============================================================================

class ParFunAnthillParams(ParFunParams):
    def __init__(self, fun = None, time_requested = 99, memory_requested = 2, \
                    job_name = None, python_bin = 'python', \
                    hostname_requested = '*', ironfs=True, \
                    progress_bar_params = None, \
                    tmp_dir = '', max_tasks = 10e3):
        self.fun = fun
        self.time_requested = time_requested
        self.memory_requested = memory_requested
        self.job_name = job_name
        self.python_bin = python_bin
        self.hostname_requested = hostname_requested
        self.ironfs = ironfs
        self.progress_bar_params = progress_bar_params
        self.tmp_dir = tmp_dir
        self.max_tasks = max_tasks

class ParFunAnthill(ParFun):
    """
    Parallelization on the Anthill cluster. Note that each add_task() will
    result in a cluster job, therefore you don't want to add thousands of tasks.

    NOTE: This class has been tested only with function defined in
    global space of the module (i.e. it has not been tested with class methods).
    """

    HOST_GROUPS = ['anthill', 'bokken', 'ennead', 'gridiron', \
                   'katana', 'killington', 'nodachi', 'tanto']

    @staticmethod
    def is_current_host_supported():
        """
        Return True if the current hostname is supported, False otherwise
        """
        supported = False
        hostname = os.uname()[1]
        for host_group in ParFunAnthill.HOST_GROUPS:
            if host_group in hostname:
                supported = True
        return supported

    def __init__(self, fun, time_requested = 99, memory_requested = 2, \
                    job_name = None, python_bin = 'python', \
                    hostname_requested = '*', ironfs=True, \
                    progress_bar_params = None, \
                    tmp_dir = '', max_tasks = 10e3):
        """
        INPUT:
        fun: the function to execute
             OR
             ParFunAnthillParams object
        time_requested: maximum number of hours for each task (int)
        memory_requested: maximum amount of memory for each task (int)
        job_name: the name of the job (string). If not present, the name will
                  be "Job_<functioName>_<date>_<time>"
        python_bin: the python interpreter to use (string)
        hostname_requested: string requesting the nodes to use. Please read the
             Anthill wiki page. Example:
             '(!gridiron-2-2&!gridiron-0-11&gridiron*)|katana*'
        """
        assert ParFunAnthill.is_current_host_supported(), \
                    'The running host does not support ParFunAnthill'
        if isinstance(fun, ParFunAnthillParams):
            params = fun
            self.fun = params.fun
            self.time_requested = params.time_requested
            self.memory_requested = params.memory_requested
            self.job_name = params.job_name
            self.python_bin = params.python_bin
            self.hostname_requested = params.hostname_requested
            self.ironfs = params.ironfs
            self.progress_bar_params = params.progress_bar_params
            self.tmp_dir = params.tmp_dir
            self.max_tasks = params.max_tasks
        else:
            self.fun = fun
            self.time_requested = time_requested
            self.memory_requested = memory_requested
            self.job_name = job_name
            self.python_bin = python_bin
            self.hostname_requested = hostname_requested
            self.ironfs = ironfs
            self.progress_bar_params = progress_bar_params
            self.tmp_dir = tmp_dir
            self.max_tasks = max_tasks
        assert self.time_requested > 0
        assert self.memory_requested > 0
        if self.progress_bar_params == None:
            try:
                # load ProgressBarPlus, if available
                self.progress_bar_params = pbar.ProgressBarPlusParams()
            except:
                # backup plan..
                self.progress_bar_params = pbar.ProgressBarDotsParams()
        self.progress = pbar.ProgressBar.create(self.progress_bar_params)
        self.args = []

    def get_jobname(self):
        return self.job_name

    def set_jobname(self, job_name):
        self.job_name = job_name

    def add_task(self, *args):
        self.args.append(args)

    def run(self):
        # 1) create a temporary directory in the current directory
        if self.job_name == None:
            self.job_name = 'Job_' + self.fun.__name__ + '_' + \
                time.strftime('%y%m%d_%H%M%S')
        print 'ParFunAnthill. Job: {0}'.format(self.job_name)
        if self.tmp_dir != '':
            self.tmp_dir += '/'
        tmpdir = self.tmp_dir + self.job_name
        if os.path.exists(tmpdir):
            print 'Warning: directory ' + tmpdir + ' already exists'
        else:
            os.makedirs(tmpdir)
        # 2) dump to disk the input data of the function to execute
        print 'Dump to the disk the input data of the function '\
              'to execute for {0}'.format(self.job_name)
        if len(self.args) <= self.max_tasks:
            num_args_per_task = [1]*len(self.args)
        else:
            num_args_per_task = [0]*self.max_tasks
            for i in range(len(self.args)):
                num_args_per_task[i % len(num_args_per_task)] += 1
        num_tasks = len(num_args_per_task)
        datain_file = tmpdir + '/input.pkl.'
        dataout_file = tmpdir + '/output.pkl.'
        self.progress.start()
        self.progress.set_max_val(num_tasks)
        idx_args = 0
        for idx_task in range(num_tasks):
            self.progress.next()
            filename = datain_file + str(idx_task)
            if os.path.exists(filename):
                print 'Warning: input file ' + filename + ' already exists.'
                idx_args += num_args_per_task[idx_task]
                continue
            args = []
            for i in range(num_args_per_task[idx_task]):
                args.append(self.args[idx_args])
                idx_args += 1
            fd = open(filename, 'wb')
            pickle.dump(args, fd)
            fd.close()
        assert idx_args == len(self.args)
        self.progress.finish()
        # 3) submit the job to the cluster
        print 'Submit the job {0} to the cluster'.format(self.job_name)
        pyscript = \
            'import pickle\n'\
            'import os\n'\
            'import sys\n'\
            'import {modname}\n'\
            'datain = sys.argv[1]\n'\
            'fd = open(datain, \'rb\')\n'\
            'din = pickle.load(fd)\n'\
            'fd.close()\n'\
            'dout = []\n'\
            'for d in din:\n'\
            '    dout.append( {modname}.{funname}(*d) )\n'\
            'dataout = sys.argv[2]\n'\
            'fd = open(dataout, \'wb\')\n'\
            'pickle.dump(dout, fd)\n'\
            'fd.close()\n'\
            .format(modname = os.path.splitext(os.path.basename(\
                        inspect.getfile(self.fun)))[0],\
                    funname = self.fun.__name__)
        fd = open(self.job_name + '.py', 'w')
        fd.write(pyscript)
        fd.close()
        num_submitted_tasks = 0
        self.progress.start()
        self.progress.set_max_val(num_tasks)
        for idx_task in range(num_tasks):
            self.progress.next()
            outputfile = dataout_file + str(idx_task)
            if os.path.exists(outputfile):
                print 'Warning: output file {0} already exists. We use it.'\
                    .format(outputfile)
                continue
            ironfs_flag = ''
            if self.ironfs:
                ironfs_flag = '-l ironfs'
            bashcmd = \
                'qsub -N {jobname}.{idxtask} -b y -j y -cwd -V {ir} '\
                '-l h_rt={timehours}:00:00 -l virtual_free={memgiga} '\
                '-l mem_free={memgiga} -l hostname=\'{host}\' '\
                '-o {tmpd} -e {tmpd} '\
                '{pybin} {pysc} {datain} {dataout}'\
                .format(jobname = self.job_name, idxtask = idx_task,\
                        ir = ironfs_flag, \
                        timehours = str(self.time_requested), \
                        memgiga = str(self.memory_requested), \
                        host = self.hostname_requested, \
                        tmpd = tmpdir, pybin = self.python_bin, \
                        pysc = self.job_name + '.py', \
                        datain = datain_file + str(idx_task), \
                        dataout = outputfile)
            try:
                bashcmd_output = subprocess.check_output(bashcmd, shell=True)
            except:
                print 'bashcmd: {0}'.format(bashcmd)
                print 'bashcmd_output: {0}'.format(bashcmd_output)
                raise RuntimeError('ERROR while submitting job on Anthill')
            num_submitted_tasks += 1
        self.progress.finish()
        # 4) wait until the job is done
        print 'Executing Job {0}...'.format(self.job_name)
        self.progress.start()
        self.progress.set_max_val(num_submitted_tasks)
        num_running_tasks = num_submitted_tasks
        while num_running_tasks > 0:
            # wait a bit
            time.sleep(30)
            # run qstat
            qstat_output = ''
            try:
                qstat_output = subprocess.check_output(\
                    'SGE_LONG_QNAMES=60 '\
                    'bash -c "qstat -r"', shell=True).strip()
            except:
                print 'Error while executing the qstat command. '\
                      'Number of tasks to complete unknown.'
                continue
            # count how many tasks there are left
            num_running_tasks = 0
            for line in qstat_output.splitlines():
                if (line.find('jobname')>=0) and (line.find(self.job_name)>=0):
                    num_running_tasks += 1
            self.progress.update(num_submitted_tasks - num_running_tasks)
            #print '{0} tasks to complete for {1}'.format(\
            #       num_running_tasks, self.job_name)
        self.progress.finish()
        # 5) collect the results and return
        print 'Collect the output for job {0}'.format(self.job_name)
        self.progress.start()
        self.progress.set_max_val(num_tasks)
        out = [None]*num_tasks
        occurred_error = False
        for idx_task in range(num_tasks):
            self.progress.next()
            try:
                fd = open(dataout_file + str(idx_task), 'rb')
                out[idx_task] = pickle.load(fd)
                fd.close()
            except:
                out[idx_task] = None
                print 'ERROR occured while loading the results from task {0}'\
                    .format(idx_task)
                print sys.exc_info()[0]
                occurred_error = True
        self.progress.finish()
        if occurred_error:
            raise RuntimeError('ERROR. Something went wrong while running '\
                                'the Job on Anthill. Please inspect the '\
                                'directory {0}'.format(tmpdir))
        out = [el for el2 in out for el in el2] # re-compact
        assert len(out) == len(self.args)
        return out

#=============================================================================
