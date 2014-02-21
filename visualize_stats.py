import numpy as np
import os
import os.path
import cPickle as pickle 
import bsddb 
import logging 
import matplotlib.pyplot as plt

from configuration import *
from annotatedimage import *
from stats import *

class Params:
    def __init__(self):
        pass


if __name__ == "__main__":
    # load configurations and parameters  
    conf = Configuration()
    params = Params()
    # Num bins histogram of overlap
    params.n_bins = 32
    # Select classifier
    params.classifier = 'DECAF'
    # experiment name
    params.exp_name_input = 'exp08_01' # take results from here
    # default Configuration, image and label files
    params.conf = conf
    # input directory
    params.input_dir = conf.experiments_output_directory \
                        + '/' + params.exp_name_input 
    logging.info('Started')
    # list the databases chuncks
    n_chunks = len(os.listdir(params.input_dir + '/'))
    # gather data from each chunk
    stats_list = []
    for i in range(n_chunks):
        inputdb = params.input_dir + '/%05d'%i + '.db'
        db_input = bsddb.btopen(inputdb, 'c')
        db_keys = db_input.keys()
        # loop over the images
        for image_key in db_keys:
            # get database entry
            anno = pickle.loads(db_input[image_key])
            # get stuff from database entry
            logging.info('***** Loading statistics ' + \
                          os.path.basename(anno.image_name))
            # append stats
            stats_list.append(anno.stats[params.classifier])

    # Aggregate results 
    logging.info('***** Aggregating stats ') 
    stats_aggr, hist_overlap = Stats.aggregate_results(stats_list, \
                                                    n_bins = params.n_bins)

    plt.bar((hist_overlap[1][0:-1]+hist_overlap[1][1:])/2, hist_overlap[0], \
            width = 1/float(params.n_bins))
    plt.show()
