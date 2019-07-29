#!/usr/bin/env python3
"""AutoML.

Runs the BOHB search

Author:
    Yvan Satyawan <y_satyawan@hotmail.com>

References:
    <https://automl.github.io/HpBandSter/build/html/auto_examples/example_1_
     local_sequential.html>
"""
import argparse
import datetime
import os
import pickle

import hpbandster.core.nameserver as hpns
import hpbandster.core.result as hpres

from hpbandster.optimizers import BOHB as BOHB
from bohb.search_worker import SearchWorker

from utils.slacker import Slacker

import logging
import traceback
logging.basicConfig(level=logging.WARNING)


def parse_args():
    """Parses command line arguments."""
    description = "runs a BOHB search for hyperparameters"
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('data_path', metavar='D', type=str,
                        help='path to the data folder')
    parser.add_argument('output_dir', metavar='O', type=str,
                        help='directory for the result output')
    parser.add_argument('min_budget', metavar='L', type=int,
                        help='minimum budget to use during optimization')
    parser.add_argument('max_budget', metavar='M', type=int,
                        help='maximum budget to use during optimization')
    parser.add_argument('iterations', metavar='I', type=int,
                        help='number of iterations to perform')

    return parser.parse_args()

def run_optimization(args):
    """Runs the optimization process."""
    print("Preparing optimization configuration.")
    date_time = datetime.datetime.now().strftime('%Y-%m-%d-%H_%M_%S')

    # First start nameserver
    NS = hpns.NameServer(run_id=date_time, host='127.0.0.1', port=None)
    NS.start()

    # Then start worker
    w = SearchWorker(args.data_path, nameserver='127.0.0.1',
                     run_id=date_time)
    w.run(background=True)

    # Also start result logger
    result_logger_path = os.path.join(args.output_dir, 'results_log.json')
    print("Result logger will be written to %s" % result_logger_path)
    if os.path.exists(result_logger_path):
        previous_run = hpres.logged_results_to_HBS_result(result_logger_path)
    else:
        previous_run = None

    result_logger = hpres.json_result_logger(directory=args.output_dir,
                                             overwrite=True)

    # Run the optimizer
    bohb = BOHB(configspace=w.get_configspace(),
                run_id=date_time,
                nameserver='127.0.0.1',
                result_logger=result_logger,
                min_budget=args.min_budget,
                max_budget=args.max_budget,
                previous_result=previous_run)

    print("Loaded configuration and optimizer. Now starting optimization run.")

    res = bohb.run(n_iterations=args.iterations)

    output_fp = os.path.join(args.output_dir, 'results.pkl')

    id2config = res.get_id2config_mapping()
    incumbent = res.get_incumbent_id()

    print("Results will be saved at:\n{}".format(output_fp))
    print("Best found configuration: ", id2config[incumbent]['config'])

    with open(output_fp, mode='wb') as file:
        pickle.dump(res, file)

    # Shutdown after completion
    bohb.shutdown(shutdown_workers=True)
    NS.shutdown()


if __name__ == '__main__':
    args = parse_args()
    try:
        print("Starting optimization run with the following parameters:")
        print("    Data path:      {}".format(args.data_path))
        print("    Output dir:     {}".format(args.output_dir))
        print("    Minimum budget: {}".format(args.min_budget))
        print("    Maximum budget: {}".format(args.max_budget))
        print("    Iterations:     {}".format(args.iterations))
        run_optimization(args)
    finally:
        exception_encountered = traceback.format_exc(0)
        if "SystemExit" in exception_encountered \
                or "KeyboardInterrupt" in exception_encountered \
                or "None" in exception_encountered:
            Slacker.send_message("AutoML Optimization finished with minimum "
                                 "budget {}, maximum budget {}, and {} "
                                 "iterations.\n"
                                 "Output file has been written in {}"
                                 .format(args.min_budget,
                                         args.max_budget,
                                         args.iterations,
                                         args.output_dir),
                                 "AutoML Optimization Finished!")

        else:
            print("I Died")
            Slacker.send_code("Exception encountered", exception_encountered)

            with open(os.path.join(os.getcwd(), "traceback.txt"), mode="w") \
                    as file:
                traceback.print_exc(file=file)
        pass
