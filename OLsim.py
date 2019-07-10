#!/usr/bin/env python
"""
    Simulation of a user working in an online learning framework. For each line in the test,
    the process consists in:
    - Generating a translation hypothesis.
    - Considering the reference translation as the user's post-edition.
    - Updating the model.
"""
from __future__ import division
import argparse
import os
import signal
import io
import codecs

import onmt.opts as opts
import onmt.utils.distributed
from onmt.train_single import main as single_main
from onmt.utils.logging import logger
import onmt.online_learning


def get_source(src, line):
    """
    :param src: (list) source sentences.
    :param line: (int) number of the desired sentence.
    :return: (generator) desired sentence.
    """
    yield src[line]


def get_target(tgt, line):
    """
    :param tgt: (list) target sentences.
    :param line: (int) number of the desired sentence.
    :return: (generator) desired sentence.
    """
    yield tgt[line]


def main(opt):
    """
    User simulation in an online learning framework.
    :param opt: (dict) user options.
    """

    # Ensure gpu consistency between training and translating.
    if (opt.gpu_ranks == [0] and opt.gpu != 0) or (opt.gpu_ranks != [0] and opt.gpu == 0):
        opt.gpu_ranks = [0]
        opt.gpu = 0

    # Check gpu/cpu usage.
    if len(opt.gpu_ranks) == 1:  # case 1 GPU only
        device_id = 0
    else:   # case only CPU
        device_id = -1

    # Load model.
    trainer, fields, model, model_opt = onmt.online_learning.load_model(opt, device_id)

    # Build translator.
    out_file = codecs.open(opt.output, 'w+', 'utf-8')

    translator = onmt.online_learning.build_translator(model, fields, opt, model_opt, out_file)

    # Open files.
    with io.open(opt.src, encoding='utf8') as f:
        src = f.readlines()
    with io.open(opt.tgt, encoding='utf8') as f:
        tgt = f.readlines()
    n_lines = len(src)
    steps = 0

    # Simulation.
    for n_line in range(n_lines):

        logger.info('Processing line %s.' % n_line)

        # Translate source.
        translator.translate(src=[src[n_line]],
                             tgt=None,
                             src_dir=None,
                             batch_size=opt.batch_size,
                             attn_debug=opt.attn_debug)

        # Update models using the reference.
        for updates in range(opt.ol_updates):
            onmt.online_learning.train(get_source(src, n_line), get_target(tgt, n_line), trainer, fields, steps, opt)
            steps += 1

    if opt.tensorboard:
        trainer.report_manager.tensorboard_writer.close()


def run(opt, device_id, error_queue):
    """ run process """
    try:
        gpu_rank = onmt.utils.distributed.multi_init(opt, device_id)
        if gpu_rank != opt.gpu_ranks[device_id]:
            raise AssertionError("An error occurred in \
                  Distributed initialization")
        single_main(opt, device_id)
    except KeyboardInterrupt:
        pass  # killed by parent, do nothing
    except Exception:
        # propagate exception to parent process, keeping original traceback
        import traceback
        error_queue.put((opt.gpu_ranks[device_id], traceback.format_exc()))


class ErrorHandler(object):
    """A class that listens for exceptions in children processes and propagates
    the tracebacks to the parent process."""

    def __init__(self, error_queue):
        """ init error handler """
        import signal
        import threading
        self.error_queue = error_queue
        self.children_pids = []
        self.error_thread = threading.Thread(
            target=self.error_listener, daemon=True)
        self.error_thread.start()
        signal.signal(signal.SIGUSR1, self.signal_handler)

    def add_child(self, pid):
        """ error handler """
        self.children_pids.append(pid)

    def error_listener(self):
        """ error listener """
        (rank, original_trace) = self.error_queue.get()
        self.error_queue.put((rank, original_trace))
        os.kill(os.getpid(), signal.SIGUSR1)

    def signal_handler(self, signalnum, stackframe):
        """ signal handler """
        for pid in self.children_pids:
            os.kill(pid, signal.SIGINT)  # kill children processes
        (rank, original_trace) = self.error_queue.get()
        msg = """\n\n-- Tracebacks above this line can probably
                 be ignored --\n\n"""
        msg += original_trace
        raise Exception(msg)


if __name__ == "__main__":
    # Load parser.
    parser = argparse.ArgumentParser(
        description='OL.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Load options.
    opts.config_opts(parser)
    opts.model_opts(parser)
    opts.OL_opts(parser)

    # Parse arguments.
    opt = parser.parse_args()

    # Init simulation.
    main(opt)
