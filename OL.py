#!/usr/bin/env python
"""
    Main training workflow
"""
from __future__ import division
import argparse
import os
import signal
import torch

import onmt.opts as opts
import onmt.utils.distributed
import onmt.inputters as inputters

from onmt.train_single import main as single_main
from onmt.inputters.inputter import build_dataset_iter, lazily_load_dataset, \
    _load_fields, _collect_report_features
from onmt.model_builder import build_model
from onmt.utils.optimizers import build_optim
from onmt.trainer import build_trainer
from onmt.models import build_model_saver
from onmt.utils.logging import init_logger, logger
from onmt.train_single import training_opt_postprocessing, _tally_parameters, _check_save_model_path


def load_model(opt, device_id):

    opt = training_opt_postprocessing(opt, device_id)
    init_logger(opt.log_file)

    # Load model.
    logger.info('Loading checkpoint from %s' % opt.models[0])
    checkpoint = torch.load(opt.models[0], map_location=lambda storage, loc: storage)
    model_opt = checkpoint['opt']

    # Peek the first dataset to determine the data_type.
    # (All datasets have the same data_type).
    first_dataset = next(lazily_load_dataset("train", opt))
    data_type = first_dataset.data_type

    # Load fields generated from preprocess phase.
    fields = _load_fields(first_dataset, data_type, opt, checkpoint)

    # Build model.
    model = build_model(model_opt, opt, fields, checkpoint)
    n_params, enc, dec = _tally_parameters(model)
    logger.info('encoder: %d' % enc)
    logger.info('decoder: %d' % dec)
    logger.info('* number of parameters: %d' % n_params)
    _check_save_model_path(opt)

    # Build optimizer.
    optim = build_optim(model, opt, checkpoint)

    # Build model saver
    model_saver = build_model_saver(model_opt, opt, model, fields, optim)

    return build_trainer(opt, device_id, model, fields,
                            optim, data_type, model_saver=model_saver), fields, data_type


def train(src, tgt, trainer, fields, data_type, cur_device):

    data = inputters. \
        build_dataset(fields,
                      data_type,
                      src_path=None,
                      src_data_iter=[src],
                      tgt_path=None,
                      tgt_data_iter=[tgt],
                      src_dir=opt.src_dir,
                      sample_rate=16000,
                      window_size=.02,
                      window_stride=.01,
                      window='hamming',
                      use_filter_pred=False,
                      image_channel_size=3)

    def train_iter_fct():
        return inputters.OrderedIterator(
            dataset=data, device=cur_device,
            batch_size=opt.batch_size, train=False, sort=False,
            sort_within_batch=True, shuffle=False)

    # Do training.
    if len(opt.gpu_ranks):
        logger.info('Starting training on GPU: %s' % opt.gpu_ranks)
    else:
        logger.info('Starting training on CPU, could be very slow')
    trainer.train(train_iter_fct, None, opt.train_steps,
                  opt.valid_steps)


def main(opt):

    if len(opt.gpu_ranks) == 1:  # case 1 GPU only
        device_id = 0
        cur_device = "cuda"
    else:   # case only CPU
        device_id = -1
        cur_device = "cpu"

    trainer, fields, data_type = load_model(opt, device_id)

    src = [line for line in open(opt.src)]
    tgt = [line for line in open(opt.tgt)]
    n_lines = len(src)

    for n_line in range(n_lines):

        logger.info('Processing line %s.' % n_line)
        logger.info('%s.' % src[n_line])

        train(src[n_line], tgt[n_line], trainer, fields, data_type, cur_device)

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
    parser = argparse.ArgumentParser(
        description='OL.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    opts.add_md_help_argument(parser)
    opts.model_opts(parser)
    opts.OL_opts(parser)

    opt = parser.parse_args()
    main(opt)
