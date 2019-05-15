#!/usr/bin/env python
"""
    Main training workflow
"""
from __future__ import division
import argparse
import os
import signal
import torch
import io
import codecs

import onmt.opts as opts
import onmt.utils.distributed

from onmt.train_single import main as single_main
from onmt.train_single import configure_process
from onmt.utils.parse import ArgumentParser
from onmt.inputters.inputter import build_dataset_iter, load_old_vocab, old_style_vocab
from onmt.model_builder import build_model
from onmt.utils.optimizers import Optimizer
from onmt.trainer import build_trainer
from onmt.models import build_model_saver
from onmt.utils.logging import init_logger, logger
from onmt.train_single import _tally_parameters, _check_save_model_path
from onmt.translate.translator import Translator


def load_model(opt, device_id):

    configure_process(opt, device_id)
    init_logger(opt.log_file)

    # Load model.
    logger.info('Loading checkpoint from %s' % opt.models[0])
    checkpoint = torch.load(opt.models[0], map_location=lambda storage, loc: storage)
    model_opt = ArgumentParser.ckpt_model_opts(checkpoint["opt"])
    ArgumentParser.update_model_opts(model_opt)
    ArgumentParser.validate_model_opts(model_opt)
    logger.info('Loading vocab from checkpoint at %s.' % opt.train_from)
    vocab = checkpoint['vocab']

    # check for code where vocab is saved instead of fields
    # (in the future this will be done in a smarter way)
    if old_style_vocab(vocab):
        fields = load_old_vocab(
            vocab, opt.model_type, dynamic_dict=opt.copy_attn)
    else:
        fields = vocab

    # Build model.
    model = build_model(model_opt, opt, fields, checkpoint)
    n_params, enc, dec = _tally_parameters(model)
    logger.info('encoder: %d' % enc)
    logger.info('decoder: %d' % dec)
    logger.info('* number of parameters: %d' % n_params)
    _check_save_model_path(opt)

    # Build optimizer.
    optim = Optimizer.from_opt(model, opt, checkpoint=checkpoint)

    # Build model saver
    model_saver = build_model_saver(model_opt, opt, model, fields, optim)

    return build_trainer(opt, device_id, model, fields, optim, model_saver=model_saver), fields, model, model_opt


def build_translator(model, fields, opt, model_opt, out_file):

    kwargs = {k: getattr(opt, k)
              for k in ["beam_size", "n_best", "max_length", "min_length",
                        "stepwise_penalty", "block_ngram_repeat",
                        "ignore_when_blocking", "dump_beam", "report_bleu",
                        "data_type", "replace_unk", "gpu", "verbose", "fast",
                        "sample_rate", "window_size", "window_stride",
                        "window", "image_channel_size"]}

    return Translator(model, fields, global_scorer=onmt.translate.GNMTGlobalScorer(opt.alpha,
                      opt.beta, opt.coverage_penalty, opt.length_penalty),
                      out_file = out_file, report_score=True,
                      copy_attn=model_opt.copy_attn, logger=logger,
                      **kwargs)


def train(trainer, fields, n, opt):
    train_iter = build_dataset_iter("train", fields, opt)
    valid_iter = build_dataset_iter("valid", fields, opt, is_train=False)

    if len(opt.gpu_ranks):
        logger.info('Starting training on GPU: %s' % opt.gpu_ranks)
    else:
        logger.info('Starting training on CPU, could be very slow')

    trainer.train(
        train_iter,
        opt.train_steps + n,
        save_checkpoint_steps=opt.save_checkpoint_steps,
        valid_iter=valid_iter,
        valid_steps=opt.valid_steps)


def main(opt):

    if len(opt.gpu_ranks) == 1:  # case 1 GPU only
        device_id = 0
    else:   # case only CPU
        device_id = -1

    trainer, fields, model, model_opt = load_model(opt, device_id)

    out_file = codecs.open(opt.output, 'w+', 'utf-8')

    translator = build_translator(model, fields, opt, model_opt, out_file)

    with io.open(opt.src, encoding='utf8') as f:
        src = f.readlines()
    with io.open(opt.tgt, encoding='utf8') as f:
        tgt = f.readlines()
    n_lines = len(src)
    steps = 0

    for n_line in range(n_lines):

        logger.info('Processing line %s.' % n_line)

        translator.translate(src_path=None,
                             src_data_iter=[src[n_line]],
                             tgt_path=None,
                             src_dir=None,
                             batch_size=opt.batch_size,
                             attn_debug=opt.attn_debug)

        for updates in range(opt.ol_updates):
            train(trainer, fields, steps, opt)
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
    parser = argparse.ArgumentParser(
        description='OL.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    opts.config_opts(parser)
    opts.model_opts(parser)
    opts.OL_opts(parser)

    opt = parser.parse_args()
    main(opt)
