"""
Functions needed to train and translating in an
online learning environment, using the same model.
"""

import torch
from onmt.train_single import configure_process
from onmt.utils.parse import ArgumentParser
from onmt.inputters.inputter import OLDatasetLazyIter, load_old_vocab, old_style_vocab, max_tok_len
from onmt.model_builder import build_model
from onmt.utils.optimizers import Optimizer
from onmt.trainer import build_trainer
from onmt.models import build_model_saver
from onmt.utils.logging import init_logger, logger
from onmt.train_single import _tally_parameters, _check_save_model_path
from onmt.translate.translator import Translator
from onmt.inputters.dataset_base import Dataset
import onmt.inputters as inputters
from onmt.translate.beam import GNMTGlobalScorer as GNMTGlobalScorer


def load_model(opt, device_id):
    """
    This function loads a model from a checkpoint,
    and builds the model and the trainer.
    opt: (dict) user options.
    device_id: (int) device id.
    :return: trainer.
    """

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
    """
    This function builds the translator.
    :param model: (onmt.modules.NMTModel) NMT model to use for translation.
    :param fields: (dict[str, torchtext.data.Field]) A dict
            mapping each side to its list of name-Field pairs.
    :param opt: (dict) user options.
    :param model_opt: (dict) model options.
    :param out_file: (TextIO or codecs.StreamReaderWriter) Output file.
    :return: translator.
    """

    return Translator.from_opt(
        model,
        fields,
        opt,
        model_opt,
        # global_scorer=onmt.translate.GNMTGlobalScorer.from_opt(opt),
        global_scorer=GNMTGlobalScorer.from_opt(opt),
        out_file=out_file,
        report_score=True,
        logger=logger
    )


def create_dataset(src, tgt, fields, opt):
    """
    This function builds the dataset used for training the model.
    :param src: (generator) source sentence.
    :param tgt: (generator) target sentence.
    :param fields: (dict[str, torchtext.data.Field]) A dict
            mapping each side to its list of name-Field pairs.
    :param opt: (dict) user options.
    :return: train dataset.
    """

    dataset = Dataset(
            fields,
            readers=[inputters.str2reader["text"].from_opt(opt), inputters.str2reader["text"].from_opt(opt)],
            data=[("src", src), ("tgt", tgt)],
            dirs=[None, None],
            sort_key=inputters.str2sortkey["text"],
            filter_pred=None
        )
    return dataset


def train(src, tgt, trainer, fields, steps_increase, opt):
    """
    This function trains the model.
    :param src: (generator) source sentence.
    :param tgt: (generator) target sentence.
    :param trainer:
    :param fields: (dict[str, torchtext.data.Field]) A dict
            mapping each side to its list of name-Field pairs.
    :param steps_increase: (int) steps increase.
    :param opt: (dict) user options.
    """

    # Build the dataset.
    dataset = create_dataset(src, tgt, fields, opt)

    # Build the train iterator.
    train_iter = OLDatasetLazyIter(dataset, fields, opt.batch_size,
                                 max_tok_len if opt.batch_type == "tokens" else None,
                                 8 if opt.model_dtype == "fp16" else 1, "cuda" if opt.gpu_ranks else "cpu",
                                 True, repeat=not opt.single_pass, num_batches_multiple=max(opt.accum_count)
                                                                                        * opt.world_size)

    if len(opt.gpu_ranks):
        logger.info('Starting training on GPU: %s' % opt.gpu_ranks)
    else:
        logger.info('Starting training on CPU, could be very slow')

    # Train the model.
    trainer.train(
        train_iter,
        opt.train_steps + steps_increase,
        save_checkpoint_steps=opt.save_checkpoint_steps,
        valid_iter=None,
        valid_steps=opt.valid_steps)
