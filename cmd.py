# -*- coding: utf-8 -*-
import locale
import logging
import math
import random
import os.path
import sys
import time
import string
import json

from . import get_version
from . import utils
from .baseline import BaselineModel, AnnotationCorpusWeight, \
    MorphLengthCorpusWeight, NumMorphCorpusWeight, FixedCorpusWeight
from .exception import ArgumentException
from .io import MorfessorIO
from .evaluation import MorfessorEvaluation, EvaluationConfig, \
    WilcoxonSignedRank, FORMAT_STRINGS

PY3 = sys.version_info[0] == 3

# _str is used to convert command line arguments to the right type (str for PY3, unicode for PY2
if PY3:
    _str = str
else:
    _str = lambda x: unicode(x, encoding=locale.getpreferredencoding())

_logger = logging.getLogger(__name__)


def get_default_argparser():
    import argparse

    parser = argparse.ArgumentParser(
        prog='morfessor.py',
        description="""
Morfessor %s
Copyright (c) 2012-2018, Sami Virpioja, Peter Smit, and Stig-Arne Grönroos.
All rights reserved.
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:
1.  Redistributions of source code must retain the above copyright
    notice, this list of conditions and the following disclaimer.
2.  Redistributions in binary form must reproduce the above
    copyright notice, this list of conditions and the following
    disclaimer in the documentation and/or other materials provided
    with the distribution.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
Command-line arguments:
""" % get_version(),
        epilog="""
Simple usage examples (training and testing):
  %(prog)s -t training_corpus.txt -s model.pickled
  %(prog)s -l model.pickled -T test_corpus.txt -o test_corpus.segmented
Interactive use (read corpus from user):
  %(prog)s -m online -v 2 -t -
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        add_help=False)

    # Options for input data files
    add_arg = parser.add_argument_group('input data files').add_argument
    add_arg('-l', '--load', dest="loadfile", default=None, metavar='<file>',
            help="load existing model from file (pickled model object)")
    add_arg('-L', '--load-segmentation', dest="loadsegfile", default=None,
            metavar='<file>',
            help="load existing model from segmentation "
                 "file (Morfessor 1.0 format)")
    add_arg('-t', '--traindata', dest='trainfiles', action='append',
            default=[], metavar='<file>',
            help="input corpus file(s) for training (text or bz2/gzipped text;"
                 " use '-' for standard input; add several times in order to "
                 "append multiple files)")
    add_arg('-T', '--testdata', dest='testfiles', action='append',
            default=[], metavar='<file>',
            help="input corpus file(s) to analyze (text or bz2/gzipped text;  "
                 "use '-' for standard input; add several times in order to "
                 "append multiple files)")
    add_arg('-P', '--score-segmentations', dest='evalfile',
            default=None, metavar='<file>',
            help="find the probability of the provided segmentations. "
                 "Output is saved to the path provided to flag -o, "
                 "with tab-separated values for compound, analysis, "
                 "logprob, and condprob")

    # Options for output data files
    add_arg = parser.add_argument_group('output data files').add_argument
    add_arg('-o', '--output', dest="outfile", default='-', metavar='<file>',
            help="output file for test data results (for standard output, "
                 "use '-'; default '%(default)s')")
    add_arg('-s', '--save', dest="savefile", default=None, metavar='<file>',
            help="save final model to file (pickled model object)")
    add_arg('-S', '--save-segmentation', dest="savesegfile", default=None,
            metavar='<file>',
            help="save model segmentations to file (Morfessor 1.0 format)")
    add_arg('--save-reduced', dest="savereduced", default=None,
            metavar='<file>',
            help="save final model to file in reduced form (pickled model "
            "object). A model in reduced form can only be used for "
            "segmentation of new words.")
    add_arg('-x', '--lexicon', dest="lexfile", default=None, metavar='<file>',
            help="output final lexicon to given file")
    add_arg('--nbest', dest="nbest", default=1, type=int, metavar='<int>',
            help="output n-best viterbi results")
    add_arg('--get-trees', dest="gettrees", action="store_true",
            help="output parse trees as well as flat analyses")

    # Options for data formats
    add_arg = parser.add_argument_group(
        'data format options').add_argument
    add_arg('-e', '--encoding', dest='encoding', metavar='<encoding>',
            help="encoding of input and output files (if none is given, "
                 "both the local encoding and UTF-8 are tried)")
    add_arg('--lowercase', dest="lowercase", default=False,
            action='store_true',
            help="lowercase input data")
    add_arg('--traindata-list', dest="list", default=False,
            action='store_true',
            help="input file(s) for batch training are lists "
                 "(one compound per line, optionally count as a prefix)")
    add_arg('--atom-separator', dest="separator", type=_str, default=None,
            metavar='<regexp>',
            help="atom separator regexp (default %(default)s)")
    add_arg('--construction-separator', dest='constrseparator',
            type=_str, default=' + ', metavar='<str>',
            help="construction separator; plain string, not regexp "
            "(default: '%(default)s')")
    add_arg('--compound-separator', dest="cseparator", type=_str, default=r'\s+',
            metavar='<regexp>',
            help="compound separator regexp (default '%(default)s')")
    add_arg('--analysis-separator', dest='analysisseparator', type=_str,
            default=', ', metavar='<str>',
            help="separator for different analyses in an annotation file. Use"
                 "  NONE for only allowing one analysis per line")
    add_arg('--output-format', dest='outputformat', type=_str,
            default=r'{compound}\t{analysis}\n', metavar='<format>',
            help="format string for --output file (default: '%(default)s'). "
            "Valid keywords are: "
            "{analysis} = constructions of the compound, "
            "{compound} = compound string, "
            "{count} = count of the compound (currently always 1), "
            "{logprob} = log-probability of the analysis, "
            "{clogprob} = log-probability of the compound, and "
            "{condprob} = conditional probability of the analysis. Valid escape "
            "sequences are '\\n' (newline) and '\\t' (tabular)")
    add_arg('--output-newlines', dest='outputnewlines', default=False,
            action='store_true',
            help="for each newline in input, print newline in --output file "
            "(default: '%(default)s')")
    add_arg('--strip-atom-sep', dest="strip_atom_sep", action="store_true",
            help="remove the atom separator in output files")

    # Options for model training
    add_arg = parser.add_argument_group(
        'training and segmentation options').add_argument
    add_arg('-m', '--mode', dest="trainmode", default='init+batch',
            metavar='<mode>',
            choices=['none', 'batch', 'init', 'init+batch', 'online',
                     'online+batch'],
            help="training mode ('none', 'init', 'batch', 'init+batch', "
                 "'online', or 'online+batch'; default '%(default)s')")
    add_arg('-a', '--algorithm', dest="algorithm", default='recursive',
            metavar='<algorithm>', choices=['recursive', 'viterbi'],
            help="algorithm type ('recursive', 'viterbi'; default "
                 "'%(default)s')")
    add_arg('-d', '--dampening', dest="dampening", type=_str, default='ones',
            metavar='<type>', choices=['none', 'log', 'ones'],
            help="frequency dampening for training data ('none', 'log', or "
                 "'ones'; default '%(default)s')")
    add_arg('-f', '--forcesplit', dest="forcesplit", type=list, default=['-'],
            metavar='<list>',
            help="force split on given atoms (default '-'). The argument "
                 "is a string of single chars, use '' for no forced splits.")
    add_arg('--regexsplit', dest="regexsplit", type=str, default=None,
            metavar='<regexp>',
            help="force split at all boundaries where the provided regex is "
                 "satisfied. Assumes the atom separator is still present. "
                 "(default %(default)s)")
    add_arg('--nosplit-re', dest="nosplit", type=_str, default=None,
            metavar='<regexp>',
            help="if the expression matches the two surrounding characters, "
                 "do not allow splitting (default %(default)s)")
    add_arg('-F', '--finish-threshold', dest='finish_threshold', type=float,
            default=0.005, metavar='<float>',
            help="Stopping threshold. Training stops when "
                 "the improvement of the last iteration is"
                 "smaller than OR EQUAL TO " 
                 "finish_threshold * #boundaries; "
                 "if set to 0 and --epochs-until-frozen is not None, stop "
                 "when all of the analyses have been frozen. "
                 "(default '%(default)s')")
    add_arg('-r', '--randseed', dest="randseed", default=None,
            metavar='<int>', type=int,
            help="seed for random number generator")
    add_arg('-R', '--randsplit', dest="splitprob", default=None, type=float,
            metavar='<float>',
            help="initialize new words by random splitting using the given "
                 "split probability (default no splitting)")
    add_arg('--skips', dest="skips", default=False, action='store_true',
            help="use random skips for frequently seen compounds to speed up "
                 "training")
    add_arg('--batch-minfreq', dest="freqthreshold", type=int, default=1,
            metavar='<int>',
            help="compound frequency threshold for batch training (default "
                 "%(default)s)")
    add_arg('--max-epochs', dest='maxepochs', type=int, default=None,
            metavar='<int>',
            help='hard maximum of epochs in training')
    add_arg('--online-epochint', dest="epochinterval", type=int,
            default=10000, metavar='<int>',
            help="epoch interval for online training (default %(default)s)")
    add_arg('--viterbi-smoothing', dest="viterbismooth", default=0,
            type=float, metavar='<float>',
            help="additive smoothing parameter for Viterbi training "
                 "and segmentation (default %(default)s)")
    add_arg('--viterbi-maxlen', dest="viterbimaxlen", default=30,
            type=int, metavar='<int>',
            help="maximum construction length in Viterbi training "
                 "and segmentation (default %(default)s)")
    add_arg('--viterbi-allow-new', dest='viterbi_allownew', action="store_true",
            help="flag for allowing Viterbi to propose a new"
                 "short construction when no attested construction"
                 "is available and smoothing is disabled")
    add_arg('--left-reduplication', dest='leftred', action="store_true",
            help="flag for analyzing left-attaching reduplication")
    add_arg('--Lred-minbase-weight', dest="Lred_minbase_weight",
            type=int, metavar="<int>", default=1,
            help="the minimum weight for the base of left-attaching"
                 " reduplication (default %(default)s)")
    add_arg('--right-reduplication', dest='rightred', action="store_true",
            help="flag for analyzing right-attaching reduplication")
    add_arg('--Rred-minbase-weight', dest="Rred_minbase_weight",
            type=int, metavar="<int>", default=1,
            help="the minimum weight for the base of right-attaching"
                 " reduplication (default %(default)s)")
    add_arg('--default-red-attachment', dest='defaultredattachment', default="L", 
            type=str, metavar='<str>',
            help="a string indicating, for cases where both left- and right-attaching"
                 " reduplication are checked, which one should be the default."
                 " Options: L or R (default %(default)s)")
    add_arg('--disable-full-red', dest="disable_fullred", action="store_true",
            help="flag for removing full reduplication from consideration, while"
                 " keeping other kinds of reduplication available")
    add_arg('--skippable-full-red', dest="skippable_fullred", action="store_true",
            help="flag for allowing analyses to keep a reduplicant and base together"
                 " in full reduplication even when the two are isolated from the rest"
                 " of the construction")
    add_arg('--enforce-full-red', dest="enforce_fullred", action="store_true",
            help="flag for greedily enforcing full reduplication for all constructions"
                 " that could be formed entirely from full reduplication")
    add_arg('--full-red-only', dest="fullredonly", action="store_true",
            help="flag for only considering full reduplication (and coercing all"
                 " other cases of reduplication to full reduplication)")
    add_arg('--delay-red-checking', dest='delayred', action="store_true",
            help="flag for delaying checking for reduplication until no"
                 " alternative splits can be identified")      
    add_arg('--delay-nonedge-red', dest='delaynonedge', action="store_true",
            help="flag for delaying checking for NON-EDGE-ALIGNED reduplication"
                 " until no alternative splits can be identified. Edge-aligned"
                 " reduplication is still checked eagerly.")     
    add_arg('--posthoc-restructuring', dest='posthocrestructure', action="store_true",
            help="flag for restructuring parse trees when post-hoc reduplication"
                 " is identified in the flat structure. Without this flag, trees"
                 " are merely relabeled when the reduplicant and base fall within"
                 " the same branch.")                                        
    add_arg('--penalized-weights', dest="penalty_weights", type=json.loads,
            metavar="<dict>", default='{}',
            help="a string representation of a dictionary mapping from construction"
                 " penalty names to the weight that construction tokens incurring"
                 " those penalties should be given. Weights should be between 0 and"
                 " 1; 0 indicates that the corresponding constructions are never"
                 " permitted, and 1 indicates that they are not penalized."
                 " Argument should be in JSON format, with double-quotes enclosing"
                 " penalty names.")
    add_arg('--separate-red', dest='sepred', action="store_true",
            help="flag for separating the <RED> kinds")
    add_arg('--backup', dest='backup', action="store_true",
            help="flag for whether or not to revert to previous analysis in"
                 " case of cost increasing")    
    add_arg('--backup-delay', dest="backupdelay", default=1,
            type=int, metavar='<int>',
            help="the number of epochs to wait before storing removed splitloc "
                 "history and backing up in case of increased cost (default %(default)s)")    
    add_arg('--debug-delay', dest="debugdelay", default=0,
            type=int, metavar='<int>',
            help="the number of epochs to wait before logging debugging output "
                 " (default %(default)s)")                                                                       

    # Options for corpusweight tuning
    add_arg = parser.add_mutually_exclusive_group().add_argument
    add_arg('-D', '--develset', dest="develfile", default=None,
            metavar='<file>',
            help="load annotated data for tuning the corpus weight parameter")
    add_arg('--morph-length', dest='morphlength', default=None, type=float,
            metavar='<float>',
            help="tune the corpusweight to obtain the desired average morph "
                 "length")
    add_arg('--num-morph-types', dest='morphtypes', default=None, type=float,
            metavar='<float>',
            help="tune the corpusweight to obtain the desired number of morph "
                 "types")

    # Options for semi-supervised model training
    add_arg = parser.add_argument_group(
        'semi-supervised training options').add_argument
    add_arg('-w', '--corpusweight', dest="corpusweight", type=float,
            default=1.0, metavar='<float>',
            help="corpus weight parameter (default %(default)s); "
                 "sets the initial value if other tuning options are used")
    add_arg('--weight-threshold', dest='threshold', default=0.01,
            metavar='<float>', type=float,
            help='percentual stopping threshold for corpusweight updaters')
    add_arg('--full-retrain', dest='fullretrain', action='store_true',
            default=False,
            help='do a full retrain after any weights have converged')
    add_arg('-A', '--annotations', dest="annofile", default=None,
            metavar='<file>',
            help="load annotated data for semi-supervised learning")
    add_arg('-W', '--annotationweight', dest="annotationweight",
            type=float, default=None, metavar='<float>',
            help="corpus weight parameter for annotated data (if unset, the "
                 "weight is set to balance the number of tokens in annotated "
                 "and unannotated data sets)")

    # Options for evaluation
    add_arg = parser.add_argument_group('Evaluation options').add_argument
    add_arg('-G', '--goldstandard', dest='goldstandard', default=None,
            metavar='<file>',
            help='If provided, evaluate the model against the gold standard')
    add_arg('--eval-properties', dest='evallocations', action='store_false',
            help='If provided, evaluate Splits based on their properties, '
                 'not just their locations')
    add_arg('--score-red-asis', dest='score_red_alternatives', action="store_false",
            help="If provided, when using option -P to score provided segmentations, "
                 "score the segmentation exactly as provided; do not consider "
                 "alternatives with the same splits but different recognition "
                 "of reduplicants")

    # Options for logging
    add_arg = parser.add_argument_group('logging options').add_argument
    add_arg('-v', '--verbose', dest="verbose", type=int, default=1,
            metavar='<int>',
            help="verbose level; controls what is written to the standard "
                 "error stream or log file (default %(default)s)")
    add_arg('--logfile', dest='log_file', metavar='<file>',
            help="write log messages to file in addition to standard "
                 "error stream")
    add_arg('--progressbar', dest='progress', default=False,
            action='store_true',
            help="Force the progressbar to be displayed (possibly lowers the "
                 "log level for the standard error stream)")

    add_arg = parser.add_argument_group('other options').add_argument
    add_arg('-h', '--help', action='help',
            help="show this help message and exit")
    add_arg('--version', action='version',
            version='%(prog)s ' + get_version(),
            help="show version number and exit")

    return parser


def initialize_logging(args):
    """Initialize loggers based on command line args"""
    if args.verbose >= 2:
        loglevel = logging.DEBUG
    elif args.verbose >= 1:
        loglevel = logging.INFO
    else:
        loglevel = logging.WARNING

    rootlogger = logging.getLogger()
    rootlogger.setLevel(logging.DEBUG)

    logfile_format = '%(asctime)s %(levelname)s:%(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    console_format = '%(message)s'

    console_level = loglevel
    if args.log_file is not None or (hasattr(args, 'progress') and args.progress):
        # If logging to a file or progress bar is forced, make INFO
        # the highest level for the error stream
        console_level = max(loglevel, logging.INFO)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(console_level)
    ch.setFormatter(logging.Formatter(console_format))
    rootlogger.addHandler(ch)

    # FileHandler for log_file
    if args.log_file is not None:
        fh = logging.FileHandler(args.log_file, 'w')
        fh.setLevel(loglevel)
        fh.setFormatter(logging.Formatter(logfile_format, date_format))
        rootlogger.addHandler(fh)

    return console_level


def main(args):

    console_level = initialize_logging(args)

    # If debug messages are printed to screen, only warning messages
    # (or above) should be printed to screen, or if stderr is not a
    # tty (but a pipe or a file), don't show the progressbar
    if (console_level != logging.INFO or
            (hasattr(sys.stderr, 'isatty') and not sys.stderr.isatty())):
        utils.show_progress_bar = False

    # Force progress bar
    if args.progress:
        utils.show_progress_bar = True

    if (args.loadfile is None and
            args.loadsegfile is None and
            len(args.trainfiles) == 0):
        raise ArgumentException("either model file or training data should "
                                "be defined")

    if args.randseed is not None:
        random.seed(args.randseed)

    io = MorfessorIO(encoding=args.encoding,
                     construction_separator=args.constrseparator,
                     compound_separator=args.cseparator,
                     atom_separator=args.separator,
                     lowercase=args.lowercase,
                     strip_atom_sep=args.strip_atom_sep,
                     nosplit_re=args.nosplit,
                     split_re=args.regexsplit,
                     forcesplit_chars=args.forcesplit,
                     check_left_red=args.leftred,
                     Lred_minbase_weight=args.Lred_minbase_weight,
                     check_right_red=args.rightred,
                     Rred_minbase_weight=args.Rred_minbase_weight,
                     default_red_attachment=args.defaultredattachment,
                     full_red_only=args.fullredonly,
                     disable_fullred=args.disable_fullred,
                     separate_red=args.sepred
                     )

    # Load exisiting model or create a new one
    if args.loadfile is not None:
        model = io.read_binary_model_file(args.loadfile)

    else:
        model = BaselineModel(corpusweight=args.corpusweight,
                              use_skips=args.skips,
                              delay_red_checking=args.delayred,
                              delay_nonedge_red=args.delaynonedge,
                              posthoc_restructuring = args.posthocrestructure,
                              penalty_weights=args.penalty_weights,
                              debugging_level=max(args.verbose -1, 0),
                              backup=args.backup,
                              backup_delay=args.backupdelay,
                              debug_delay=args.debugdelay,
                              analyze_reduplication=(args.leftred or args.rightred or args.fullredonly),
                              full_red_only=args.fullredonly,
                              skippable_fullred=(args.disable_fullred or args.skippable_fullred),
                              enforce_fullred=args.enforce_fullred)

    if args.loadsegfile is not None:
        model.load_segmentations(io.read_segmentation_file(args.loadsegfile))

    analysis_sep = (args.analysisseparator
                    if args.analysisseparator != 'NONE' else None)

    if args.annofile is not None:
        annotations = io.read_annotations_file(args.annofile,
                                               analysis_sep=analysis_sep)
        model.set_annotations(annotations, args.annotationweight)

    if args.develfile is not None:
        develannots = io.read_annotations_file(args.develfile,
                                               analysis_sep=analysis_sep)
        updater = AnnotationCorpusWeight(develannots, args.threshold)
        model.set_corpus_weight_updater(updater)

    if args.morphlength is not None:
        updater = MorphLengthCorpusWeight(args.morphlength, args.threshold)
        model.set_corpus_weight_updater(updater)

    if args.morphtypes is not None:
        updater = NumMorphCorpusWeight(args.morphtypes, args.threshold)
        model.set_corpus_weight_updater(updater)

    start_corpus_weight = model.get_corpus_coding_weight()

    # Set frequency dampening function
    if args.dampening == 'none':
        dampfunc = None
    elif args.dampening == 'log':
        dampfunc = lambda x: int(round(math.log(x + 1, 2)))
    elif args.dampening == 'ones':
        dampfunc = lambda x: 1
    else:
        raise ArgumentException("unknown dampening type '%s'" % args.dampening)

    # Set algorithm parameters
    if args.algorithm == 'viterbi':
        algparams = (args.viterbismooth, args.viterbimaxlen, args.viterbi_allownew)
    else:
        algparams = ()

    # Train model
    if args.trainmode == 'none':
        pass
    elif args.trainmode == 'batch':
        if len(model.get_compounds()) == 0:
            _logger.warning("Model contains no compounds for batch training."
                            " Use 'init+batch' mode to add new data.")
        else:
            if len(args.trainfiles) > 0:
                _logger.warning("Training mode 'batch' ignores new data "
                                "files. Use 'init+batch' or 'online' to "
                                "add new compounds.")
            ts = time.time()
            e, c = model.train_batch(args.algorithm, algparams,
                                     args.finish_threshold, args.maxepochs)
            te = time.time()
            _logger.info("Epochs: %s", e)
            _logger.info("Final cost: %s", c)
            _logger.info("Training time: %.3fs", (te - ts))
    elif len(args.trainfiles) > 0:
        ts = time.time()
        if args.trainmode == 'init':
            if args.list:
                data = io.read_corpus_list_files(args.trainfiles)
            else:
                data = io.read_corpus_files(args.trainfiles)
            c = model.load_data(data, args.freqthreshold, dampfunc,
                                args.splitprob)
        elif args.trainmode == 'init+batch':
            if args.list:
                data = io.read_corpus_list_files(args.trainfiles)
            else:
                data = io.read_corpus_files(args.trainfiles)
            c = model.load_data(data, args.freqthreshold, dampfunc,
                                args.splitprob)
            e, c = model.train_batch(args.algorithm, algparams,
                                     args.finish_threshold, args.maxepochs)
            _logger.info("Epochs: %s", e)
            if args.fullretrain:
                if abs(model.get_corpus_coding_weight() - start_corpus_weight) > 0.1:
                    model.set_corpus_weight_updater(
                        FixedCorpusWeight(model.get_corpus_coding_weight()))
                    model.clear_segmentation()
                    e, c = model.train_batch(args.algorithm, algparams,
                                             args.finish_threshold,
                                             args.maxepochs)
                    _logger.info("Retrain Epochs: %s", e)
        elif args.trainmode == 'online':
            data = io.read_corpus_files(args.trainfiles)
            e, c = model.train_online(data, dampfunc, args.epochinterval,
                                      args.algorithm, algparams,
                                      args.splitprob, args.maxepochs)
            _logger.info("Epochs: %s", e)
        elif args.trainmode == 'online+batch':
            data = io.read_corpus_files(args.trainfiles)
            e, c = model.train_online(data, dampfunc, args.epochinterval,
                                      args.algorithm, algparams,
                                      args.splitprob, args.maxepochs)
            if args.maxepochs is None:
                maxepochs = None
            else:
                maxepochs = args.maxepochs - e
            e, c = model.train_batch(args.algorithm, algparams,
                                     args.finish_threshold, maxepochs)
            _logger.info("Epochs: %s", e)
            if args.fullretrain:
                if abs(model.get_corpus_coding_weight() - start_corpus_weight) > 0.1:
                    model.clear_segmentation()
                    e, c = model.train_batch(args.algorithm, algparams,
                                             args.finish_threshold,
                                             args.maxepochs)
                    _logger.info("Retrain Epochs: %s", e)
        else:
            raise ArgumentException("unknown training mode '%s'", args.trainmode)
        te = time.time()
        _logger.info("Final cost: %s", c)
        _logger.info("Training time: %.3fs", (te - ts))
    else:
        _logger.warning("No training data files specified.")

    # Save model
    if args.savefile is not None:
        io.write_binary_model_file(args.savefile, model)

    if args.savesegfile is not None:
        if args.leftred and args.rightred:
            redup_info = " with left-reduplication and right-reduplication"
        elif args.leftred:
            redup_info = " with left-reduplication"
        elif args.rightred:
            redup_info = " with right-reduplication"
        else:
            redup_info = ""
        io.write_segmentation_file(args.savesegfile, model.get_segmentations(get_trees=args.gettrees), redup_info=redup_info)

    # Output lexicon
    if args.lexfile is not None:
        io.write_lexicon_file(args.lexfile, model.get_constructions())

    if args.savereduced is not None:
        model.make_segment_only()
        io.write_binary_model_file(args.savereduced, model)

    # Segment test data
    if len(args.testfiles) > 0:
        _logger.info("Segmenting test data...")
        outformat = args.outputformat
        outformat = outformat.replace(r"\n", "\n")
        outformat = outformat.replace(r"\t", "\t")
        if "prob:" not in outformat: outformat = outformat.replace("prob", "prob:.3f")
        keywords = {x[1] for x in string.Formatter().parse(outformat) if x[1] is not None}
        if args.nbest > 1 and keywords == {"compound", "analysis"}:
            concatenate_analyses = True
        else:
            concatenate_analyses = False
        with io._open_text_file_write(args.outfile) as fobj:
            testdata = io.read_corpus_files(args.testfiles)
            i = 0
            for compound in testdata:
                if compound is None:
                    # Newline in corpus
                    if args.outputnewlines:
                        fobj.write("\n")
                    continue
                if io.strip_atom_sep:
                    compound_str = "".join(compound)
                else:
                    compound_str = io.atom_separator.join(compound)
                if "clogprob" in keywords or "condprob" in keywords:
                    clogprob = model.forward_logprob(compound,
                                                     addcount=args.viterbismooth,
                                                     maxlen=args.viterbimaxlen,
                                                     allow_new=args.viterbi_allownew)
                else:
                    clogprob = 0
                if args.nbest > 1:
                    nbestlist = model.viterbi_nbest(compound, args.nbest,
                                                    addcount=args.viterbismooth,
                                                    maxlen=args.viterbimaxlen,
                                                    allow_new=args.viterbi_allownew)
                    analyses = []
                    for constructions, logp in nbestlist:
                        analysis = io.format_constructions(constructions)
                        if concatenate_analyses:
                            analyses.append(analysis)
                        else:
                            if "condprob" in keywords:
                                condprob = math.exp(logp - clogprob)
                            else:
                                condprob = 0
                            fobj.write(outformat.format(analysis=analysis,
                                                        compound=compound_str,
                                                        count=compound.count, logprob=logp,
                                                        clogprob=clogprob, condprob=condprob))
                    if concatenate_analyses:
                        all_analyses = args.analysisseparator.join(analyses)
                        fobj.write(outformat.format(analysis=all_analyses, compound=compound_str))
                else:
                    constructions, logp = model.viterbi_segment(
                        compound, addcount=args.viterbismooth, maxlen=args.viterbimaxlen, allow_new=args.viterbi_allownew)
                    if "condprob" in keywords:
                        condprob = math.exp(logp - clogprob)
                    else:
                        condprob = 0
                    analysis = io.format_constructions(constructions)
                    fobj.write(outformat.format(analysis=analysis,
                                                compound=compound_str,
                                                count=compound.count, logprob=logp,
                                                clogprob=clogprob, condprob=condprob))
                i += 1
                if i % 10000 == 0:
                    sys.stderr.write(".")
            sys.stderr.write("\n")
        _logger.info("Done.")

    if args.goldstandard is not None:
        _logger.info("Evaluating Model")
        e = MorfessorEvaluation(io.read_annotations_file(args.goldstandard))
        result = e.evaluate_model(model, meta_data={'name': 'MODEL'}, locations_only=args.evallocations)
        print(result.format(FORMAT_STRINGS['default']))
        _logger.info("Done")

    if args.evalfile is not None:
        # Read segmentations
        segmentations = io.read_segmentation_file(args.evalfile, has_counts=False)
        # Get cost arguments
        addcount = args.viterbismooth
        log_corptokens = model.get_log_smoothedcorpustokens(addcount)
        with io._open_text_file_write(args.outfile) as fobj:
            for compound, parts in segmentations:
                if io.strip_atom_sep:
                    compound_str = str(compound)
                    analysis = " + ".join(str(part) for part in parts)
                else:
                    compound_str = io.atom_separator.join(compound._atoms)
                    analysis = " + ".join(io.atom_separator.join(part) for part in parts)
                badlikelihood = len(compound) * log_corptokens + 1.0
                logp = model.get_segmentation_logp(compound, parts,
                                                   addcount, badlikelihood, log_corptokens,
                                                   red_alternatives=args.score_red_alternatives,
                                                   allow_new=args.viterbi_allownew)
                clogprob = model.forward_logprob(compound,
                                                 addcount=addcount,
                                                 maxlen=args.viterbimaxlen,
                                                 allow_new=args.viterbi_allownew)
                condprob = math.exp(logp - clogprob)
                fobj.write("{compound}\t{analysis}\t{logprob:.3f}\t{condprob:.3f}\n".format(
                    analysis=analysis,
                    compound=compound_str,
                    logprob=logp,
                    clogprob=clogprob,
                    condprob=condprob
                ))


def get_evaluation_argparser():
    import argparse
    #TODO factor out redundancies with get_default_argparser()
    standard_parser = get_default_argparser()
    parser = argparse.ArgumentParser(
        prog="morfessor-evaluate",
        epilog="""Simple usage example:
  %(prog)s gold_standard model1 model2
""",
        description=standard_parser.description,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        add_help=False
    )

    add_arg = parser.add_argument_group('evaluation options').add_argument
    add_arg('--num-samples', dest='numsamples', type=int, metavar='<int>',
            default=10, help='number of samples to take for testing')
    add_arg('--sample-size', dest='samplesize', type=int, metavar='<int>',
            default=1000, help='size of each testing samples')
    add_arg('--eval-properties', dest='evallocations', action='store_false',
            help='If provided, evaluate Splits based on their properties, '
                 'not just their locations')

    add_arg = parser.add_argument_group('formatting options').add_argument
    add_arg('--format-string', dest='formatstring', metavar='<format>',
            help='Python new style format string used to report evaluation '
                 'results. The following variables are a value and and action '
                 'separated with and underscore. E.g. fscore_avg for the '
                 'average f-score. The available values are "precision", '
                 '"recall", "fscore", "samplesize" and the available actions: '
                 '"avg", "max", "min", "values", "count". A last meta-data '
                 'variable (without action) is "name", the filename of the '
                 'model See also the format-template option for predefined '
                 'strings')
    add_arg('--format-template', dest='template', metavar='<template>',
            default='default',
            help='Uses a template string for the format-string options. '
                 'Available templates are: default, table and latex. '
                 'If format-string is defined this option is ignored')

    add_arg = parser.add_argument_group('file options').add_argument
    add_arg('--atom-separator', dest="separator", type=_str, default=None,
            metavar='<regexp>',
            help="atom separator regexp (default %(default)s)")
    add_arg('--construction-separator', dest='constrseparator',
            type=_str, default=' + ', metavar='<str>',
            help="construction separator; plain string, not regexp "
            "(default: '%(default)s')")
    add_arg('--compound-separator', dest="cseparator", type=_str, default=r'\s+',
            metavar='<regexp>',
            help="compound separator regexp (default '%(default)s')")
    add_arg('-e', '--encoding', dest='encoding', metavar='<encoding>',
            help="encoding of input and output files (if none is given, "
                 "both the local encoding and UTF-8 are tried)")
    add_arg('--strip-atom-sep', dest="strip_atom_sep", action="store_true",
            help="remove the atom separator in output files")

    add_arg = parser.add_argument_group('segmentation options').add_argument
    add_arg('-f', '--forcesplit', dest="forcesplit", type=list, default=['-'],
            metavar='<list>',
            help="force split on given atoms (default '-'). The argument "
                 "is a string of single chars, use '' for no forced splits.")
    add_arg('--regexsplit', dest="regexsplit", type=str, default=None,
            metavar='<regexp>',
            help="force split at all boundaries where the provided regex is "
                 "satisfied. Assumes the atom separator is still present. "
                 "(default %(default)s)")
    add_arg('--nosplit-re', dest="nosplit", type=_str, default=None,
            metavar='<regexp>',
            help="if the expression matches the two surrounding characters, "
                 "do not allow splitting (default %(default)s)")
    add_arg('--left-reduplication', dest='leftred', action="store_true",
            help="flag for analyzing left-attaching reduplication")
    add_arg('--Lred-minbase-weight', dest="Lred_minbase_weight",
            type=int, metavar="<int>", default=1,
            help="the minimum weight for the base of left-attaching"
                 " reduplication (default %(default)s)")
    add_arg('--right-reduplication', dest='rightred', action="store_true",
            help="flag for analyzing right-attaching reduplication")
    add_arg('--Rred-minbase-weight', dest="Rred_minbase_weight",
            type=int, metavar="<int>", default=1,
            help="the minimum weight for the base of right-attaching"
                 " reduplication (default %(default)s)")
    add_arg('--default-red-attachment', dest='defaultredattachment', default="L",
            type=str, metavar='<str>',
            help="a string indicating, for cases where both left- and right-attaching"
                 " reduplication are checked, which one should be the default."
                 " Options: L or R (default %(default)s)")
    add_arg('--separate-red', dest='sepred', action="store_true",
            help="flag for separating the <RED> kinds")

    add_arg = parser.add_argument_group('logging options').add_argument
    add_arg('-v', '--verbose', dest="verbose", type=int, default=1,
            metavar='<int>',
            help="verbose level; controls what is written to the standard "
                 "error stream or log file (default %(default)s)")
    add_arg('--logfile', dest='log_file', metavar='<file>',
            help="write log messages to file in addition to standard "
                 "error stream")

    add_arg = parser.add_argument_group('other options').add_argument
    add_arg('-h', '--help', action='help',
            help="show this help message and exit")
    add_arg('--version', action='version',
            version='%(prog)s ' + get_version(),
            help="show version number and exit")

    add_arg = parser.add_argument
    add_arg('goldstandard', metavar='<goldstandard>', nargs=1,
            help='gold standard file in standard annotation format')
    add_arg('models', metavar='<model>', nargs='*', default=[],
            help='model files to segment (either binary or Morfessor 1.0 style'
                 ' segmentation models).')
    add_arg('-t', '--testsegmentation', dest='test_segmentations',
            nargs='*', default=[], metavar='<testseg>',
            help='Files containing alternate segmentations of the test set.'
                 ' Note that all words in the gold-standard must be segmented')

    return parser


def main_evaluation(args):
    """ Separate main for running evaluation and statistical significance
    testing. Takes as argument the results of an get_evaluation_argparser()
    """
    initialize_logging(args)

    io = MorfessorIO(encoding=args.encoding,
                     construction_separator=args.constrseparator,
                     compound_separator=args.cseparator,
                     atom_separator=args.separator,
                     strip_atom_sep=args.strip_atom_sep,
                     nosplit_re=args.nosplit,
                     split_re=args.regexsplit,
                     forcesplit_chars=args.forcesplit,
                     check_left_red=args.leftred,
                     Lred_minbase_weight=args.Lred_minbase_weight,
                     check_right_red=args.rightred,
                     Rred_minbase_weight=args.Rred_minbase_weight,
                     default_red_attachment=args.defaultredattachment,
                     separate_red=args.sepred
                     )

    ev = MorfessorEvaluation(io.read_annotations_file(args.goldstandard[0]))

    results = []

    sample_size = args.samplesize
    num_samples = args.numsamples

    f_string = args.formatstring
    if f_string is None:
        f_string = FORMAT_STRINGS[args.template]

    for f in args.models:
        result = ev.evaluate_model(io.read_any_model(f),
                                   configuration=EvaluationConfig(num_samples,
                                                                  sample_size),
                                   meta_data={'name': os.path.basename(f)},
                                   locations_only=args.evallocations)
        results.append(result)
        print(result.format(f_string))

    for f in args.test_segmentations:
        segmentation = io.read_segmentation_file(f, False)
        result = ev.evaluate_segmentation(segmentation,
                                          configuration=
                                          EvaluationConfig(num_samples,
                                                           sample_size),
                                          meta_data={'name':
                                                     os.path.basename(f)},
                                          locations_only=args.evallocations)
        results.append(result)
        print(result.format(f_string))

    if len(results) > 1 and num_samples > 1:
        wsr = WilcoxonSignedRank()
        r = wsr.significance_test(results)
        WilcoxonSignedRank.print_table(r)