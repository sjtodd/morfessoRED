import collections
import heapq
import logging
import math
import numbers
import random
import re
import copy

from .utils import _progress
from .exception import MorfessorException, SegmentOnlyModelException
from .representations import Split, ChangeSplit, RedSplit, SplitStore, \
    RedSpanStore, Construction, Reduplicant, ConstructionStore

_logger = logging.getLogger(__name__)


def _constructions_to_str(constructions):
    """Return a readable string for a list of constructions.
    Constructions are assumed to be Construction objects"""
    return ' + '.join(constr.label for constr in constructions)


class BaselineModel(object):
    """Morfessor Baseline model class.
    Implements training of and segmenting with a Morfessor model."""

    logzero = -9999.9 # cutoff value for log(0)
    eps = 1e-12 # threshold for testing (in)equality of floating-point numbers

    def __init__(self, use_skips=False,
                 delay_red_checking=False, delay_nonedge_red=True,
                 posthoc_restructuring=False,
                 penalty_weights=None, corpusweight=None,
                 debugging_level=0, debug_delay=1,
                 backup=False, backup_delay=1,
                 analyze_reduplication=False,
                 full_red_only=False,
                 skippable_fullred=False, enforce_fullred=True):
        """Initialize a new model instance.
        Arguments:
            use_skips: randomly skip frequently occurring constructions
                         to speed up training
            delay_red_checking: Boolean flag for whether or not to delay checking
                                for reduplication until after all other options
                                are exhausted
            delay_nonedge_red: like delay_red_checking, but only delays checking
                               for RED instances that are not edge-aligned.
            posthoc_restructuring: Boolean flag for restructuring parse trees when
                                   post-hoc reduplication is identified in the flat
                                   structure. When false, trees are merely relabeled
                                   when the reduplicant and base fall within the same branch.
            penalty_weights: a dict mapping from from construction penalty names to the
                             weight that construction tokens incurring those penalties
                             should be given. Weights are between 0 and 1; 0 indicates that
                             the corresponding constructions are never permitted, and 1
                             indicates that they are not penalized.
            corpusweight: weight for the corpus cost
            debugging_level: variable level (integer) for debugging. 0 represents
                             no debugging (verbose < 2). 1 represents basic debugging
                             (verbose = 2). 2 represents decision-based debugging
                             (verbose = 3). 3 represents complete debugging, including
                             all count adjustments (verbose = 4)
            debug_delay: the number of training epochs that should pass before
                         logging any requested debugging printout.
            backup: a Boolean flag indicating whether or not an analysis should
                    revert to the previous version if it increases cost.
            backup_delay: the number of training epochs that should pass before
                          storing a history of removed splitlocs, and backing up
                          in cases of increased cost (if backup_delay is True).
            analyze_reduplication: a Boolean flag indicating whether or not reduplication
                                   is to be analyzed.
            full_red_only: Boolean. If True, all cases of reduplication will be coerced
                           to full reduplication.
            skippable_fullred: Boolean. If False, lazily forces a separation of the reduplicant and
                              base in cases of full reduplication whenever they are separated
                              from the rest of the construction, provided no better alternative
                              is available. For Viterbi, this means that full reduplication always
                              includes the probability mass for the corresponding unseparated red + base.
                              Essentially, unskippable full red means that any skipped full red is
                              accepted but then posthoc reanalyzed to have red + base separation.
            enforce_fullred: Boolean. If True, greedily enforces full reduplication for all
                             constructions that could be formed entirely from full reduplication.
                             For Viterbi, this means that any path that passes through the red + base
                             outer edges but does not enforce full reduplication is ignored, and its
                             probability mass is discarded. Essentially, enforced full red means that
                             skipped or alternately-analyzed full red is not permitted at all.
        """
        self.debugging_level = debugging_level
        self.backup = backup
        self.backup_delay = backup_delay
        self.debug_delay = debug_delay
        self.total_epochs = 0

        # Initialize flags for reduplication
        self.analyze_reduplication = analyze_reduplication
        self.delay_red_checking = self.analyze_reduplication and delay_red_checking
        self.delay_nonedge_red = self.delay_red_checking or (self.analyze_reduplication and delay_nonedge_red)
        self.posthoc_restructuring = posthoc_restructuring
        self.full_red_only = full_red_only
        self.enforce_fullred = self.analyze_reduplication and enforce_fullred
        self.skippable_fullred = self.enforce_fullred or not(self.analyze_reduplication) or skippable_fullred
        # NB: the conditioning of skippable_fullred on enforce_fullred is to ensure that enforced full reduplication
        # does not gain the probability mass from skipped full reduplication in Viterbi analysis.
        # There is no effect on the Morfessor algorithm because enforcing is greedy, whereas not-skipping is lazy.

        # Create a ConstructionStore
        self._store = ConstructionStore()

        # Flag to indicate the model is only useful for segmentation
        self._segment_only = False

        # Cost variables
        self._lexicon_coding = LexiconEncoding()
        if penalty_weights is None:
            penalty_weights = {}
        self.penalty_weights = penalty_weights
        self._corpus_coding = CorpusEncoding(self._lexicon_coding, logzero=self.logzero, penalty_weights=self.penalty_weights)
        self._annot_coding = None

        # Set corpus weight updater
        self.set_corpus_weight_updater(corpusweight)

        # Configuration variables
        self._use_skips = use_skips  # Random skips for frequent constructions
        self._supervised = False

        # Counter for random skipping
        self._counter = collections.Counter()

        # Used only for (semi-)supervised learning
        self.annotations = None

    def get_log_smoothedcorpustokens(self, addcount):
        """Gets the logarithm of the number of tokens in the corpus,
        taking into account smoothing via addcount

        Arguments:
            addcount: constant for additive smoothing (0 = no smoothing) """
        corpus_tokens = self._corpus_coding.tokens + self._corpus_coding.boundaries - self._corpus_coding.total_token_penalty
        if corpus_tokens > 0:
            smoothing_tokens = addcount * (self._corpus_coding.tokens - self._corpus_coding.total_token_penalty + 1)
            return math.log(corpus_tokens + smoothing_tokens)
        else:
            raise Exception("Attempting to calculate probabilities in a model with no compounds")

    def _debug_permitted(self, priority):
        """A flag for whether to debug at the given priority,
        based on preferences and the total number of epochs
        completed"""
        return priority <= self.debugging_level and self.total_epochs >= self.debug_delay

    @property
    def _backup_permitted(self):
        """A flag for whether it is OK to backup now, based
        on the total number of epochs completed"""
        return self.backup and self.total_epochs >= self.backup_delay

    @property
    def tokens(self):
        """Return the number of construction tokens."""
        return self._corpus_coding.tokens

    @property
    def types(self):
        """Return the number of construction types."""
        return self._corpus_coding.types - 1  # do not include boundary

    def segment(self, construction):
        """Segments a construction by looking it up in the store"""
        self._check_segment_only()
        return self._store.segment(construction)

    def get_segmentations(self, get_trees=False):
        """Retrieve segmentations for all compounds encoded by the model."""
        self._check_segment_only()
        yield from self._store.get_segmentations(get_trees=get_trees)

    def get_compounds(self):
        """Return the compound types stored by the model."""
        self._check_segment_only()
        return self._store.get_compounds()

    def get_constructions(self):
        """Return a list of the present constructions and their counts."""
        return self._store.get_constructions()

    def get_cost(self):
        """Return current model encoding cost."""
        cost = self._corpus_coding.get_cost() + self._lexicon_coding.get_cost()
        if self._supervised:
            return cost + self._annot_coding.get_cost()
        else:
            return cost

    def get_corpus_coding_weight(self):
        return self._corpus_coding.weight

    def set_corpus_coding_weight(self, weight):
        self._check_segment_only()
        self._corpus_coding.weight = weight

    def set_corpus_weight_updater(self, corpus_weight):
        if corpus_weight is None:
            self._corpus_weight_updater = FixedCorpusWeight(1.0)
        elif isinstance(corpus_weight, numbers.Number):
            self._corpus_weight_updater = FixedCorpusWeight(corpus_weight)
        else:
            self._corpus_weight_updater = corpus_weight

        self._corpus_weight_updater.update(self, 0)

    def set_annotations(self, annotations, annotatedcorpusweight=None):
        """Prepare model for semi-supervised learning with given
         annotations.
         """
        self._check_segment_only()
        self._supervised = True
        self.annotations = annotations
        self._annot_coding = AnnotatedCorpusEncoding(self._corpus_coding, weight=annotatedcorpusweight, logzero=self.logzero)
        self._annot_coding.boundaries = len(self.annotations)

    def make_segment_only(self):
        """Reduce the size of this model by removing all non-morphs from the
        analyses. After calling this method it is not possible anymore to call
        any other method that would change the state of the model. Anyway
        doing so would throw an exception.
        """
        self._segment_only = True
        self._store = self._store.remove_nonterminals()

    def clear_segmentation(self):
        """Clears all segmentations in the store and encodings"""
        if self._debug_permitted(1): _logger.debug("Clearing all existing segmentations")
        for compound in self.get_compounds():
            self._clear_analysis(compound)

    def load_data(self, data, freqthreshold=1, count_modifier=None,
                  init_rand_split=None):
        """Load data to initialize the model for batch training.
        Arguments:
            data: iterator of compound Constructions
            freqthreshold: discard compounds that occur less than
                             given times in the corpus (default 1)
            count_modifier: function for adjusting the counts of each
                              compound
            init_rand_split: If given, random split the word with
                               init_rand_split as the probability for each
                               split
        Adds the compounds in the corpus to the model lexicon. Returns
        the total cost.
        """
        self._check_segment_only()
        totalcount = collections.Counter()
        for compound in data:
            if compound is not None:
                totalcount[compound] += compound.count

        for compound, count in totalcount.items():
            if count < freqthreshold:
                continue
            if count_modifier is not None:
                count = count_modifier(count)
            self._add_compound(compound, count)

        # After adding all compounds, randomly split if required
        if init_rand_split is not None and init_rand_split > 0:
            for compound in self.get_compounds():
                self._random_split(compound, init_rand_split)

        return self.get_cost()

    def load_segmentations(self, segmentations):
        """Load model from existing segmentations.
         Arguments:
            segmentations:  list of iterable atoms as string representation of compound segmentation
        Compounds are assumed not to contain any characters that would
        force a split into subcompounds (this will be guaranteed if the
        segmentations were generated by this version of Morfessor).
        """
        self._check_segment_only()
        for compound, segmentation in segmentations:
            self._add_compound(compound)
            splitlist = compound.segmentation_to_splitlist(segmentation)
            self._change_splitlist(compound, splitlist)

    def train_batch(self, algorithm='recursive', algorithm_params=(),
                    finish_threshold=0.005, max_epochs=None):
        """Train the model in batch fashion.
        The model is trained with the data already loaded into the model (by
        using an existing model or calling one of the load_ methods).
        In each iteration (epoch) all compounds in the training data are
        optimized once, in a random order. If applicable, corpus weight,
        annotation cost, and random split counters are recalculated after
        each iteration.
        Arguments:
            algorithm: string in ('recursive', 'viterbi') that indicates
                         the splitting algorithm used.
            algorithm_params: parameters passed to the splitting algorithm.
            finish_threshold: the stopping threshold. Training stops when
                                the improvement of the last iteration is
                                smaller than OR EQUAL TO
                                finish_threshold * #boundaries
            max_epochs: maximum number of epochs to train
        """
        self._check_segment_only()
        epochs = 0
        forced_epochs = max(1, self._epoch_update(epochs))
        newcost = self.get_cost()
        compounds = sorted(self.get_compounds())
        _logger.info("Compounds in training data: %s types / %s tokens",
                     len(compounds), self._corpus_coding.boundaries)

        _logger.info("Starting batch training")
        _logger.info("Epochs: %s\tCost: %s", epochs, newcost)

        while True:
            # One epoch
            random.shuffle(compounds)

            for compound in _progress(compounds):
                if algorithm == 'recursive':
                    self._recursive_optimize(compound, *algorithm_params)
                elif algorithm == 'viterbi':
                    self._viterbi_optimize(compound, *algorithm_params)
                else:
                    raise MorfessorException("unknown algorithm '%s'" %
                                             algorithm)
                if self._debug_permitted(1): _logger.debug("#%s -> %s" % (compound.label, _constructions_to_str(self._store.segment(compound))))
            epochs += 1
            self.total_epochs += 1

            forced_epochs = max(forced_epochs, self._epoch_update(epochs))
            oldcost = newcost
            newcost = self.get_cost()

            _logger.info("Epochs: %s\tCost: %s", epochs, newcost)
            if forced_epochs == 0:
                if newcost >= oldcost - finish_threshold * self._corpus_coding.boundaries: break
            if forced_epochs > 0:
                forced_epochs -= 1
            if max_epochs is not None and epochs >= max_epochs:
                _logger.info("Max number of epochs reached, stop training")
                break
        _logger.info("Done.")
        return epochs, newcost

    def train_online(self, data, count_modifier=None, epoch_interval=10000,
                     algorithm='recursive', algorithm_params=(),
                     init_rand_split=None, max_epochs=None):
        """Train the model in online fashion.
        The model is trained with the data provided in the data argument.
        As example the data could come from a generator linked to standard in
        for live monitoring of the splitting.
        All compounds from data are only optimized once. After online
        training, batch training could be used for further optimization.
        Epochs are defined as a fixed number of compounds. After each epoch (
        like in batch training), the annotation cost, and random split counters
        are recalculated if applicable.
        Arguments:
            data: iterator over compound Constructions.
            count_modifier: function for adjusting the counts of each
                              compound
            epoch_interval: number of compounds to process before starting
                              a new epoch
            algorithm: string in ('recursive', 'viterbi') that indicates
                         the splitting algorithm used.
            algorithm_params: parameters passed to the splitting algorithm.
            init_rand_split: probability for random splitting a compound to
                               at any point for initializing the model. None
                               or 0 means no random splitting.
            max_epochs: maximum number of epochs to train
        """
        self._check_segment_only()
        if count_modifier is not None:
            counts = {}

        _logger.info("Starting online training")

        epochs = 0
        i = 0
        more_tokens = True
        while more_tokens:
            self._epoch_update(epochs)
            newcost = self.get_cost()
            _logger.info("Tokens processed: %s\tCost: %s", i, newcost)

            for _ in _progress(range(epoch_interval)):
                try:
                    compound = next(data)
                except StopIteration:
                    more_tokens = False
                    break

                if compound is None:
                    # Newline in corpus
                    continue

                if count_modifier is not None:
                    if compound not in counts:
                        c = 0
                        counts[compound] = 1
                        addc = 1
                    else:
                        c = counts[compound]
                        counts[compound] = c + 1
                        addc = count_modifier(c + 1) - count_modifier(c)
                    if addc > 0:
                        self._add_compound(compound, addc)
                else:
                    self._add_compound(compound)

                # Make sure to work with stored version
                compound = self._store[compound]
                if init_rand_split is not None and init_rand_split > 0:
                    self._random_split(compound, init_rand_split)

                if algorithm == 'recursive':
                    segments = self._recursive_optimize(compound, *algorithm_params)
                elif algorithm == 'viterbi':
                    segments = self._viterbi_optimize(compound, *algorithm_params)
                else:
                    raise MorfessorException("unknown algorithm '%s'" %
                                             algorithm)
                if self._debug_permitted(1): _logger.debug("#%s: %s -> %s" % (i, compound.label, _constructions_to_str(segments)))
                i += 1

            epochs += 1
            self.total_epochs += 1
            if max_epochs is not None and epochs >= max_epochs:
                _logger.info("Max number of epochs reached, stop training")
                break

        self._epoch_update(epochs)
        newcost = self.get_cost()
        _logger.info("Tokens processed: %s\tCost: %s", i, newcost)
        return epochs, newcost

    def forward_logprob(self, compound, addcount=1.0, maxlen=30, allow_new=False):
        """Find log-probability of a compound using the forward algorithm,
        with the same adjustments as are employed for Viterbi, for consistency.
        Arguments:
          compound: compound to process
          addcount: constant for additive smoothing (0 = no smoothing)
          maxlen: maximum length for the constructions
          allow_new: boolean indicator to allow proposal of a morph (atom) outside of known morphs
        If additive smoothing is applied, new complex construction types are
        considered in deriving the probability. Without smoothing, only the
        shortest permissible new construction types are considered,
        and then only if allow_new is True.
        Also considers reduplication, provided the compound has RedSpans populated.
        If self.skippable_fullred is False, ignores any analyses that separate a full
        red + base combo from the rest of the compound, but do not separate the
        reduplicant from the base.
        if self.enforce_fullred is True, ignores any analyses that pass through the
        edges of a full redspan without being treated as full reduplication. If
        addcount==0, this forces smoothing with an addcount of 1 for known
        constructions and unknown constructions associated with full reduplication.
        Returns the log-probability (negative cost) of the compound.
        If the probability is zero, returns the logzero attribute of the model object.
        NOTE: the smoothing assumes only a single UNK type, encompassing
        all unattested constructions. However, unattested constructions
        that incur a penalty will be evaluated with a correspondingly
        lower probability than is held back for UNK.
        NOTE: this does not account for ChangeSplits, or RedSplits where the
        base undergoes a change.
        """
        smooth_unknown = addcount > 0
        if self.enforce_fullred and addcount == 0:
            addcount = 1
        log_corptokens = self.get_log_smoothedcorpustokens(addcount)
        clen = len(compound)
        badlikelihood = clen * log_corptokens + 1.0
        cost_args = (addcount, badlikelihood, log_corptokens)
        cost_kwargs = {"smooth_unknown": smooth_unknown, "allow_new": allow_new}

        # Check if compound can be skipped by enforcing full reduplication
        if self.enforce_fullred:
            full_redspan = compound.full_redspan
            if full_redspan is not None:
                # Get the cost of the reduplicant and base, forcing new
                # constructions to be allowed.
                # Don't include the cost of not recognizing reduplication,
                # because reduplication is forced
                red = Reduplicant(full_redspan.kind, full_redspan.red_label)
                red_cost = self._get_viterbi_partcost(red, *cost_args) # Will apply smoothing by default
                base = full_redspan.minbase
                base_cost = self._get_viterbi_partcost(base, *cost_args) # Will apply smoothing by default
                # Add boundary cost, adjusted for smoothing
                cost = red_cost + base_cost + log_corptokens - math.log(self._corpus_coding.boundaries)
                return cost

        # Set up grid and required information
        grid = [{None: 0.0}] # {(redspan, is_enforced_fullred): cost}, for each splitloc
        prev_forcedsplitloc = 0
        red_splitlocs_to_redspans = compound.red_splitlocs_to_redspans
        compound_right_red_splitlocs = [splitloc for splitloc, redspans in red_splitlocs_to_redspans.items()
                                        for redspan in redspans if redspan.attachment == "R"]

        # Forward main loop
        for splitloc in range(1, clen + 1):
            # Sum probabilities from all paths to the current node.
            # Note that we can come from any node in history that
            # includes the forced splits.
            if splitloc in compound.badsplits:
                grid.append({None: None})
                continue

            # Keep track of costs in case of right-reduplication
            if splitloc in compound_right_red_splitlocs:
                prev_splitloc_costs = {}
            else:
                prev_splitloc_costs = None

            if self.enforce_fullred:
                fullred_left_edge = compound.potential_fullred_right_to_left_edges.get(splitloc, None)

            # Add paths through all previous viable splits
            psum = 0.0
            for prev_splitloc in range(max(prev_forcedsplitloc, splitloc - maxlen), splitloc):
                if prev_splitloc in compound.badsplits:
                    continue
                cost = grid[prev_splitloc][None]
                construction = compound[prev_splitloc:splitloc]
                partcost = self._get_viterbi_partcost(construction, *cost_args, **cost_kwargs)
                # Ignore paths that pass through a split that can't be made
                ignore_path = cost is None or partcost is None
                # Remove contribution for paths that go through the edges
                # of a possible full redsplit if full redsplits are being enforced
                fullred_enforced = False
                if not(ignore_path) and self.enforce_fullred:
                    if fullred_left_edge is not None and fullred_left_edge <= prev_splitloc:
                        construction = compound.slice(Split(fullred_left_edge), Split(prev_splitloc))
                        construction.redspans = RedSpanStore()
                        logp = self.forward_logprob(construction, addcount=addcount, maxlen=maxlen, allow_new=allow_new)
                        bdry_cost = (log_corptokens - math.log(self._corpus_coding.boundaries))
                        alt_cost = -logp - bdry_cost + grid[fullred_left_edge][None]
                        if alt_cost - cost <= self.eps:
                            ignore_path = True
                        else:
                            cost = -math.log(math.exp(-cost) - math.exp(-alt_cost))
                        fullred_enforced = True
                # Remove contribution for paths that recapitulate
                # reduplication (they are reallocated to reduplication)
                # Note: this is redundant if fullred has been enforced
                if not(ignore_path or fullred_enforced) and prev_splitloc in red_splitlocs_to_redspans:
                    red_psum = 0.0
                    for redspan in red_splitlocs_to_redspans[prev_splitloc]:
                        # For right-attaching RED, remove paths from the right_edge to the
                        # RedSplit, and then to somewhere at or beyond the left_edge
                        if redspan.attachment == "R" and splitloc == redspan.right_edge:
                            for prev_prevsplitloc in range(prev_forcedsplitloc, redspan.left_edge + 1):
                                prev_cost = grid[prev_prevsplitloc][None]
                                prev_construction = compound[prev_prevsplitloc:prev_splitloc]
                                prev_partcost = self._get_viterbi_partcost(prev_construction, *cost_args, **cost_kwargs)
                                if prev_partcost is not None:
                                    red_psum += math.exp(-(prev_cost + prev_partcost))
                        # For left-attaching RED, remove paths from at or beyond the right_edge
                        # to the RedSplit, and then to the left_edge
                        elif redspan.attachment == "L" and splitloc >= redspan.right_edge:
                            prev_prevsplitloc = redspan.left_edge
                            prev_cost = grid[prev_prevsplitloc][None]
                            prev_construction = compound[prev_prevsplitloc:prev_splitloc]
                            prev_partcost = self._get_viterbi_partcost(prev_construction, *cost_args, **cost_kwargs)
                            if prev_partcost is not None:
                                red_psum += math.exp(-(prev_cost + prev_partcost))
                    if red_psum > 0:
                        old_psum = math.exp(-cost)
                        if old_psum - red_psum <= self.eps:
                            ignore_path = True
                        else:
                            cost = -math.log(old_psum - red_psum)
                # Also ignore paths that skip over a valid full_redspan,
                # if full redspans are not skippable
                # (their probability mass is incorporated into the full red path)
                if not (ignore_path or self.skippable_fullred):
                    full_redspan = compound.slice(Split(prev_splitloc), Split(splitloc)).full_redspan
                    if full_redspan is not None:
                        ignore_path = True
                if not ignore_path:
                    cost += partcost
                    # Keep track of costs if required
                    if prev_splitloc_costs is not None:
                        prev_splitloc_costs[prev_splitloc] = cost
                    # Update total probability tracker
                    psum += math.exp(-cost)

                # If looking back to a RedSplit, add that route
                for redspan_info in grid[prev_splitloc]:
                    if redspan_info is not None:
                        # Get cost starting point from RedSplit
                        cost = grid[prev_splitloc][redspan_info]
                        redspan, is_enforced_fullred = redspan_info
                        # If enforcing fullred, check full reduplication separately from normal red
                        if is_enforced_fullred and redspan.reduplicant == redspan.minbase and splitloc == redspan.right_edge:
                            if redspan.attachment == "R":
                                # Part is RED
                                part = Reduplicant(redspan.kind, redspan.red_label)
                            elif redspan.attachment == "L":
                                # Part is base
                                part = redspan.minbase
                            # Path goes through part to split at RedSplit
                            partcost = self._get_viterbi_partcost(part, *cost_args)  # Will apply smoothing by default
                            cost += partcost
                        else:
                            # Add probability mass from skipped full reduplication when unskippable, if applicable
                            # (this is because a skipped full reduplication will be enforced as full red)
                            if not(self.skippable_fullred) and redspan.reduplicant == redspan.minbase and splitloc == redspan.right_edge:
                                alt_cost = grid[redspan.left_edge][None]
                                if alt_cost is not None:
                                    red_and_base = compound[redspan.left_edge:redspan.right_edge]
                                    red_and_base_cost = self._get_viterbi_partcost(red_and_base, *cost_args, **cost_kwargs)
                                    if red_and_base_cost is not None:
                                        alt_cost += red_and_base_cost
                                        # Direct addition to the overall probability is more efficient than incorporating in cost
                                        psum += math.exp(-alt_cost)
                            # Remove contribution from fullred if it should be enforced
                            if self.enforce_fullred and redspan.reduplicant == redspan.minbase and splitloc == redspan.right_edge:
                                if redspan.attachment == "R":
                                    # Contribution is from left edge over base
                                    fullred_partcost = self._get_viterbi_partcost(redspan.minbase, *cost_args, **cost_kwargs)
                                elif redspan.attachment == "L":
                                    # Contribution is from left edge over RED
                                    fullred_partcost = self._get_viterbi_redcost(redspan, *cost_args, **cost_kwargs)
                                fullred_cost = grid[left_edge][None]
                                if fullred_cost is not None and fullred_partcost is not None:
                                    fullred_cost += fullred_partcost
                                    if fullred_cost - cost <= self.eps:
                                        continue
                                    cost = -math.log(math.exp(-cost) - math.exp(-fullred_cost))
                            # For right-attaching RED, only add if currently at the right_edge
                            # since passing through the right_edge is required
                            if redspan.attachment == "R" and splitloc == redspan.right_edge:
                                # Path goes through RED to RedSplit
                                partcost = self._get_viterbi_redcost(redspan, *cost_args, **cost_kwargs)
                                if partcost is None:
                                    continue
                                cost += partcost
                            # For left-attaching RED, only consider if at or beyond the right_edge
                            elif redspan.attachment == "L" and splitloc >= redspan.right_edge:
                                # Path goes through base to RedSplit
                                base = compound[prev_splitloc:splitloc]
                                partcost = self._get_viterbi_partcost(base, *cost_args, **cost_kwargs)
                                if partcost is None:
                                    continue
                                cost += partcost
                            else:
                                continue
                        # Update total probability tracker
                        psum += math.exp(-cost)

            # Add information to grid
            if psum > 0:
                grid.append({None: -math.log(psum)})
            else:
                grid.append({None: None})

            # Add information to the grid for redsplits.
            for redspan in red_splitlocs_to_redspans.get(splitloc, []):
                left_edge = redspan.left_edge
                left_edge_cost = grid[left_edge][None]
                # If enforcing full red, add those first since they are fixed
                if self.enforce_fullred and redspan.reduplicant == redspan.minbase and left_edge_cost is not None:
                    if redspan.attachment == "L":
                        # Part is RED
                        part = Reduplicant(redspan.kind, redspan.red_label)
                    elif redspan.attachment == "R":
                        # Part is base
                        part = redspan.minbase
                    # Path goes through part to split at left edge
                    partcost = self._get_viterbi_partcost(part, *cost_args)  # Will apply smoothing by default
                    cost = left_edge_cost + partcost
                    # Add to grid
                    grid[-1][(redspan, True)] = cost
                if redspan.attachment == "L":
                    # Path goes through RED to split at left edge
                    cost = left_edge_cost
                    if cost is None:
                        continue
                    partcost = self._get_viterbi_redcost(redspan, *cost_args, **cost_kwargs)
                    if partcost is None:
                        continue
                    cost += partcost
                elif redspan.attachment == "R":
                    # Path goes through base, at least to left_edge
                    # Use stored values to add all paths from left_edge and beyond
                    psum = 0.0
                    for prev_splitloc, cost in prev_splitloc_costs.items():
                        if prev_splitloc <= left_edge and cost is not None:
                            psum += math.exp(-cost)
                    # Convert to cost
                    if psum == 0:
                        continue
                    cost = -math.log(psum)
                # Add to grid
                grid[-1][(redspan, False)] = cost

            # Update forced splitloc tracker
            if splitloc in compound.forced_splitlocs:
                prev_forcedsplitloc = splitloc

        cost = grid[-1][None]
        if cost is None:
            # Compound is unsegmentable
            return self.logzero
        else:
            # Add boundary cost, adjusted for smoothing
            cost += (log_corptokens - math.log(self._corpus_coding.boundaries))
            return -cost

    def viterbi_splitlist(self, compound, addcount=1.0, maxlen=30, allow_new=False):
        """Find optimal splitlist using the Viterbi algorithm.
        Arguments:
          compound: compound to be segmented
          addcount: constant for additive smoothing (0 = no smoothing)
          maxlen: maximum length for the constructions
          allow_new: boolean indicator to allow proposal of a morph (atom) outside of known morphs
        If additive smoothing is applied, new complex construction types can
        be selected during the search. Without smoothing, only the
        shortest permissible new construction types can be added,
        and then only if allow_new is True.
        Also considers reduplication, provided the compound has RedSpans populated.
        If self.skippable_fullred is False, final segmentations that skip over a
        full red + base combo are forced to make the full RedSplit.
        if self.enforce_fullred is True, compounds that look like instances of full
        reduplication are treated as such (without running Viterbi), and analyses
        that pass through the edges of the full redspan must be treated as
        full reduplication. If addcount==0, this forces smoothing with an addcount of 1
        for known constructions and unknown constructions associated with full reduplication.
        Returns the most probable splitlist and its cost
        (negative log-probability).
        If no new constructions are allowed, and the compound cannot
        be segmented into known constructions, then returns the
        unsegmented compound, with cost given by -logzero.
        NOTE: the smoothing assumes only a single UNK type, encompassing
        all unattested constructions. However, unattested constructions
        that incur a penalty will be evaluated with a correspondingly
        lower probability than is held back for UNK.
        NOTE: this does not account for ChangeSplits, or RedSplits where the
        base undergoes a change.
        """
        def _passes_through(current_entry, target_splitloc):
            """Returns a Boolean indicating whether the traceback from the
            current entry passes through a split at the target splitloc"""
            _, prev_split, prev_redspan_info = current_entry
            if prev_split is not None:
                while prev_split.splitloc >= target_splitloc:
                    if prev_split.splitloc == target_splitloc:
                        return True
                    _, prev_split, prev_redspan_info = grid[prev_split.splitloc][prev_redspan_info]
            return False

        smooth_unknown = addcount > 0
        if self.enforce_fullred and addcount == 0:
            addcount = 1
        log_corptokens = self.get_log_smoothedcorpustokens(addcount)
        clen = len(compound)
        badlikelihood = clen * log_corptokens + 1.0
        cost_args = (addcount, badlikelihood, log_corptokens)
        cost_kwargs = {"smooth_unknown": smooth_unknown, "allow_new": allow_new}

        # Check if compound can be skipped by enforcing full reduplication
        if self.enforce_fullred:
            full_redspan = compound.full_redspan
            if full_redspan is not None:
                # Get the cost of the reduplicant and base, forcing new
                # constructions to be allowed.
                # Don't include the cost of not recognizing reduplication,
                # because reduplication is forced
                red = Reduplicant(full_redspan.kind, full_redspan.red_label)
                red_cost = self._get_viterbi_partcost(red, *cost_args) # Will apply smoothing by default
                base = full_redspan.minbase
                base_cost = self._get_viterbi_partcost(base, *cost_args) # Will apply smoothing by default
                # Add boundary cost, adjusted for smoothing
                cost = red_cost + base_cost + log_corptokens - math.log(self._corpus_coding.boundaries)
                if self._debug_permitted(2): _logger.debug("Forcing full reduplication in compound %s; cost: %2f" % (compound.label, cost))
                return SplitStore([full_redspan.to_redsplit(forced=True)]), cost

        # Set up Viterbi grid and required information
        grid = [{None: (0.0, None, None)}]  # {redspan_info: (logprob, prev_split, prev_redspan_info)}, where redspan_info = (redspan, is_enforced_full), at each splitloc
        prev_forcedsplitloc = 0
        red_splitlocs_to_redspans = compound.red_splitlocs_to_redspans
        compound_right_red_splitlocs = [splitloc for splitloc, redspans in red_splitlocs_to_redspans.items() for redspan in redspans if redspan.attachment == "R"]

        # Viterbi main loop
        for splitloc in range(1, clen + 1):
            # Select the best path to current node.
            # Note that we can come from any node in history that
            # includes the previous forced split.
            best_prevsplit = Split(prev_forcedsplitloc, forced=True)
            best_cost = None
            best_redspan_info = None
            if splitloc in compound.badsplits:
                grid.append({None: (best_cost, best_prevsplit, best_redspan_info)})
                continue

            # Keep track of costs in case of right-reduplication
            if splitloc in compound_right_red_splitlocs:
                prev_splitloc_costs = {}
            else:
                prev_splitloc_costs = None

            if self.enforce_fullred:
                fullred_left_edge = compound.potential_fullred_right_to_left_edges.get(splitloc, None)

            for prev_splitloc in range(max(prev_forcedsplitloc, splitloc - maxlen), splitloc):
                if prev_splitloc in compound.badsplits:
                    continue
                cost, prev_prevsplit, _ = grid[prev_splitloc][None]
                # Ignore paths that pass through a split that can't be made.
                ignore_path = cost is None
                # Ignore paths that go through the edges of a possible full redsplit
                # if full redsplits are being enforced
                if not(ignore_path) and self.enforce_fullred:
                    if fullred_left_edge is not None and _passes_through(grid[prev_splitloc][None], fullred_left_edge):
                        ignore_path = True
                # Ignore paths that recapitulate reduplication (their probability
                # mass is accounted for via reduplication)
                if not(ignore_path) and prev_splitloc in red_splitlocs_to_redspans:
                    for redspan in red_splitlocs_to_redspans[prev_splitloc]:
                        # For right-attaching RED, ignore paths from the right_edge to the
                        # RedSplit, and then to somewhere at or beyond the left_edge;
                        # For left-attaching RED, ignore paths from at or beyond the right_edge
                        # to the RedSplit, and then to the left_edge
                        if (prev_prevsplit is not None and
                            (
                                redspan.attachment == "R" and
                                splitloc == redspan.right_edge and
                                prev_prevsplit.splitloc <= redspan.left_edge
                            ) or (
                                redspan.attachment == "L" and
                                splitloc >= redspan.right_edge and
                                prev_prevsplit.splitloc == redspan.left_edge
                            )
                        ):
                            ignore_path = True
                            break
                # Ignore paths that skip over a valid full_redspan,
                # if full redsplits are not skippable
                # (their probability mass is incorporated into the full red path)
                if not(ignore_path or self.skippable_fullred):
                    full_redspan = compound.slice(Split(prev_splitloc), Split(splitloc)).full_redspan
                    if full_redspan is not None:
                        ignore_path = True
                # Only check the ordinary splits on this path if not ignoring it
                # (don't just continue because still need to check RED path)
                if not ignore_path:
                    construction = compound[prev_splitloc:splitloc]
                    partcost = self._get_viterbi_partcost(construction, *cost_args, **cost_kwargs)
                    if partcost is not None:
                        cost += partcost
                        # Keep track of costs if required
                        if prev_splitloc_costs is not None:
                            prev_splitloc_costs[prev_splitloc] = cost
                        # Update Viterbi grid
                        if best_cost is None or best_cost - cost > self.eps:
                            best_cost = cost
                            best_prevsplit = Split(prev_splitloc)
                            if prev_splitloc == prev_forcedsplitloc:
                                best_prevsplit.forced = True
                            best_redspan_info = None

                # If looking back to a RedSplit, consider taking that route
                for redspan_info in grid[prev_splitloc]:
                    if redspan_info is not None:
                        # Get cost starting point from RedSplit
                        cost, prev_prevsplit, _ = grid[prev_splitloc][redspan_info]
                        redspan, is_enforced_fullred = redspan_info
                        # If enforcing fullred, check full reduplication separately from normal red
                        if self.enforce_fullred and redspan.reduplicant == redspan.minbase and splitloc == redspan.right_edge and prev_prevsplit.splitloc == redspan.left_edge:
                            # Only let enforced fullreds through, not equivalent non-enforced ones,
                            # so that RedSplits have the appropriate forcing property
                            if not is_enforced_fullred:
                                continue
                            if redspan.attachment == "R":
                                # Part is RED
                                part = Reduplicant(redspan.kind, redspan.red_label)
                            elif redspan.attachment == "L":
                                # Part is base
                                part = redspan.minbase
                            # Path goes through part to split at RedSplit
                            partcost = self._get_viterbi_partcost(part, *cost_args)  # Will apply smoothing by default
                            cost += partcost
                        else:
                            # For right-attaching RED, only consider if currently at the right_edge
                            # since passing through the right_edge is required
                            if redspan.attachment == "R" and splitloc == redspan.right_edge:
                                # Path goes through RED to RedSplit
                                # Also factor in the cost of identical splits not recognized as RED
                                partcost = self._get_viterbi_redcost(redspan, *cost_args, **cost_kwargs)
                                if partcost is None:
                                    continue
                                cost += partcost
                            # For left-attaching RED, only consider if at or beyond the right_edge
                            elif redspan.attachment == "L" and splitloc >= redspan.right_edge:
                                # Path goes through base to RedSplit
                                base = compound[prev_splitloc:splitloc]
                                partcost = self._get_viterbi_partcost(base, *cost_args, **cost_kwargs)
                                if partcost is None:
                                    cost = None
                                else:
                                    cost += partcost
                                # Add probability mass from skipped full reduplication when unskippable, if applicable
                                # (this is because a skipped full reduplication will be enforced as full red)
                                if not(self.skippable_fullred) and redspan.reduplicant == redspan.minbase and splitloc == redspan.right_edge:
                                    alt_cost = grid[redspan.left_edge][None][0]
                                    if alt_cost is not None:
                                        red_and_base = compound[redspan.left_edge:redspan.right_edge]
                                        red_and_base_cost = self._get_viterbi_partcost(red_and_base, *cost_args, **cost_kwargs)
                                        if red_and_base_cost is not None:
                                            alt_cost += red_and_base_cost
                                            if cost is None:
                                                cost = alt_cost
                                            else:
                                                cost = -math.log(math.exp(-cost) + math.exp(-alt_cost))
                            else:
                                continue
                        # Update Viterbi grid if RedSplit gives lower cost
                        if cost is not None and (best_cost is None or best_cost - cost > self.eps):
                            best_cost = cost
                            best_prevsplit = redspan.to_redsplit()
                            if prev_splitloc == prev_forcedsplitloc or is_enforced_fullred:
                                best_prevsplit.forced = True
                            best_redspan_info = redspan_info

            # Update grid
            grid.append({None: (best_cost, best_prevsplit, best_redspan_info)})

            # Add information to the grid for redsplits.
            for redspan in red_splitlocs_to_redspans.get(splitloc, []):
                left_edge = redspan.left_edge
                left_edge_cost = grid[left_edge][None][0]
                # If enforcing full red, add those first since they are fixed
                if self.enforce_fullred and redspan.reduplicant == redspan.minbase and left_edge_cost is not None:
                    if redspan.attachment == "L":
                        # Part is RED
                        part = Reduplicant(redspan.kind, redspan.red_label)
                    elif redspan.attachment == "R":
                        # Part is base
                        part = redspan.minbase
                    # Path goes through part to split at left edge
                    partcost = self._get_viterbi_partcost(part, *cost_args)  # Will apply smoothing by default
                    cost = left_edge_cost + partcost
                    prevsplit = Split(left_edge)
                    if left_edge == prev_forcedsplitloc:
                        prevsplit.forced = True
                    # Add to grid
                    grid[-1][(redspan, True)] = (cost, prevsplit, None)
                if redspan.attachment == "L":
                    # Path goes through RED to split at left edge
                    # Also factor in the cost of identical splits not recognized as RED
                    best_cost = left_edge_cost
                    if best_cost is None:
                        continue
                    partcost = self._get_viterbi_redcost(redspan, *cost_args, **cost_kwargs)
                    if partcost is None:
                        continue
                    best_cost += partcost
                    best_prevsplit = Split(left_edge)
                    if left_edge == prev_forcedsplitloc:
                        best_prevsplit.forced = True
                elif redspan.attachment == "R":
                    # Path goes through base, at least to left_edge
                    # If previously-calculated best is back at or past left_edge, use it;
                    # otherwise, use stored values to find best back at or past left_edge
                    if best_prevsplit.splitloc > left_edge:
                        best_cost = None
                        best_prevsplit = Split(prev_forcedsplitloc, forced=True)
                        for prev_splitloc, cost in prev_splitloc_costs.items():
                            if prev_splitloc <= left_edge:
                                if best_cost is None or best_cost - cost > self.eps:
                                    best_cost = cost
                                    best_prevsplit = Split(prev_splitloc)
                                    if prev_splitloc == prev_forcedsplitloc:
                                        best_prevsplit.forced = True
                    # Try adding probability mass from skipped full reduplication when unskippable, if applicable;
                    # this means testing if a split at the left edge is better than the current best
                    # (this is because a skipped full reduplication will be enforced as full red)
                    if not(self.skippable_fullred) and redspan.reduplicant == redspan.minbase:
                        left_edge_cost = grid[left_edge][None][0]
                        if left_edge_cost is not None:
                            red_and_base = compound[left_edge:redspan.right_edge]
                            red_and_base_cost = self._get_viterbi_partcost(red_and_base, *cost_args, **cost_kwargs)
                            if red_and_base_cost is not None:
                                red_cost = self._get_viterbi_redcost(redspan, *cost_args, **cost_kwargs)
                                if red_cost is not None:
                                    alt_cost = left_edge_cost + red_and_base_cost - red_cost
                                    base_cost = self._get_viterbi_partcost(reduplicant.minbase, *cost_args, **cost_kwargs)
                                    if base_cost is not None:
                                        orig_cost = left_edge_cost + base_cost
                                        cost = -math.log(math.exp(-orig_cost) + math.exp(-alt_cost))
                                    else:
                                        cost = alt_cost
                                    if best_cost is None or best_cost - cost > self.eps:
                                        best_cost = cost
                                        best_prevsplit = Split(left_edge)
                                        if prev_splitloc == prev_forcedsplitloc:
                                            best_prevsplit.forced = True
                # Add to grid
                if best_cost is not None:
                    grid[-1][(redspan, False)] = (best_cost, best_prevsplit, None)

            # Update forced splitloc tracker
            if splitloc in compound.forced_splitlocs:
                prev_forcedsplitloc = splitloc

        endpoint = grid[-1][None]
        if endpoint[0] is None:
            # Compound is not segmentable
            if self._debug_permitted(2): _logger.debug(
                "Viterbi segmentation of compound %s failed; using full compound" % (compound.label,))
            return SplitStore(), -self.logzero
        else:
            # Compound can be segmented by tracing back path
            splits = []
            cost, prev_split, prev_redspan_info = endpoint
            prev_splitloc = prev_split.splitloc
            while prev_splitloc != 0:
                splits.append(prev_split)
                _, prev_split, prev_redspan_info = grid[prev_splitloc][prev_redspan_info]
                prev_splitloc = prev_split.splitloc
            # Add boundary cost, adjusted for smoothing
            cost += log_corptokens - math.log(self._corpus_coding.boundaries)
            if self._debug_permitted(2): _logger.debug("Viterbi traceback for compound %s: %s" % (compound.label, grid))
            return SplitStore(splits), cost

    def _get_viterbi_partcost(self, construction, addcount, badlikelihood, log_corptokens, smooth_unknown=True, allow_new=False):
        """Gets the partial cost of having the designated Construction
        as part of the Viterbi path.

        addcount is the constant for additive smoothing (0 = no smoothing).
        badlikelihood is the cost to be assigned to each atom in the
        event of adding a construction that doesn't already exist, if
        addcount is 0 and allow_new is True.
        log_corptokens is the logarithm of the number of tokens in the corpus
        (taking smoothing and penalties into account).
        If smooth_unknown is False, unknown constructions will not be smoothed"""

        """Gets the partial cost of having the designated Construction
        as part of the Viterbi path.
        Arguments:
          construction: Construction object
          addcount:   constant for additive smoothing (0 = no smoothing)
          badlikelihood:   cost to be assigned to each atom in the
                           event of adding a construction that doesn't already exist, if
                           addcount is 0 and allow_new is True
          log_corptokens:   logarithm of the number of tokens in the corpus
                            (taking smoothing and penalties into account)
          smooth_unknown: boolean indicator to allow smoothing of unknown constructions
          allow_new: boolean indicator to allow proposal of a morph (atom) outside of known morphs
        """

        penalty_logweight = 0
        for penalty in construction.penalties:
            if penalty in self.penalty_weights and self.penalty_weights[penalty] != 1:
                if self.penalty_weights[penalty] == 0:
                    penalty_logweight = self.logzero
                    break
                else:
                    penalty_logweight += math.log(self.penalty_weights[penalty])
        if penalty_logweight == self.logzero:
            return None
        elif (construction in self._store and
              not self._store[construction].has_children):
            if self._store[construction].count <= 0:
                raise MorfessorException(
                    "Construction count of '%s' is %s" %
                    (construction.label,
                     self._store[construction].count))
            log_constrtokens = math.log(self._store[construction].count + addcount) + penalty_logweight
            return log_corptokens - log_constrtokens
        elif addcount > 0 and smooth_unknown:
            log_newtokens = math.log(addcount) + penalty_logweight
            old_ncompounds = self._lexicon_coding.boundaries
            new_ncompounds = old_ncompounds + addcount
            return (log_corptokens - log_newtokens +
                     (new_ncompounds * math.log(new_ncompounds)
                      - old_ncompounds * math.log(old_ncompounds)
                      + self._lexicon_coding.get_codelength(construction))
                     / self._corpus_coding.weight)
        elif allow_new:
            return len(construction) * badlikelihood
        else:
            return None

    def _get_viterbi_redcost(self, redspan, *cost_args, **cost_kwargs):
        """Gets the partial cost of having the reduplicant associated with the
        RedSpan in the given compound Construction as part of the Viterbi path.
        Accounts for the possibility that the reduplicant may not be recognized as RED."""

        """Gets the partial cost of having the reduplicant associated with the
        RedSpan in the given compound Construction as part of the Viterbi path.
        Accounts for the possibility that the reduplicant may not be recognized as RED.
        Arguments:
            redspan: RedSpan object for potential instance of reduplication
            *cost_args: additional arguments passed to the cost calculation method (tuple)
            **cost_kwargs: additional keyword arguments passed 
                            to the cost calculation method (dictionary)

        """

        red = Reduplicant(redspan.kind, redspan.red_label)
        redcost = self._get_viterbi_partcost(red, *cost_args, **cost_kwargs)
        nonred = redspan.reduplicant
        nonredcost = self._get_viterbi_partcost(nonred, *cost_args, **cost_kwargs)
        if redcost is None:
            if nonredcost is None:
                return None
            else:
                return nonredcost
        else:
            if nonredcost is None:
                return redcost
            else:
                return -math.log(math.exp(-redcost) + math.exp(-nonredcost))

    def get_segmentation_logp(self, compound, parts, addcount, badlikelihood, log_corptokens, red_alternatives=True, **cost_kwargs):
        """Gets the log-probability (negative cost) of splitting the provided
        compound Construction into the provided part Constructions.
        If red_alternatives is True, includes the possibility that parts corresponding
        to RED may be recognized as RED or not; if red_alternatives is False, uses
        only the recognition represented by the original parts."""

        """Gets the log-probability (negative cost) of splitting the provided
        compound Construction into the provided part Constructions.
        Arguments:
            compound: compound Construction object to be segmented
            parts: parts of the provided compound Construction in atom form
            addcount:   constant for additive smoothing (0 = no smoothing)
            badlikelihood:   cost to be assigned to each atom in the
                               event of adding a construction that doesn't already exist, if
                               addcount is 0 and allow_new is True
            log_corptokens:   logarithm of the number of tokens in the corpus
                                (taking smoothing and penalties into account)
            red_alternatives:   boolean indicating whether RED should be considered
                                as an alternative underlying form of appropriate parts
            **cost_kwargs: dictionary; additional keyword arguments passed 
                            to the cost calculation method
        """

        cost = 0.0
        cost_args = (addcount, badlikelihood, log_corptokens)
        if red_alternatives and compound.redspans:
            # Consider RED alternatives by testing if each part could be RED
            splitlist = compound.segmentation_to_splitlist(parts)
            splitlocs = (0,) + splitlist.splitlocs + (len(compound),)
            # Get a map from pairs of splitlocs to RedSpans they implicate
            redspan_splitlocs = {tuple(sorted([redspan.splitloc, redspan.red_edge])): redspan
                                 for redspan in compound.redspans}
            # Get cost of parts, counting potential REDs
            for i, splitloc_pair in enumerate(zip(*(splitlocs[j:] for j in range(2)))):
                if splitloc_pair in redspan_splitlocs:
                    cost += self._get_viterbi_redcost(redspan_splitlocs[splitlot_pair], *cost_args, **cost_kwargs)
                else:
                    cost += self._get_viterbi_partcost(parts[i], *cost_args, **cost_kwargs)
        else:
            # Use parts as given
            for part in parts:
                cost += self._get_viterbi_partcost(part, *cost_args, **cost_kwargs)
        # Add boundary cost, adjusted for smoothing
        cost += log_corptokens - math.log(self._corpus_coding.boundaries)
        return -cost

    def viterbi_segment(self, compound, **kwargs):
        """Find optimal segmentation using the Viterbi algorithm.
        Returns the most probable segmentation and its log-probability
        (negative cost).
        NOTE: constructions in segmentation do not inherit properties.
        NOTE: this does not enforce that Splits that recapitulate reduplication
        be treated as RedSplits.
        """

        """Find optimal segmentation using the Viterbi algorithm.
        Returns the most probable segmentation and its log-probability
        (negative cost).

        Arguments:
            compound: compound Construction object to be segmented
            **kwargs: dictionary; additional keyword arguments passed 
                            to the cost calculation method

        NOTE: constructions in segmentation do not inherit properties.
        NOTE: this does not enforce that Splits that recapitulate reduplication
        be treated as RedSplits.
        """

        splitlist, cost = self.viterbi_splitlist(compound, **kwargs)
        constructions = compound.multi_split(splitlist)
        return constructions, -cost

    def _viterbi_nbest_splitlists(self, compound, n, addcount=1.0, maxlen=30, allow_new=False):
        """Find top-n optimal segmentations using the Viterbi algorithm.

        Arguments:
          compound: compound to be segmented
          n: how many segmentations to return
          addcount: constant for additive smoothing (0 = no smoothing)
          maxlen: maximum length for the constructions
          allow_new: boolean indicator to allow proposal of a morph (atom) outside of known morphs

        If additive smoothing is applied, new complex construction types can
        be selected during the search. Without smoothing, only the
        shortest permissible new construction types can be added,
        and then only if allow_new is True.
        Also considers reduplication, provided the compound has RedSpans populated.
        If self.skippable_fullred is False, final segmentations that skip over a
        full red + base combo are forced to make the full RedSplit.
        if self.enforce_fullred is True, compounds that look like instances of full
        reduplication are treated as such (without running Viterbi), and analyses
        that pass through the edges of the full redspan must be treated as
        full reduplication. If addcount==0, this forces smoothing with an addcount of 1
        for known constructions and unknown constructions associated with full reduplication.
        Returns the n most probable splitlists and their costs
        (negative log-probabilities).
        NOTE: the smoothing assumes only a single UNK type, encompassing
        all unattested constructions. However, unattested constructions
        that incur a penalty will be evaluated with a correspondingly
        lower probability than is held back for UNK.
        NOTE: this does not account for ChangeSplits, or RedSplits where the
        base undergoes a change.
        """
        def _passes_through(current_entry, target_splitloc):
            """Returns a Boolean indicating whether the traceback from the
            current entry passes through a split at the given target splitloc"""

            """Returns a Boolean indicating whether the traceback from the
            current entry passes through a split at the given target splitloc.
            Arguments:
                current_entry: dictionary of Viterbi grid; representation of the partial structure
                                of the current word split and the split locations
                target_splitloc: target split location (some kind of coordinate?) 
            """

            _, prev_split, prev_k, prev_redspan_info = current_entry
            if prev_split is not None:
                while prev_split.splitloc >= target_splitloc:
                    if prev_split.splitloc == target_splitloc:
                        return True
                    _, prev_split, prev_k, prev_redspan_info = grid[prev_split.splitloc][prev_redspan_info][prev_k]
            return False

        smooth_unknown = addcount > 0
        if self.enforce_fullred and addcount == 0:
            addcount = 1
        log_corptokens = self.get_log_smoothedcorpustokens(addcount)
        clen = len(compound)
        badlikelihood = clen * log_corptokens + 1.0
        cost_args = (addcount, badlikelihood, log_corptokens)
        cost_kwargs = {"smooth_unknown": smooth_unknown, "allow_new": allow_new}

        # Check if compound can be skipped by enforcing full reduplication
        if self.enforce_fullred:
            full_redspan = compound.full_redspan
            if full_redspan is not None:
                # Get the cost of the reduplicant and base, forcing new
                # constructions to be allowed.
                # Don't include the cost of not recognizing reduplication,
                # because reduplication is forced
                red = Reduplicant(full_redspan.kind, full_redspan.red_label)
                red_cost = self._get_viterbi_partcost(red, *cost_args) # Will apply smoothing by default
                base = full_redspan.minbase
                base_cost = self._get_viterbi_partcost(base, *cost_args) # Will apply smoothing by default
                # Add boundary cost, adjusted for smoothing
                cost = red_cost + base_cost + log_corptokens - math.log(self._corpus_coding.boundaries)
                if self._debug_permitted(2): _logger.debug("Forcing full reduplication in compound %s; cost: %2f" % (compound.label, cost))
                return [SplitStore([full_redspan.to_redsplit(forced=True)]), cost]

        # Set up Viterbi grid and required information
        grid = [{None: [(0.0, None, None, None)]}] # {redspan_info: [(logprob, prev_split, prev_nbest_index, prev_redspan_info), up to n times]}, where redspan_info = (redspan, is_enforced_full), at each splitloc
        prev_forcedsplitloc = 0
        red_splitlocs_to_redspans = compound.red_splitlocs_to_redspans
        compound_right_red_splitlocs = [splitloc for splitloc, redspans in red_splitlocs_to_redspans.items() for redspan in redspans if redspan.attachment == "R"]

        # Viterbi main loop
        for splitloc in range(1, clen + 1):
            # Select the best path to current node.
            # Note that we can come from any node in history that
            # includes the forced splits.
            bestn = []
            if splitloc in compound.badsplits:
                grid.append({None: [(None, Split(prev_forcedsplitloc, forced=True), 0, None)]})
                continue

            # Keep track of negcosts in case of right-reduplication
            if splitloc in compound_right_red_splitlocs:
                prev_splitloc_negcosts = {}
            else:
                prev_splitloc_negcosts = None

            if self.enforce_fullred:
                fullred_left_edge = compound.potential_fullred_right_to_left_edges.get(splitloc, None)

            for prev_splitloc in range(max(prev_forcedsplitloc, splitloc - maxlen), splitloc):
                if prev_splitloc in compound.badsplits:
                    continue
                if prev_splitloc_negcosts is not None:
                    prev_splitloc_negcosts[prev_splitloc] = []

                # Case 1: prev_split is not a RedSplit
                for k in range(len(grid[prev_splitloc][None])):
                    negcost, prev_prevsplit, *_ = grid[prev_splitloc][None][k] # Using negative cost for the heapq implementation
                    # Ignore paths that pass through a split that can't be made
                    ignore_path = negcost is None
                    # Ignore paths that go through the edges of a possible full redsplit
                    # if full redsplits are being enforced
                    if not(ignore_path) and self.enforce_fullred:
                        if fullred_left_edge is not None and _passes_through(grid[prev_splitloc][None][k], fullred_left_edge):
                            ignore_path = True
                    # Ignore paths that recapitulate reduplication (their probability mass
                    # is reallocated to reduplication)
                    if not(ignore_path) and prev_splitloc in red_splitlocs_to_redspans:
                        for redspan in red_splitlocs_to_redspans[prev_splitloc]:
                            # For right-attaching RED, ignore paths from the right_edge to the
                            # RedSplit, and then to somewhere at or beyond the left_edge;
                            # For left-attaching RED, ignore paths from at or beyond the right_edge
                            # to the RedSplit, and then to the left_edge
                            if (prev_prevsplit is not None and
                                    (
                                            redspan.attachment == "R" and
                                            splitloc == redspan.right_edge and
                                            prev_prevsplit.splitloc <= redspan.left_edge
                                    ) or (
                                            redspan.attachment == "L" and
                                            splitloc >= redspan.right_edge and
                                            prev_prevsplit.splitloc == redspan.left_edge
                                    )
                            ):
                                ignore_path = True
                                break
                    # Also ignore paths that skip over a valid full_redspan,
                    # if full redsplits are not skippable
                    # (their probability mass is incorporated into the full red path)
                    if not(ignore_path or self.skippable_fullred):
                        full_redspan = compound.slice(Split(prev_splitloc), Split(splitloc)).full_redspan
                        if full_redspan is not None:
                            ignore_path = True
                    if ignore_path:
                        continue
                    construction = compound[prev_splitloc:splitloc]
                    partcost = self._get_viterbi_partcost(construction, *cost_args, **cost_kwargs)
                    if partcost is None:
                        continue
                    negcost -= partcost
                    # Keep track of negcosts if required
                    if prev_splitloc_negcosts is not None:
                        prev_splitloc_negcosts[prev_splitloc].append(negcost)
                    prev_split = Split(prev_splitloc)
                    if prev_splitloc == prev_forcedsplitloc:
                        prev_split.forced = True
                    if len(bestn) < n:
                        heapq.heappush(bestn, (negcost, prev_split, k, None))
                    else:
                        heapq.heappushpop(bestn, (negcost, prev_split, k, None))

                # Case 2: prev_split is a RedSplit
                for redspan_info in grid[prev_splitloc]:
                    if redspan_info is None:
                        continue
                    redspan, is_enforced_fullred = redspan_info
                    # If enforcing fullred, check full reduplication separately from normal red
                    if is_enforced_fullred and redspan.reduplicant == redspan.minbase and splitloc == redspan.right_edge:
                        if redspan.attachment == "R":
                            # Part is RED
                            part = Reduplicant(redspan.kind, redspan.red_label)
                        elif redspan.attachment == "L":
                            # Part is base
                            part = redspan.minbase
                        # Path goes through part to split at RedSplit
                        partcost = self._get_viterbi_partcost(part, *cost_args)  # Will apply smoothing by default
                    else:
                        # For right-attaching RED, only consider if currently at the right_edge
                        # since passing through the right_edge is required
                        if redspan.attachment == "R" and splitloc == redspan.right_edge:
                            # Path goes through RED to RedSplit
                            # Also factor in the cost of identical splits not recognized as RED
                            partcost = self._get_viterbi_redcost(redspan, *cost_args, **cost_kwargs)
                        # For left-attaching RED, only consider if at or beyond the right_edge
                        elif redspan.attachment == "L" and splitloc >= redspan.right_edge:
                            # Path goes through base to RedSplit
                            base = compound[prev_splitloc:splitloc]
                            partcost = self._get_viterbi_partcost(base, *cost_args, **cost_kwargs)
                            # Add probability mass from skipped full reduplication when unskippable, if applicable
                            # (this is because a skipped full reduplication will be enforced as full red)
                            if not(self.skippable_fullred) and redspan.reduplicant == redspan.minbase and splitloc == redspan.right_edge:
                                red_and_base = compound[redspan.left_edge:redspan.right_edge]
                                red_and_base_cost = self._get_viterbi_partcost(red_and_base, *cost_args, **cost_kwargs)
                                if red_and_base_cost is not None:
                                    red_cost = self._get_viterbi_redcost(redspan, *cost_args, **cost_kwargs)
                                    if red_cost is not None:
                                        if partcost is None:
                                            partcost = red_and_base_cost - red_cost
                                        else:
                                            partcost = -math.log(math.exp(-partcost) + math.exp(-(red_and_base_cost - red_cost)))
                        else:
                            partcost = None
                        if partcost is None:
                            continue
                    prev_split = redspan.to_redsplit()
                    if prev_splitloc == prev_forcedsplitloc or is_enforced_fullred:
                        prev_split.forced = True
                    # Combine the partcost with each of the k negcosts for this RedSplit
                    for k in range(len(grid[prev_splitloc][redspan_info])):
                        negcost, prev_prevsplit, *_ = grid[prev_splitloc][redspan_info][k]
                        # Ensure that fullreds are only found in the fullred stream, if enforcing fullred
                        if self.enforce_fullred and not(is_enforced_fullred) and redspan.reduplicant == redspan.minbase and prev_prevsplit.splitloc == redspan.left_edge:
                            continue
                        negcost -= partcost
                        # Add to heapq
                        if len(bestn) < n:
                            heapq.heappush(bestn, (negcost, prev_split, k, redspan_info))
                        else:
                            heapq.heappushpop(bestn, (negcost, prev_split, k, redspan_info))

            # Add heapq to grid
            if len(bestn) == 0:
                bestn = [(None, Split(0, forced=True), 0, None)]
            grid.append({None: bestn})

            # Add information to the grid for redsplits.
            for redspan in red_splitlocs_to_redspans.get(splitloc, []):
                redspan_bestn = []
                left_edge = redspan.left_edge
                # If enforcing full red, add those first since they are fixed
                if self.enforce_fullred and redspan.reduplicant == redspan.minbase:
                    if grid[left_edge][None][0][0] is not None:
                        prev_split = Split(left_edge)
                        if left_edge == prev_forcedsplitloc:
                            prev_split.forced = True
                        fullred_bestn = []
                        if redspan.attachment == "L":
                            # Part is RED
                            part = Reduplicant(redspan.kind, redspan.red_label)
                        elif redspan.attachment == "R":
                            # Part is base
                            part = redspan.minbase
                        # Path goes through part to split at left edge
                        partcost = self._get_viterbi_partcost(part, *cost_args)  # Will apply smoothing by default
                        for k in range(len(grid[left_edge][None])):
                            negcost = grid[left_edge][None][k][0] - partcost
                            # Add to heapq -- can't exceed numbers cap
                            heapq.heappush(fullred_bestn, (negcost, prev_split, k, None))
                        # Add to grid
                        if len(fullred_bestn) != 0:
                            grid[-1][(redspan, True)] = fullred_bestn
                if redspan.attachment == "L":
                    # Path goes through RED to split at left edge
                    if grid[left_edge][None][0][0] is None:
                        continue
                    # Also factor in the cost of identical splits not recognized as RED
                    partcost = self._get_viterbi_redcost(redspan, *cost_args, **cost_kwargs)
                    if partcost is None:
                        continue
                    prev_split = Split(left_edge)
                    if left_edge == prev_forcedsplitloc:
                        prev_split.forced = True
                    for k in range(len(grid[left_edge][None])):
                        negcost = grid[left_edge][None][k][0] - partcost
                        # Add to heapq -- can't exceed numbers cap
                        heapq.heappush(redspan_bestn, (negcost, prev_split, k, None))
                elif redspan.attachment == "R":
                    # Path goes through base, at least to left_edge
                    # Use stored values for paths back to or past left_edge
                    for prev_splitloc, negcosts in prev_splitloc_negcosts.items():
                        if prev_splitloc <= left_edge:
                            prev_split = Split(prev_splitloc)
                            if prev_splitloc == prev_forcedsplitloc:
                                prev_split.forced = True
                            # In the case where full reduplication is not skippable, calculate an addition
                            # for skipped reduplication to be incorporated into the cost of full reduplication
                            # (this is because a skipped full reduplication will be enforced as full red)
                            alt_partcost = None
                            if prev_splitloc == left_edge and not(self.skippable_fullred) and redspan.reduplicant == redspan.minbase:
                                red_and_base = compound[left_edge:redspan.right_edge]
                                red_and_base_cost = self._get_viterbi_partcost(red_and_base, *cost_args, **cost_kwargs)
                                if red_and_base_cost is not None:
                                    red_cost = self._get_viterbi_redcost(redspan, *cost_args, **cost_kwargs)
                                    if red_cost is not None:
                                        alt_partcost = red_and_base_cost - red_cost
                            # Check the paths through reduplication
                            negcosts = sorted(negcosts)[:n]
                            for k, negcost in enumerate(negcosts):
                                if alt_partcost is not None:
                                    negcost = math.log(math.exp(negcost) + math.exp(negcost - alt_partcost))
                                # Add to heapq
                                item = (negcost, prev_split, k, None)
                                if len(redspan_bestn) < n:
                                    heapq.heappush(redspan_bestn, item)
                                else:
                                    removed = heapq.heappushpop(redspan_bestn, item)
                                    if item == removed:
                                        break
                # Add to grid
                if len(redspan_bestn) != 0:
                    grid[-1][(redspan, False)] = redspan_bestn

            # Update forced splitloc tracker
            if splitloc in compound.forced_splitlocs:
                prev_forcedsplitloc = splitloc

        endpoints = grid[-1][None]
        if endpoints[0][0] is None:
            # Compound is not segmentable
            return [(SplitStore(), -self.logzero)]
        else:
            # Compound can be segmented by tracing back path
            results = []
            for k in range(len(endpoints)):
                splits = []
                negcost, prev_split, prev_k, prev_redspan_info = endpoints[k]
                prev_splitloc = prev_split.splitloc
                while prev_splitloc != 0:
                    splits.append(prev_split)
                    _, prev_split, prev_k, prev_redspan_info = grid[prev_splitloc][prev_redspan_info][prev_k]
                    prev_splitloc = prev_split.splitloc
                # Add boundary cost, adjusted for smoothing
                negcost -= log_corptokens - math.log(self._corpus_coding.boundaries)
                results.append((-negcost, SplitStore(splits)))
            return [(splitlist, cost) for cost, splitlist in sorted(results)]

    def viterbi_nbest(self, compound, n, **kwargs):
        """Find top-n optimal segmentations using the Viterbi algorithm.
        Returns the n most probable segmentations and their log-probabilities
        (negative costs).

        Arguments:
        compound: compound Construction object to be segmented
        n: integer; number of top segmentations to be found
        **kwargs: dictionary; additional keyword arguments passed to the cost calculation method

        NOTE: constructions do not inherit properties.
        NOTE: this does not enforce that Splits that recapitulate reduplication
        be treated as RedSplits.
        """
        splitlists_costs = self._viterbi_nbest_splitlists(compound, n, **kwargs)
        return [(compound.multi_split(splitlist), -cost) for splitlist, cost in splitlists_costs]

    def _viterbi_optimize(self, compound, addcount=0, maxlen=30, allow_new=False):
        """Optimize segmentation of the compound using the Viterbi algorithm.
        Arguments:
          compound: compound to optimize
          addcount: constant for additive smoothing of Viterbi probs
          maxlen: maximum length for a construction
          allow_new: boolean indicator to allow proposal of a morph (atom) outside of known morphs
        """
        orig_cost = self.get_cost()
        # Skip single-atom compounds
        if len(compound) == 1:
            return
        # Skip compounds probabilistically based on frequency
        if self._use_skips:
            if self._test_skip(compound):
                return

        # Be sure to work off stored version
        compound = self._store[compound]
        history = []
        if self._backup_permitted:
            history = self._store.get_history(compound)
        # Use Viterbi algorithm to optimize the compound
        splitlist, _ = self.viterbi_splitlist(compound, addcount=addcount, maxlen=maxlen, allow_new=allow_new)
        # Set splitlist
        self._change_splitlist(compound, splitlist)

        # If the new analysis INCREASES cost, reset to the old one
        final_cost = self.get_cost()
        delta_cost = final_cost - orig_cost
        if self._debug_permitted(2): _logger.debug("Cost change incurred for compound %s optimization: %3f" % (compound.label, delta_cost))
        # Using a small tolerance to deal with floating point arithmetic
        if self._backup_permitted and delta_cost > self.eps:
            if self._debug_permitted(2): _logger.debug("Rejecting analysis of %s and restoring history" % (compound.label,))
            self._restore_history(history)

    def _check_segment_only(self):
        if self._segment_only:
            raise SegmentOnlyModelException()

    def _update_annotation_choices(self):
        """Update the selection of alternative analyses in annotations.
        For semi-supervised models, select the most likely alternative
        analyses included in the annotations of the compounds.
        """
        if not self._supervised:
            return

        # Collect constructions from the most probable segmentations
        # and add missing compounds also to the unannotated data
        constructions = collections.Counter()
        for compound, alternatives in self.annotations.items():
            if compound not in self._store:
                self._add_compound(compound, 1)

            analysis, cost = self._best_analysis(alternatives)
            for part in analysis:
                constructions[part] += self._store[compound].r_count

        # Apply the selected constructions in annotated corpus coding
        self._annot_coding.set_constructions(constructions)
        for part in constructions:
            count = 0
            if part in self._store and not self._store[part].has_children:
                count = self._store[part].count
            self._annot_coding.set_count(part, count)

    def _best_analysis(self, choices):
        """Select the best analysis out of the given choices.

        Arguments:
                choices: list of construction analyses
        """
        bestcost = None
        bestanalysis = None
        corpus_tokens = self._corpus_coding.tokens + self._corpus_coding.boundaries - self._corpus_coding.total_token_penalty
        for analysis in choices:
            cost = 0.0
            for construction in analysis:
                penalty_logweight = 0
                for penalty in construction.penalties:
                    if penalty in self.penalty_weights and self.penalty_weights[penalty] != 1:
                        if self.penalty_weights[penalty] == 0:
                            penalty_logweight = self.logzero
                            break
                        else:
                            penalty_logweight += math.log(self.penalty_weights[penalty])
                if construction in self._store and not self._store[construction].has_children and penalty_logweight != self.logzero:
                    count_logweight = math.log(self._store[construction].count) + penalty_logweight
                    cost += math.log(corpus_tokens) - count_logweight
                else:
                    cost -= self.logzero
            if bestcost is None or cost < bestcost:
                bestcost = cost
                bestanalysis = analysis
        # Add boundary cost
        bestcost += math.log(corpus_tokens) - math.log(self._corpus_coding.boundaries)
        return bestanalysis, bestcost

    def _test_skip(self, construction):
        """Return true if construction should be skipped.
        Arguments:
            construction: Construction object
                """
        if construction in self._counter:
            t = self._counter[construction]
            if random.random() > 1.0 / max(1, t):
                return True
        self._counter[construction] += 1
        return False

    def _epoch_update(self, epoch_num):
        """Do model updates that are necessary between training epochs.
        Arguments:
            epoch_num: the number of training epochs finished in training
        In practice, this does two things:
        - If random skipping is in use, reset construction counters.
        - If semi-supervised learning is in use and there are alternative
          analyses in the annotated data, select the annotations that are
          most likely given the model parameters. If not hand-set, update
          the weight of the annotated corpus.
        This method should also be run prior to training (with the
        epoch number argument as 0).
        """
        forced_epochs = 0
        if self._corpus_weight_updater.update(self, epoch_num):
            forced_epochs += 2

        if self._use_skips:
            self._counter = collections.Counter()
        if self._supervised:
            self._update_annotation_choices()
            self._annot_coding.update_weight()

        return forced_epochs

    def _add_compound(self, compound, count=None):
        """Add compound to data.
        Arguments:
            compound: compound Construction object to be added to data.
            count: provided count; if count = None,  uses the count that came with the compound
        """
        if count is not None:
            compound.r_count = count
            compound.count = count
        # Add to store
        self._store.add(compound)
        # Add to encoding
        self._add_encoding_contributions(compound)
        # Update the boundary tracker
        self._corpus_coding.boundaries += compound.count

    def _add_encoding_contributions(self, construction):
        """Adds the counts for a construction's terminal children
        to the encodings.
        The children are assumed to have already had their contributions
        added to the store.
        Arguments:
            construction: Construction object
        """
        parts = self._store.segment(construction)
        # Combine the counts of repeated parts
        combined_parts = self._combine_counts(parts)
        for part in combined_parts:
            new_count = self._store[part].count
            old_count = new_count - part.count
            self._update_encoding_count(part, old_count, new_count)

    def _remove_encoding_contributions(self, construction):
        """Removes the counts for a construction's terminal children
        from the encodings.
        The children are assumed to exist within the store.
        Arguments:
            construction: Construction object
        """
        parts = self._store.segment(construction)
        # Combine the counts of repeated parts
        combined_parts = self._combine_counts(parts)
        for part in combined_parts:
            old_count = self._store[part].count
            new_count = old_count - part.count
            self._update_encoding_count(part, old_count, new_count)

    def _update_encoding_count(self, construction, old_count, new_count):
        """Updates a Construction's count in the encodings.
            Arguments:
                construction: Construction object with count to be updated; assumed to be terminal
                old_count: count to be replaced in the encodings
                new_count: count to replace the old_count in the encodings
        """
        if old_count == new_count:
            return
        if self._debug_permitted(7):
            if old_count < new_count:
                operation = "Increasing"
            else:
                operation = "Decreasing"
            _logger.debug("%s encoding count of construction %s; %s -> %s" % (operation, construction.label, old_count, new_count))
        self._update_construction_count_trackers(construction, old_count, new_count)
        self._update_atom_count_trackers(construction, old_count, new_count)

    def _combine_counts(self, constructions):
        """Returns a list of Construction objects, which are only guaranteed
        to have the atoms and counts required (i.e. no other properties).
        Arguments:
            constructions: list of Construction objects to have duplicates combined into a single count"""
        # Get the unique constructions
        unique_constructions = set(constructions)
        if len(unique_constructions) == len(constructions):
            return constructions
        else:
            # Combine the counts of repeated parts
            combined_constructions = [constr.__class__(constr._atoms, count=constr.count * constructions.count(constr))
                                      for constr in unique_constructions]
            return combined_constructions

    def _update_construction_count_trackers(self, construction, count, newcount):
        """Updates the construction count trackers in the corpus encoding.
            Arguments:
                construction: Construction object to update count in encoding
                count: current construction count trackers to be updated
                newcount: new construction count trackers to be updated to
        """
        self._corpus_coding.update_count(construction, count, newcount)
        if self._supervised:
            self._annot_coding.update_count(construction, count, newcount)

    def _update_atom_count_trackers(self, construction, count, newcount):
        """Updates the atom count trackers in the lexicon encoding
            Arguments:
                construction: Construction object to update count in encoding
                count: current atom count trackers to be updated
                newcount: new atom count trackers to be updated to
            """
        if count == 0 and newcount > 0:
            self._lexicon_coding.add(construction)
        elif count > 0 and newcount == 0:
            self._lexicon_coding.remove(construction)

    def _remove(self, construction, update_boundary_count=True):
        """Removes a construction, and all its contributions, from both
        the encodings and the store.
            Arguments:
                construction: Construction object to be removed
                update_boundary_count: indicator to update the boundary tracker with removal of Construction object
            """
        self._remove_encoding_contributions(construction)
        self._store.remove(construction)
        # Update the boundary tracker
        if update_boundary_count: self._corpus_coding.boundaries -= construction.r_count

    def _change_splitlist(self, construction, new_splitlist, branching=True, red_delayed=False):
        """Changes the splitlist of a construction in both the store
        and the encodings -- similarly named function in representations
            Arguments:
                construction: Construction object to have splitlist changed
                new_splitlist: new splitlist to be updated to
                branching: if false, has no intermediate interpretations (?)
                red_delayed: only relevant if branching is true
                 if true, reduplication splits occur at the edge of branching(?)
        """
        # Make sure to use the stored version of the construction
        construction = self._store[construction]
        if len(new_splitlist) > 1 and branching:
            splittree = new_splitlist.to_splittree(red_delayed=red_delayed)
            self._set_splittree(construction, splittree)
        else:
            # Only go ahead if the splitlist is not the same as the existing one
            if construction.splitlist != new_splitlist:
                self._remove_encoding_contributions(construction)
                self._store.change_splitlist(construction, new_splitlist, branching=branching, red_delayed=red_delayed)
                self._add_encoding_contributions(construction)

    def _set_splittree(self, construction, splittree, return_history=False):
        """Sets the split tree for a given construction in both the
        store and the encodings
            Arguments:
                construction: Construction object to have split tree set
                splittree: split tree to be set
                return_history: flag -- if true, it returns the history for construction (if there is a need to go back to a prev. store)
                history is list of previous constructions from a previous state of the store (?)
        """
        # Make sure to use the version in the store
        construction = self._store[construction]
        history = []
        if return_history:
            history = self._store.get_history(construction)
        if splittree.is_branching:
            self._split(construction, splittree.split)
            # Recurse to children
            L_child, R_child = self._store[construction].get_children()
            L_subtree, R_subtree = splittree.get_subtrees()
            Lchild_history = self._set_splittree(L_child, L_subtree, return_history=return_history)
            Rchild_history = self._set_splittree(R_child, R_subtree, return_history=return_history)
            history = Rchild_history + Lchild_history + history
        else:
            self._clear_analysis(construction)
        return history

    def _split(self, construction, split):
        """Applies a binary split to a construction in both the store
        and the encodings.
            Arguments:
                    construction: Construction object to be split
                    split: split to be applied to construction
        """
        self._change_splitlist(construction, SplitStore([split]))

    def _clear_analysis(self, construction, return_history=False):
        """Clears the analysis of a construction in the store and encoding,
        but keeps the construction intact with its count.
        If desired (default), returns the history of contributions that
        the construction has to the store before removal; this can be
        reinstated with _restore_history.
        Makes no changes if the construction does not have children
        in the store.

        Arguments:
            construction: Construction object to be cleared of analysis in the store and encoding
            return_history: bool; returns history of contributions that the construction has to the store before removal
        """
        history = []
        if return_history:
            history = self._store.get_history(construction)
        if self._store[construction].has_children:
            # Clear splitlist
            self._change_splitlist(construction, SplitStore())
        return history

    def _restore_history(self, history):
        """Restores the history from a previous state of the store,
        together with corresponding contributions to the encodings

        Arguments:
            history: list of previous constructions from a previous state of the store
        """
        # Reset each construction in the history
        for construction in history:
            if construction.splitlist != self._store[construction].splitlist:
                self._change_splitlist(construction, construction.splitlist, branching=False)

    def _random_split(self, compound, threshold):
        """Enforce a random split for compound Construction.
        The random split always contains any forced splits.
        If full redsplits are not skippable and the random split yields
        an intact part that contains a full RedSplit, that split
        is enforced (unsplit full red + base is not permitted).
        Arguments:
            compound: Construction to split
            threshold: probability of splitting at each position
        """
        forced_splitlocs = list(compound.forced_splitlocs.keys())
        forced_splits = [Split(splitloc, forced=True) for splitloc in forced_splitlocs]
        random_splits = [split for split in compound.valid_splits if split.splitloc not in forced_splitlocs and random.random() < threshold]
        splitlist = SplitStore(forced_splits + random_splits)
        self._change_splitlist(compound, splitlist)
        # Ensure RED has been labeled correctly
        self._posthoc_enforce_red(compound)
        # Check for full red to be enforced
        if self.enforce_fullred or not self.skippable_fullred:
            for child in self._store.segment(compound):
                full_redsplit = child.full_redsplit
                if full_redsplit is not None:
                    self._split(child, full_redsplit)
        if self._debug_permitted(2): _logger.debug("Forced random split on compound %s: %s" % (compound.label, _constructions_to_str(self._store.segment(compound))))

    def _posthoc_enforce_red(self, compound, return_history=False):
        """If the splits of a compound recapitulate a known reduplication,
        this enforces that they be RedSplits (posthoc, i.e. after all
        split selection has completed).
        If self.posthoc_restructuring is False, a reduplicant must be at the
        compound edge in order for the corresponding RedSplit to be
        enforced; thus, a structure like [[mana ti] tia] or [[mana ti][tia Na]]
        would not be considered an instance of reduplication, but something like
        [ti [tia Na]] would be. Conversely, if self.posthoc_restructuring is
        True, any reduplicant that is linearly adjacent to its minimal
        base would have the corresponding RedSplit enforced, and the tree
        would be restructured to ensure that the RedSplit is at the edge.

        Arguments:
            compound: compound to be analyzed for reduplication
            return_history: ?? is this recursion; history is a list of previous constructions from a previous state of the store
        """
        history = []
        if self.analyze_reduplication:
            splittree = self._store.get_red_enforced_splittree(compound, allow_restructuring=self.posthoc_restructuring, full_red_only=self.full_red_only)
            if splittree != self._store.get_splittree(compound):
                if self._debug_permitted(2): _logger.debug("Applied post-hoc enforcement of reduplication to compound %s" % (compound.label,))
                history = self._set_splittree(compound, splittree, return_history=return_history)
        return history

    def _get_splitlist_cost(self, construction, splitlist):
        """Gets the cost of setting the splitlist of a construction.
        Adds contributions to the encoding for cost calculation and then
        removes them; bypasses adjustment of the store.
        **IMPORTANT**: assumes that the existing contributions of the
        construction to the encoding have already been removed.

        Arguments:
            construction: Construction object to set splitlist on
            splitlist: splitlist to get cost on when set on a construction
        """
        dcount = construction.count
        # Get children from making split
        split_children = construction.multi_split(splitlist)
        # Get terminal children and combine them
        part_dcounts = collections.Counter()
        for child in split_children:
            if child not in self._store:
                part_dcounts[child] += dcount
            else:
                for part in self._store.segment(child):
                    part_dcounts[part] += dcount
        # Get original counts
        original_counts = {part: 0 if part not in self._store else self._store[part].count for part in part_dcounts}
        # Adjust counts and get cost
        for part, part_dcount in part_dcounts.items():
            old_count = original_counts[part]
            new_count = old_count + part_dcount
            self._update_encoding_count(part, old_count, new_count)
        cost = self.get_cost()
        # Restore counts
        for part, part_dcount in part_dcounts.items():
            old_count = original_counts[part]
            new_count = old_count + part_dcount
            self._update_encoding_count(part, new_count, old_count)
        return (cost)

    def _get_best_splitlist(self, construction, candidate_splits):
        """Finds the best of the provided candidate Splits for a construction.
        Returns the best Split in a SplitStore, together with any additional
        splits that it requires; this means that non-edge RedSplits will
        be returned together with their corresponding red_edge Split.
        Requires that the construction has already had its analysis cleared,
        in order to get accurate estimates.

        Arguments:
            construction: Construction object to get best Split of
            candidate_splits: list of splits in a SplitStore for provided construction to be compared
        """
        mincost = None
        best_splitlist = None

        for split in candidate_splits:
            # Get a new splitlist, including any required additional split(s)
            splits = [split]
            if isinstance(split, RedSplit):
                # Need to enforce a split at red edge if it doesn't exist already
                if not split.is_at_edge_of(construction):
                    splits.append(Split(split.red_edge))
                # If requiring full reduplication, need to enforce a split at base edge
                if self.full_red_only and not split.has_base_at_edge_of(construction):
                    splits.append(Split(split.minbase_edge))
            splitlist = SplitStore(splits)
            cost = self._get_splitlist_cost(construction, splitlist)
            if self._debug_permitted(6): _logger.debug("Potential splitlist for construction %s: %s; cost: %2f" % (construction.label, splitlist, cost))
            if mincost is None or mincost - cost > self.eps:
                mincost = cost
                best_splitlist = splitlist

        return (best_splitlist, mincost)

    def _recursive_split(self, construction, return_history=False):
        """Optimize the segmentation of a Construction by recursively
        splitting.
        If return_history is True, returns the state of the store
        contributions from the Construction before optimization.

        Arguments:
            construction: Construction object to get optimized segmentation
            return_history: indicator to return state of the store
                            contributions from the Construction before optimization
        """
        # Make sure to use the stored version of the construction
        construction = self._store[construction]
        history = []
        if return_history:
            history = self._store.get_history(construction)
        # Skip constructions that are too short (including Reduplicants)
        if len(construction) == 1:
            return history
        if self._debug_permitted(3): _logger.debug("> Optimizing analysis of construction %s" % (construction.label,))
        # Check if the construction can be skipped through being frequent
        if self._use_skips and self._test_skip(construction):
            if self._debug_permitted(4): _logger.debug("Skipping reanalysis of construction %s" % (construction.label,))
        else:
            valid_splits = list(construction.valid_splits)
            full_redsplit = construction.full_redsplit
            # Check if there are no valid splits for the construction
            if len(valid_splits) == 0:
                if self._debug_permitted(4): _logger.debug("No valid splits in construction %s" % (construction.label,))
                self._clear_analysis(construction)
            # If there are valid splits, enforce fullred if applicable
            elif self.enforce_fullred and full_redsplit is not None:
                    if self._debug_permitted(4): _logger.debug("Enforcing full reduplication in construction %s" % (construction.label,))
                    self._split(construction, full_redsplit)
            else:
                splitlist = None
                # Remove the current contributions to the encoding, but not the store
                # (store updating is slower than encoding updating, so if the analysis
                # isn't going to change, it's better not to reset it first)
                constr_count = construction.count
                orig_part_counts = {part: self._store[part].count for part in self._store.segment(construction)}
                for part, part_count in orig_part_counts.items():
                    self._update_encoding_count(part, part_count, part_count - constr_count)
                # Get cost for no splits
                self._update_encoding_count(construction, 0, constr_count)
                mincost = self.get_cost()
                if self._debug_permitted(5): _logger.debug("Unsplit cost of construction %s: %2f" % (construction.label, mincost))
                # Remove unsplit version from encoding for speed in checking other splits
                self._update_encoding_count(construction, constr_count, 0)

                # Check the valid normal splits
                if splitlist is None:
                    splitlist_normal, mincost_normal = self._get_best_splitlist(construction, valid_splits)
                    if mincost_normal is not None:
                        if self._debug_permitted(5): _logger.debug("Best normal splitlist for construction %s: %s; cost: %2f" % (construction.label, splitlist_normal, mincost_normal))
                        if mincost - mincost_normal > self.eps:
                            splitlist = splitlist_normal
                            mincost = mincost_normal
                # Next check the edge redsplits
                # If delaying red checking, only go ahead if there are no good
                # splits so far
                if splitlist is None or not self.delay_red_checking:
                    valid_edge_redsplits, valid_nonedge_redsplits = construction.get_valid_redsplits()
                    splitlist_edgered, mincost_edgered = self._get_best_splitlist(construction, valid_edge_redsplits)
                    if mincost_edgered is not None:
                        if self._debug_permitted(5): _logger.debug("Best edge-aligned reduplication splitlist for construction %s: %s; cost: %2f" % (construction.label, splitlist_edgered, mincost_edgered))
                        if mincost - mincost_edgered > self.eps:
                            splitlist = splitlist_edgered
                            mincost = mincost_edgered
                # Check the non-edge redsplits if there are no good splits so far
                # or if not delaying red checking
                if splitlist is None or not self.delay_nonedge_red:
                    splitlist_nonedgered, mincost_nonedgered = self._get_best_splitlist(construction, valid_nonedge_redsplits)
                    if mincost_nonedgered is not None:
                        if self._debug_permitted(5): _logger.debug("Best non-edge-aligned reduplication splitlist for construction %s: %s; cost: %2f" % (construction.label, splitlist_nonedgered, mincost_nonedgered))
                        if mincost - mincost_nonedgered > self.eps:
                            splitlist = splitlist_nonedgered
                            mincost = mincost_nonedgered
                # If no good splits have been identified, make a forced split
                # or force a full redsplit if applicable
                if splitlist is None:
                    forced_splits = construction.forced_splits
                    if forced_splits:
                        splitlist, mincost = self._get_best_splitlist(construction, forced_splits)
                        if self._debug_permitted(5): _logger.debug("Best forced splitlist for construction %s: %s; cost: %2f" % (construction.label, splitlist, mincost))
                    elif not self.skippable_fullred and full_redsplit is not None:
                            splitlist = SplitStore([full_redsplit])
                            mincost = self._get_splitlist_cost(construction, splitlist)
                            if self._debug_permitted(5): _logger.debug("Forcing skipped full reduplication in construction %s; cost: %2f" % (construction.label, mincost))

                # Restore the contributions to the encodings
                for part, part_count in orig_part_counts.items():
                    self._update_encoding_count(part, part_count - constr_count, part_count)

                # If no split has been identified, clear the analysis
                if splitlist is None:
                    self._clear_analysis(construction)
                # Otherwise, make the identified split
                else:
                    if self._debug_permitted(4): _logger.debug("Setting splitlist for construction %s to %s; cost: %2f" % (construction.label, splitlist, mincost))
                    self._change_splitlist(construction, splitlist, red_delayed=self.full_red_only)
                    # Recurse into children
                    # Can use a shell representation of children here because they are looked up upon recursing,
                    # and the entries in the store have inherited properties through changing the splitlist
                    children = Construction(construction._atoms, splitlist=splitlist).get_children()
                    for child in children:
                        child_history = self._recursive_split(child, return_history=return_history)
                        history = child_history + history

        if self._debug_permitted(3): _logger.debug("Best analysis of construction %s: %s" % (construction.label, _constructions_to_str(self._store.segment(construction))))
        return history

    def _recursive_optimize(self, compound):
        """Optimize the segmentation of a compound by recursively
        splitting.

        Arguments:
            compound: compound to be segmented optimally
            """
        orig_cost = self.get_cost()
        # Skip single-atom compounds
        if len(compound) == 1:
            return
        # Skip compounds probabilistically based on frequency
        if self._use_skips:
            if self._test_skip(compound):
                return

        if self._debug_permitted(2): _logger.debug(">> Optimizing analysis of compound %s" % (compound.label,))

        # Recursively split
        history = self._recursive_split(compound, return_history=self._backup_permitted)
        # Check for unidentified reduplication
        enforce_red_history = self._posthoc_enforce_red(compound, return_history=self._backup_permitted)
        history = enforce_red_history + history

        # If the new analysis INCREASES cost, reset to the old one
        final_cost = self.get_cost()
        delta_cost = final_cost - orig_cost
        if self._debug_permitted(2): _logger.debug("Cost change incurred for compound %s optimization: %3f" % (compound.label, delta_cost))
        # Using a small tolerance to deal with floating point arithmetic
        if self._backup_permitted and delta_cost > self.eps:
            if self._debug_permitted(2): _logger.debug("Rejecting analysis of %s and restoring history" % (compound.label,))
            self._restore_history(history)


class CorpusWeight(object):
    @classmethod
    def move_direction(cls, model, direction, epoch):
        if direction != 0:
            weight = model.get_corpus_coding_weight()
            if direction > 0:
                weight *= 1 + 2.0 / epoch
            else:
                weight *= 1.0 / (1 + 2.0 / epoch)
            model.set_corpus_coding_weight(weight)
            _logger.info("Corpus weight set to %s", weight)
            return True
        return False


class FixedCorpusWeight(CorpusWeight):
    def __init__(self, weight):
        self.weight = weight

    def update(self, model, _):
        model.set_corpus_coding_weight(self.weight)
        return False


class AnnotationCorpusWeight(CorpusWeight):
    """Class for using development annotations to update the corpus weight
    during batch training
    """

    def __init__(self, devel_set, threshold=0.01):
        self.data = devel_set
        self.threshold = threshold

    def update(self, model, epoch):
        """Tune model corpus weight based on the precision and
        recall of the development data, trying to keep them equal"""
        if epoch < 1:
            return False
        annotations = self.data
        segmentations = {w: [model.viterbi_segment(w)[0]] for w in annotations}
        d = self._estimate_segmentation_dir(segmentations, annotations)

        return self.move_direction(model, d, epoch)

    @classmethod
    def _boundary_recall(cls, prediction, reference):
        """Calculate average boundary recall for given segmentations.
        NB: both prediction and reference are dicts mapping from
        compounds to list of potential segmentations."""
        rec_total = 0
        rec_sum = 0.0
        for compound in prediction:
            pre_list = prediction[compound]
            ref_list = reference[compound]
            best = -1
            for ref in ref_list:
                # list of internal boundary positions
                ref_b = set(compound.segmentation_to_splitlist(ref).splitlocs)
                if len(ref_b) == 0:
                    best = 1.0
                    break
                for pre in pre_list:
                    pre_b = set(compound.segmentation_to_splitlist(pre).splitlocs)
                    r = len(ref_b.intersection(pre_b)) / float(len(ref_b))
                    if r > best:
                        best = r
            if best >= 0:
                rec_sum += best
                rec_total += 1
        return rec_sum, rec_total

    @classmethod
    def _bpr_evaluation(cls, prediction, reference):
        """Return boundary precision, recall, and F-score for segmentations."""
        rec_s, rec_t = cls._boundary_recall(prediction, reference)
        pre_s, pre_t = cls._boundary_recall(reference, prediction)
        rec = rec_s / rec_t
        pre = pre_s / pre_t
        f = 2.0 * pre * rec / (pre + rec)
        return pre, rec, f

    def _estimate_segmentation_dir(self, segmentations, annotations):
        """Estimate if the given compounds are under- or oversegmented.
        The decision is based on the difference between boundary precision
        and recall values for the given sample of segmented data.
        Arguments:
          segmentations: dict mapping from compounds to list of
                         predicted segmentations for that compound
          annotations: dict mapping from compounds to list of reference
                       segmentations (annotations) for that compound
        Return 1 in the case of oversegmentation, -1 in the case of
        undersegmentation, and 0 if no changes are required.
        """
        pre, rec, f = self._bpr_evaluation(segmentations, annotations)
        _logger.info("Boundary evaluation: precision %.4f; recall %.4f", pre, rec)
        if abs(pre - rec) < self.threshold:
            return 0
        elif rec > pre:
            return 1
        else:
            return -1


class MorphLengthCorpusWeight(CorpusWeight):
    def __init__(self, morph_length, threshold=0.01):
        self.morph_length = morph_length
        self.threshold = threshold

    def update(self, model, epoch):
        if epoch < 1:
            return False
        cur_length = self.calc_morph_length(model)

        _logger.info("Current morph-length: %s", cur_length)

        if (abs(self.morph_length - cur_length) / self.morph_length >
                self.threshold):
            d = abs(self.morph_length - cur_length) / (self.morph_length
                                                       - cur_length)
            return self.move_direction(model, d, epoch)
        return False

    @classmethod
    def calc_morph_length(cls, model):
        total_constructions = 0
        total_atoms = 0
        for compound in model.get_compounds():
            total_atoms += len(compound)
            total_constructions += len(model.segment(compound))
        if total_constructions > 0:
            return float(total_atoms) / total_constructions
        else:
            return 0.0


class NumMorphCorpusWeight(CorpusWeight):
    def __init__(self, num_morph_types, threshold=0.01):
        self.num_morph_types = num_morph_types
        self.threshold = threshold

    def update(self, model, epoch):
        if epoch < 1:
            return False
        cur_morph_types = model._lexicon_coding.boundaries

        _logger.info("Number of morph types: %s", cur_morph_types)


        if (abs(self.num_morph_types - cur_morph_types) / self.num_morph_types
                > self.threshold):
            d = (abs(self.num_morph_types - cur_morph_types) /
                 (self.num_morph_types - cur_morph_types))
            return self.move_direction(model, d, epoch)
        return False

class Encoding(object):
    """Base class for calculating the entropy (encoding length) of a corpus
    or lexicon.
    Commonly subclassed to redefine specific methods.
    """
    def __init__(self, weight=1.0, logzero=-9999.9):
        """Initizalize class
        Arguments:
            weight: weight used for this encoding
        """
        self.logtokensum = 0.0
        self.tokens = 0
        self.boundaries = 0
        self.weight = weight
        self.logzero = logzero

    # constant used for speeding up logfactorial calculations with Stirling's
    # approximation
    _log2pi = math.log(2 * math.pi)

    @property
    def types(self):
        """Define number of types as 0. types is made a property method to
        ensure easy redefinition in subclasses
        """
        return 0

    @classmethod
    def _logfactorial(cls, n):
        """Calculate logarithm of n!.
        For large n (n > 20), use Stirling's approximation.
        """
        if n < 2:
            return 0.0
        if n < 20:
            return math.log(math.factorial(n))
        logn = math.log(n)
        return n * logn - n + 0.5 * (logn + cls._log2pi)

    def frequency_distribution_cost(self):
        """Calculate -log[(u - 1)! (v - u)! / (v - 1)!]
        v is the number of tokens+boundaries and u the number of types
        """
        if self.types < 2:
            return 0.0
        tokens = self.tokens + self.boundaries
        return (self._logfactorial(tokens - 1) -
                self._logfactorial(self.types - 1) -
                self._logfactorial(tokens - self.types))

    def permutations_cost(self):
        """The permutations cost for the encoding."""
        return -self._logfactorial(self.boundaries)

    def update_count(self, construction, old_count, new_count):
        """Update the counts in the encoding."""
        self.tokens += new_count - old_count
        if old_count > 1:
            self.logtokensum -= old_count * math.log(old_count)
        if new_count > 1:
            self.logtokensum += new_count * math.log(new_count)

    def get_cost(self):
        """Calculate the cost for encoding the corpus/lexicon"""
        if self.boundaries == 0:
            return 0.0
        
        n = self.tokens + self.boundaries
        return ((n * math.log(n)
                 - self.boundaries * math.log(self.boundaries)
                 - self.logtokensum
                 + self.permutations_cost()) * self.weight
                + self.frequency_distribution_cost())


class CorpusEncoding(Encoding):
    """Encoding the corpus class
    The basic difference to a normal encoding is that the number of types is
    not stored directly but fetched from the lexicon encoding. Also does the
    cost function not contain any permutation cost.
    """
    def __init__(self, lexicon_encoding, weight=1.0, logzero=-9999.9, penalty_weights=None):
        super(CorpusEncoding, self).__init__(weight=weight, logzero=logzero)
        self.lexicon_encoding = lexicon_encoding
        # Set up penalty structure
        if penalty_weights is None:
            penalty_weights = {}
        self.penalty_weights = penalty_weights
        self.total_token_penalty = 0.0

    @property
    def types(self):
        """Return the number of types of the corpus, which is the same as the
         number of boundaries in the lexicon + 1, plus the number of red labels
         that have been identified so far.
        """
        return self.lexicon_encoding.boundaries + 1 + len(self.lexicon_encoding._red_labels)

    def frequency_distribution_cost(self):
        """Calculate -log[(M - 1)! (N - M)! / (N - 1)!] for M types and N
        tokens.
        """
        if self.types < 2:
            return 0.0
        tokens = self.tokens
        return (self._logfactorial(tokens - 1) -
                self._logfactorial(self.types - 2) -
                self._logfactorial(tokens - self.types + 1))

    def update_count(self, construction, old_count, new_count):
        """Update the counts in the encoding, taking stock of penalties.
        To do so, uses logweight instead of logtokens in the logtokensum,
        where logweight is the log of the total weight of the tokens
        in the encoding (penalized tokens are weighted less than 1 each)."""
        self.tokens += new_count - old_count
        # Get the penalties for this construction and update total penalized tokens
        penalty_logweight = 0
        for penalty in construction.penalties:
            if penalty in self.penalty_weights and self.penalty_weights[penalty] != 1:
                if self.penalty_weights[penalty] == 0:
                    penalty_logweight = self.logzero
                    break
                else:
                    penalty_logweight += math.log(self.penalty_weights[penalty])
                self.total_token_penalty += (new_count - old_count) * (1 - self.penalty_weights[penalty])

        if old_count > 1:
            if penalty_logweight == self.logzero:
                self.logtokensum -= old_count * self.logzero
            else:
                old_logweight = math.log(old_count) + penalty_logweight
                self.logtokensum -= old_count * old_logweight

        if new_count > 1:
            if penalty_logweight == self.logzero:
                self.logtokensum += new_count * self.logzero
            else:
                new_logweight = math.log(new_count) + penalty_logweight
                self.logtokensum += new_count * new_logweight

    def get_cost(self):
        """Override for the Encoding get_cost function. A corpus does not
        have a permutation cost.
        Including adjustment for downweighting of penalized tokens.
        This changes n=N+v to n_down=N+v-sum((1-penalty_weight)*penalty_tokens)
        for probability denominator (within log); adjustment of logtokensum
        has already been taken care of in update_count.
        """
        if self.boundaries == 0:
            return 0.0

        n = self.tokens + self.boundaries
        n_down = n - self.total_token_penalty
        
        return ((n * math.log(n_down)
                 - self.boundaries * math.log(self.boundaries)
                 - self.logtokensum) * self.weight
                + self.frequency_distribution_cost())


class AnnotatedCorpusEncoding(Encoding):
    """Encoding the cost of an Annotated Corpus.
    In this encoding constructions that are missing are penalized.
    """
    def __init__(self, corpus_coding, weight=None, logzero=-9999.9, penalty_weights=None):
        """
        Initialize encoding with appropriate meta data
        Arguments:
            corpus_coding: CorpusEncoding instance used for retrieving the
                             number of tokens and boundaries in the corpus
            weight: The weight of this encoding. If the weight is None,
                      it is updated automatically to be in balance with the
                      corpus
            penalty: log penalty used for missing constructions
        """
        super(AnnotatedCorpusEncoding, self).__init__(logzero=logzero)
        self.do_update_weight = True
        self.weight = 1.0
        if weight is not None:
            self.do_update_weight = False
            self.weight = weight
        self.corpus_coding = corpus_coding
        self.constructions = collections.Counter()
        # Set up penalty structure
        if penalty_weights is None:
            penalty_weights = {}
        self.penalty_weights = penalty_weights

    def set_constructions(self, constructions):
        """Method for re-initializing the constructions. The count of the
        constructions must still be set with a call to set_count
        """
        self.constructions = constructions
        self.tokens = sum(constructions.values())
        self.logtokensum = 0.0

    def set_count(self, construction, count):
        """Set an initial count for each construction. Missing constructions
        are penalized
        """
        annot_count = self.constructions[construction]
        penalty_logweight = 0
        for penalty in construction.penalties:
            if penalty in self.penalty_weights and self.penalty_weights[penalty] != 1:
                if self.penalty_weights[penalty] == 0:
                    penalty_logweight = self.logzero
                    break
                else:
                    penalty_logweight += math.log(self.penalty_weights[penalty])

        if count > 0 and penalty_logweight != self.logzero:
            count_logweight = math.log(count) + penalty_logweight
            self.logtokensum += annot_count * count_logweight
        else:
            self.logtokensum += annot_count * self.logzero

    def update_count(self, construction, old_count, new_count):
        """Update the counts in the Encoding, setting (or removing) a penalty
         for missing constructions
        """
        if construction in self.constructions:
            annot_count = self.constructions[construction]
            penalty_logweight = 0
            for penalty in construction.penalties:
                if penalty in self.penalty_weights and self.penalty_weights[penalty] != 1:
                    if self.penalty_weights[penalty] == 0:
                        penalty_logweight = self.logzero
                        break
                    else:
                        penalty_logweight += math.log(self.penalty_weights[penalty])

            if old_count > 0 and penalty_logweight != self.logzero:
                old_logweight = math.log(old_count) + penalty_logweight
                self.logtokensum -= annot_count * old_logweight
            else:
                self.logtokensum -= annot_count * self.logzero

            if new_count > 0 and penalty_logweight != self.logzero:
                new_logweight = math.log(new_count) + penalty_logweight
                self.logtokensum += annot_count * new_logweight
            else:
                self.logtokensum += annot_count * self.logzero

    def update_weight(self):
        """Update the weight of the Encoding by taking the ratio of the
        corpus boundaries and annotated boundaries
        """
        if not self.do_update_weight:
            return
        old = self.weight
        self.weight = (self.corpus_coding.weight *
                       float(self.corpus_coding.boundaries) / self.boundaries)
        if self.weight != old:
            _logger.info("Corpus weight of annotated data set to %s", self.weight)

    def get_cost(self):
        """Return the cost of the Annotation Corpus."""
        if self.boundaries == 0:
            return 0.0
        n = self.tokens + self.boundaries
        corpus_tokens = (self.corpus_coding.tokens
                         + self.corpus_coding.boundaries
                         - self.corpus_coding.total_token_penalty)
        return ((n * math.log(corpus_tokens)
                 - self.boundaries * math.log(self.corpus_coding.boundaries)
                 - self.logtokensum) * self.weight)


class LexiconEncoding(Encoding):
    """Class for calculating the encoding cost for the Lexicon"""

    def __init__(self):
        """Initialize Lexcion Encoding"""
        super(LexiconEncoding, self).__init__()
        self.atoms = collections.Counter()
        self._red_labels = set()

    @property
    def types(self):
        """Return the number of different atoms in the lexicon + 1 for the
        compound-end-token
        """
        return len(self.atoms) + 1

    def add(self, construction):
        """Add a construction to the lexicon, updating automatically the
        count for its atoms
        """
        if isinstance(construction, Reduplicant):
            self._red_labels.add(construction.label)
        else:
            self.boundaries += 1
            for atom in construction:
                c = self.atoms[atom]
                self.atoms[atom] = c + 1
                self.update_count(atom, c, c + 1)

    def remove(self, construction):
        """Remove construction from the lexicon, updating automatically the
        count for its atoms
        """
        if isinstance(construction, Reduplicant):
            self._red_labels.remove(construction.label)
        else:
            self.boundaries -= 1
            for atom in construction:
                c = self.atoms[atom]
                self.atoms[atom] = c - 1
                self.update_count(atom, c, c - 1)

    def get_codelength(self, construction):
        """Return an approximate codelength for new construction."""
        # Reduplicants are assumed to be zero-cost
        if isinstance(construction, Reduplicant):
            return 0.0
        l = len(construction) + 1
        cost = l * math.log(self.tokens + l)
        cost -= math.log(self.boundaries + 1)
        for atom in construction:
            if atom in self.atoms:
                c = self.atoms[atom]
            else:
                c = 1
            cost -= math.log(c)
        return cost