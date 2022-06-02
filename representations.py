import copy
import re
import itertools

class Split(object):
    """A container object for a simple split.
    Used mainly to promote consistency with special kinds of splits."""
    __slots__ = ('splitloc', 'forced')

    def __init__(self, splitloc=0, forced=False):
        """Initializes by storing splitloc; default value is 0 (no split)"""
        self.splitloc = splitloc
        self.forced = forced

    def __repr__(self):
        """Provides a string representation of the object"""
        cls_name = type(self).__name__
        if self.forced:
            cls_name = "Forced" + cls_name
        splitloc = self.splitloc
        other_attr = ", ".join("{}={}".format(attr, self.__getattribute__(attr)) for cls in type(self).mro() for attr in
                               getattr(cls, '__slots__', ()) if
                               (attr not in ["splitloc", "forced"] and self.__getattribute__(attr) is not None))
        if other_attr:
            other_attr = ", " + other_attr
        return "{}({}{})".format(cls_name, splitloc, other_attr)

    def __bool__(self):
        """Return False iff splitloc == 0"""
        return self.splitloc != 0

    def __lt__(self, other):
        """For sorting, use the splitloc"""
        return self.splitloc < other.splitloc

    def __gt__(self, other):
        """For comparisons, use the splitloc"""
        return self.splitloc > other.splitloc

    def __eq__(self, other):
        """For object equality, compare the attributes"""
        return type(self) == type(other) and all(
            self.__getattribute__(attr) == other.__getattribute__(attr) for cls in type(self).mro() for attr in
            set(getattr(cls, '__slots__', ())) - {"forced", "new_badsplits"})

    def __hash__(self):
        """For hashing, use the splitloc"""
        return hash(self.splitloc)

    def __copy__(self):
        """A method for copying, to improve speed"""
        return Split(self.splitloc, self.forced)

    def reindex(self, other_split):
        """Reindexes by subtracting the splitloc of the other split.
        Returns a new Split object

        Arguments:
            other_split: split with location to be reindexed to"""
        new_split = copy.copy(self)
        new_split.splitloc -= other_split.splitloc
        return new_split

    def is_left_of(self, other_split):
        """Checks whether a Split is left of another"""
        return isinstance(other_split, Split) and self < other_split

    def is_right_of(self, other_split):
        """Checks whether a Split is right of another"""
        return isinstance(other_split, Split) and self > other_split

    def trim_R_newform(self, other_split):
        """A placeholder for a method that is only valid for ChangeSplits"""
        return self


class ChangeSplit(Split):
    """A container for a split that introduces a change to the form
    of its left and/or right child"""
    __slots__ = ('L_newform', 'R_newform')

    def __init__(self, splitloc, L_newform=None, R_newform=None, forced=False):
        """Initializes by storing data

        Arguments:
            splitloc: split location
            L_newform: form of left child (default = None)
            R_newform: form of right child (default = None)
            forced: forced (? split) (default = False)
        """
        # Store splitloc based on split
        super().__init__(splitloc, forced=forced)
        # Store form changes
        self.L_newform = L_newform
        self.R_newform = R_newform

    def __copy__(self):
        """A method for copying, to improve speed"""
        return ChangeSplit(self.splitloc, self.L_newform, self.R_newform, self.forced)

    def reindex(self, other_split):
        """Reindexes the current split by subtracting the splitloc of the
        provided other_split, and trimming the non-RED L_newform.

        Arguments:
            other_split: split with location to be trimmed at
            """
        new_split = super().reindex(other_split)
        if not (new_split.L_newform is None or isinstance(new_split.L_newform, Reduplicant)):
            new_split.L_newform = new_split.L_newform[other_split.splitloc:]
        return new_split

    def trim_R_newform(self, other_split):
        """Trims a non-RED R_newform, if it exists, to remove any
        atoms that are to the right of other_split.splitloc

        Arguments:
            other_split: split with location to remove right-side atoms"""
        new_split = copy.copy(self)
        if not (new_split.R_newform is None or isinstance(new_split.R_newform, Reduplicant)):
            new_split.R_newform = new_split.R_newform[:other_split.splitloc - new_split.splitloc]
        return new_split


class RedSplit(ChangeSplit):
    """A container for a split that introduces reduplication"""
    __slots__ = ('red_edge', 'minbase_edge', 'new_badsplits')

    def __init__(self, splitloc, red_edge, minbase_edge, new_badsplits, L_newform=None, R_newform=None, forced=False):
        """Initializes by storing data.
        new_badsplits is a tuple of splitlocs (ints) to be avoided

        Arguments:
            splitloc: split location
            red_edge: reduplication edge location
            minbase_edge: minbase edge location
            new_badsplits: tuple of splitlocs (ints) to be avoided
            L_newform: left new form (default = None)
            R_newform: right new form (default = None)
            forced: forced (? split) (default = False)
        """
        # Store splitloc and provided changed forms
        super().__init__(splitloc, L_newform=L_newform, R_newform=R_newform, forced=forced)
        # Store RED-specific info
        self.red_edge = red_edge
        self.minbase_edge = minbase_edge
        self.new_badsplits = new_badsplits

    def __copy__(self):
        """A method for copying, to improve speed"""
        return RedSplit(self.splitloc, self.red_edge, self.minbase_edge, self.new_badsplits, self.L_newform, self.R_newform, self.forced)

    @property
    def red_start(self):
        """Get the index of the start of RED"""
        return min(self.splitloc, self.red_edge)

    @property
    def red_end(self):
        """Get the index of the end of RED"""
        return max(self.splitloc, self.red_edge)

    @property
    def attachment(self):
        """Gets the side to which RED attaches: L or R"""
        if self.red_edge < self.splitloc:
            return ("L")
        elif self.red_edge > self.splitloc:
            return ("R")

    def reindex(self, other_split):
        """Reindexes by subtracting the splitloc of the other split,
        also updating the non-RED L_newform, red_edge and new_badsplits.
        Returns a new RedSplit object

        Arguments:
            other_split: split to set new split location for red_edge"""
        new_split = super().reindex(other_split)
        new_split.red_edge -= other_split.splitloc
        new_split.new_badsplits = tuple(badsplit - other_split.splitloc for badsplit in new_split.new_badsplits)
        return new_split

    def is_left_of(self, other_split):
        """Checks whether a RedSplit is left of another Split"""
        return isinstance(other_split, Split) and self < other_split and self.red_edge <= other_split.splitloc

    def is_right_of(self, other_split):
        """Checks whether a RedSplit is right of another Split"""
        return isinstance(other_split, Split) and self > other_split and self.red_edge >= other_split.splitloc

    def is_at_edge_of(self, construction):
        """Checks whether a RedSplit is at the edge of a Construction"""
        return self.red_edge == 0 or self.red_edge == len(construction)

    def has_base_at_edge_of(self, construction):
        """Checks whether the minbase of the RedSplit is at the edge
        of a Construction"""
        base_edge = self.minbase_edge
        return base_edge == 0 or base_edge == len(construction)


class SplitStore(tuple):
    """A container for storing the splits associated with a construction.
    The underlying data structure is a tuple."""
    __slots__ = ()

    def get(self, splitloc):
        """Gets the Split object associated with a given splitloc.
        If there is no such Split, returns None

        Arguments:
            splitloc: split location to retrieve associated Split object
        """
        for split in self:
            if split.splitloc == splitloc:
                return split
        return None

    @property
    def splitlocs(self):
        """Returns a tuple of the splitlocs of all Splits in the store"""
        return tuple(split.splitloc for split in self)

    def reindex(self, other_split):
        """Returns a new SplitStore in which all of the splits
        have been reindexed by subtracting the splitloc of other_split.

        Arguments:
            other_split: split with location to be reindexed to
        """
        return SplitStore(split.reindex(other_split) for split in self)

    def slice(self, start_split, end_split=None):
        """Returns a new SplitStore containing all of the Splits that fall
        entirely within the range spanned by start_split.splitloc and
        end_split.splitloc (defaults to end of construction), reindexed
        according to start_split (and with any non-RED R_newforms trimmed
        according to end_split).
        Only includes RedSplits if the start or end splits are not in their
        new badsplits.

        Arguments:
            start_split: split with location to begin slice at
            end_split: split with location to end slice at (default = None)
        """
        # If there are no splits, return an empty SplitStore
        if not self:
            return SplitStore()
        # Otherwise, slice as required
        sliced_splits = (split for split in self if split.is_right_of(start_split) and not (
                    isinstance(split, RedSplit) and start_split.splitloc in split.new_badsplits))
        if end_split is not None:
            sliced_splits = (split.trim_R_newform(end_split) for split in sliced_splits if
                             split.is_left_of(end_split) and not (
                                         isinstance(split, RedSplit) and end_split.splitloc in split.new_badsplits))
        modified_splits = SplitStore(sliced_splits).reindex(start_split)
        return modified_splits

    def partition(self, other_split):
        """Returns a pair of SplitStores, (L_splitlist, R_splitlist),
        containing all of the splits to the left and right of the provided
        other_split, respectively. Splits to the right are reindexed,
        and any ChangeSplits to the left have their non-RED R_newforms
        shortened.

        Arguments:
            other_split: split with location to partition
        """
        L_splitlist = SplitStore(split.trim_R_newform(other_split) for split in self if split.is_left_of(other_split))
        R_splitlist = SplitStore(split.reindex(other_split) for split in self if split.is_right_of(other_split))
        return (L_splitlist, R_splitlist)

    def to_splittree(self, red_delayed=False):
        """Converts a SplitStore into a SplitTree, assuming primarily
        right-branching structure, except that RedSplits are constrained
        to be made after their corresponding edge splits, so that each
        instance of RED is introduced at the edge of a construction.
        If red_delayed is True, RedSplits are additionally delayed so
        that they introduce terminal nodes, i.e. so that the red + base
        combination forms its own construction.
        Note: this will not retrieve splits from a construction store; only the
        splits in the existing splitlist for this construction will be used.

        Arguments:
            red_delayed: boolean to additional delay RedSplits so that
                         they introduce terminal nodes (default = False)
        """
        splittree = SplitTree(terminal=True)
        # Build bottom-up
        for split in sorted(self, reverse=True):
            splittree = SplitTree(split=split, R_subtree=splittree)
        # Make structural enforcements
        if red_delayed:
            splittree = splittree.enforce_delayed_red()
        else:
            splittree = splittree.enforce_edge_red()
        return splittree


class SplitTree(object):
    """A container for storing a tree of binary Splits associated with
    a construction. Splits are indexed to the construction as a whole,
    not to its parts. Behaves like a namedtuple, with attributes split,
    L_subtree, and R_subtree."""
    __slots__ = ('split', 'L_subtree', 'R_subtree')

    def __init__(self, split=None, L_subtree=None, R_subtree=None, terminal=False):
        """Creates a new SplitTree object

        Arguments:
            split = split object
            L_subtree = SplitTree object left of new SplitTree (default = None)
            R_subtree = SplitTree object right of new SplitTree (default = None)
            terminal = boolean to indicate final SplitTree object (?) (default = False)
        """
        if not terminal:
            if L_subtree is None:
                L_subtree = SplitTree(terminal=True)
            if R_subtree is None:
                R_subtree = SplitTree(terminal=True)
        self.split = split
        self.L_subtree = L_subtree
        self.R_subtree = R_subtree

    def __iter__(self):
        """Iterate over subtrees"""
        if self.is_branching:
            yield self.L_subtree
            yield self.R_subtree

    def __repr__(self):
        """String representation: like a namedtuple"""
        if self.is_branching:
            return "SplitTree({}, L={}, R={})".format(self.split, repr(self.L_subtree), repr(self.R_subtree))
        else:
            return "SplitTree()"

    def __str__(self):
        """Pretty printing: indented"""
        return self._pretty_print()

    def __eq__(self, other_tree):
        """Two trees are equal if they have the same splits and same children"""
        if not isinstance(other_tree, SplitTree):
            return False
        if self.is_branching:
            return (self.split == other_tree.split and
                    self.L_subtree == other_tree.L_subtree and
                    self.R_subtree == other_tree.R_subtree)
        else:
            return not other_tree.is_branching

    def __hash__(self):
        """For hashing, use the corresponding splitlist"""
        return hash(self.to_splitlist())

    def _pretty_print(self, indent_level=0):
        """Pretting printing: indented"""
        indent = "   " * indent_level
        if self.is_branching:
            return "{}\n{}{}\n{}".format(self.R_subtree._pretty_print(indent_level=indent_level + 1),
                                         indent, "{}({})".format(type(self.split).__name__[0], self.split.splitloc),
                                         self.L_subtree._pretty_print(indent_level=indent_level + 1))
        else:
            return "{}xxx".format(indent)

    @property
    def is_branching(self):
        """Checks if the tree is branching"""
        return self.split is not None

    @property
    def _contained_splits(self):
        """Returns a generator over all of the Split objects contained in
        the SplitTree; that is, the split itself, plus the splits of the
        left and right subtrees."""
        if self.is_branching:
            yield from self.L_subtree._contained_splits
            yield self.split
            yield from self.R_subtree._contained_splits

    def is_ancestor(self, split1, split2):
        """Checks if split1 is ancestor of split2 in this SplitTree.
        split1 is an ancestor if split2 is contained within the subtree
        corresponding to split1.
        Note: this uses the Splits from the tree that have the same splitloc
        as split1 and split2, not split1 and split2 themselves.

        Arguments:
            split1: Split to check if is an ancestor to split2
            split2: Split to check if is in split1 subtree
        """
        subtree = self.get(split1.splitloc)
        split2 = self.to_splitlist().get(split2.splitloc)
        return split2 in subtree._contained_splits

    def get_terminal_parents(self):
        """Returns a generator over the parent of each terminal node,
        paired with the side of the subtree that is the terminal node"""
        if self.is_branching:
            if self.L_subtree.is_branching:
                yield from self.L_subtree.get_terminal_parents()
            else:
                yield (self, "L")
            if self.R_subtree.is_branching:
                yield from self.R_subtree.get_terminal_parents()
            else:
                yield (self, "R")

    def _get_subtree(self, side):
        """Gets the subtree on the given side

        Arguments:
            side: side to retrieve subtree
        """
        if side == "L":
            return self.L_subtree
        elif side == "R":
            return self.R_subtree

    def _set_subtree(self, side, new_subtree):
        """Sets the subtree on the given side

        Arguments:
            side: side to set subtree
            new_subtree: subtree to be set
        """
        if side == "L":
            self.L_subtree = new_subtree
        elif side == "R":
            self.R_subtree = new_subtree

    def reindex(self, split):
        """Reindex all splits in the SplitTree by the given Split.
        Arguments:
            split: Split to be used to reindex splits in SplitTree"""
        if self.is_branching:
            return SplitTree(split=self.split.reindex(split),
                             L_subtree=self.L_subtree.reindex(split),
                             R_subtree=self.R_subtree.reindex(split))
        else:
            return SplitTree(terminal=True)

    def to_splitlist(self):
        """Converts a tree into a flat splitlist"""
        return SplitStore(self._contained_splits)

    def replace(self, target_split, replacement_subtree):
        """Returns a new SplitTree object, in which the subtree introduced by
         a split at the splitloc of a provided Split is replaced by a new subtree

         Arguments:
             target_split: Split with location to replace subtree
             replacement_subtree: replacement subtree to replace
                                    introduced subtree with
        """
        # If the current split matches, make the swap
        if self.split.splitloc == target_split.splitloc:
            new_tree = replacement_subtree
        # Otherwise, recurse
        else:
            if self.split < target_split:
                # Go down the right branch
                new_tree = SplitTree(split=self.split, L_subtree=self.L_subtree,
                                     R_subtree=self.R_subtree.replace(target_split, replacement_subtree))
            elif self.split > target_split:
                # Go down the left branch
                new_tree = SplitTree(split=self.split,
                                     L_subtree=self.L_subtree.replace(target_split, replacement_subtree),
                                     R_subtree=self.R_subtree)

        return new_tree

    def to_brackets(self, brackets, first_split=True):
        """Converts a SplitTree into a list of brackets to be inserted
        into a word string

        Arguments:
            brackets: list of brackets to be inserted into a word string
            first_split: boolean to indicate if split is first in subtree (?)
                            (default = True)
        """
        if first_split:
            brackets = copy.copy(brackets)

        if self.is_branching:
            splitloc = self.split.splitloc
            # Insert brackets for left component
            brackets[0] += "["
            brackets[splitloc] = "]" + brackets[splitloc]
            # Insert brackets for right component
            brackets[splitloc] += "["
            brackets[-1] = "]" + brackets[-1]
            # Add angled brackets to flag RED
            if isinstance(self.split, RedSplit):
                brackets[self.split.red_start] += "<"
                brackets[self.split.red_end] = ">" + brackets[self.split.red_end]

            # Recurse
            Lbrackets = self.L_subtree.to_brackets(brackets[:splitloc] + [brackets[splitloc].strip("[<")],
                                                   first_split=False)
            Rbrackets = self.R_subtree.reindex(self.split).to_brackets(
                [brackets[splitloc].strip("]>")] + brackets[splitloc + 1:], first_split=False)
            brackets[:splitloc] = Lbrackets[:-1]
            brackets[splitloc] = Lbrackets[-1] + Rbrackets[0]
            brackets[splitloc + 1:] = Rbrackets[1:]

            # Add external brackets to the entire word if there is internal structure
            if first_split:
                brackets[0] = "[" + brackets[0]
                brackets[-1] += "]"

        return brackets

    def get(self, splitloc):
        """Extracts the subtree of a splittree
        that is introduced by a split at a given splitloc.
        If there is no such Split, returns None.

        Arguments:
            splitloc: location of splittree to extract subtree"""
        # First check the split is actually present
        split = self.to_splitlist().get(splitloc)
        if split is None:
            return None
        # Now get the subtree
        current_tree = self
        while current_tree.split != split:
            if split < current_tree.split:
                current_tree = current_tree.L_subtree
            else:
                current_tree = current_tree.R_subtree
        return current_tree

    def update_split_type(self, new_split):
        """Returns a new tree that has a Split replaced with new_split
        of the same splitloc.

        Arguments:
            new_split: new Split to replace with"""
        subtree = self.get(new_split.splitloc)
        old_split = subtree.split
        subtree.split = new_split
        return self.replace(old_split, subtree)

    def enforce_edge_red(self):
        """Return a restructured SplitTree where all reduplicants are at the edge
        of the construction when the corresponding RedSplit is made.
        Thus, the RedSplit for an L-attaching RED is made to have a
        terminal L_subtree, and the RedSplit for an R-attaching RED is
        made to have a terminal R_subtree, while the Split at the red_edge
        is moved to be parent of the RedSplit."""
        splits = SplitStore(self._contained_splits)
        redsplits = [split for split in splits if isinstance(split, RedSplit)]
        edgesplits = [splits.get(redsplit.red_edge) for redsplit in redsplits]
        # Now enforce the edge-alignment
        modified_splittree = self
        for redsplit, edgesplit in zip(redsplits, edgesplits):
            # We can skip edgesplits of None, assuming that RED extends to the construction edge
            if edgesplit is not None:
                attach_side = redsplit.attachment
                other_side = "L" if attach_side == "R" else "R"
                # We are only concerned with the cases where redsplit is an ancestor of edgesplit
                if modified_splittree.is_ancestor(redsplit, edgesplit):
                    # Let R represent the RedSplit subtree and E the EdgeSplit subtree
                    E = modified_splittree.get(edgesplit.splitloc)
                    # Replace E in the tree with its subtree from the attachment side
                    modified_splittree = modified_splittree.replace(edgesplit, E._get_subtree(attach_side))
                    # Get R from the modified tree
                    R = modified_splittree.get(redsplit.splitloc)
                    # Modify E to have the subtree of R on the attachment side
                    E._set_subtree(attach_side, R._get_subtree(attach_side))
                    # Modify R to have an empty subtree on the attachment side
                    R._set_subtree(attach_side, SplitTree(terminal=True))
                    # Modify E to have R as its subtree on the other side
                    E._set_subtree(other_side, R)
                    # Replace R in the tree with E
                    modified_splittree = modified_splittree.replace(redsplit, E)

        return modified_splittree

    def enforce_delayed_red(self):
        """Restructure the SplitTree so that any instances of reduplication
        are as low/late as possible, so the RedSplit introduces no subtrees,
        and the split between RED and the base is terminal.
        This also enforces that all instances of RED be at the edge of their
        constructions."""
        # First make sure that all REDs are at the edge
        modified_splittree = self.enforce_edge_red()
        # Now get the redsplits and their corresponding base splits
        splits = SplitStore(modified_splittree._contained_splits)
        redsplits = [split for split in splits if isinstance(split, RedSplit)]
        basesplits = []
        for redsplit in redsplits:
            basesplit = None  # Leftover Nones represent bases that go to the construction edge
            if redsplit.attachment == "L":
                following_splits = [split for split in splits if split > redsplit]
                if following_splits: basesplit = min(following_splits)
            elif redsplit.attachment == "R":
                previous_splits = [split for split in splits if split < redsplit]
                if previous_splits: basesplit = max(previous_splits)
            basesplits.append(basesplit)
        for redsplit, basesplit in zip(redsplits, basesplits):
            # We can skip cases where there is no basesplit,
            # since then the edge of the construction is the basesplit,
            # and the redsplit is terminal due to enforcing edge-alignment
            if basesplit is not None:
                attach_side = redsplit.attachment
                other_side = "L" if attach_side == "R" else "R"
                # We are only concerned with cases where redsplit is an ancestor of basesplit
                if modified_splittree.is_ancestor(redsplit, basesplit):
                    # Let R represent the RedSplit subtree and B the basesplit subtree
                    R = modified_splittree.get(redsplit.splitloc)
                    B = modified_splittree.get(basesplit.splitloc)
                    # First replace R in the tree with its subtree from the other side
                    modified_splittree = modified_splittree.replace(redsplit, R._get_subtree(other_side))
                    # Then remove the other subtree of R
                    R._set_subtree(other_side, SplitTree(terminal=True))
                    # Set R to be the subtree of B on the attachment side
                    B._set_subtree(attach_side, R)
                    # Finally, update B in the tree
                    modified_splittree = modified_splittree.replace(basesplit, B)

        return modified_splittree

    def get_recapitulated_redsplits(self, construction):
        """Gets the potential RedSplits of a reference Construction
        that are recapitulated in this tree

        Arguments:
            construction: Construction object to get recapitulated RedSplits of
        """
        splits = self.to_splitlist()
        splitlocs = splits.splitlocs
        possible_redsplits = [redspan.to_redsplit() for redspan in construction.redspans]
        # Recapitulated redsplits have the same splitloc as some existing split,
        # have a split at their red_edge, and have no splits at their badsplits
        recapitulated_redsplits = [redsplit for redsplit in possible_redsplits if (
                redsplit not in splits and
                redsplit.splitloc in splitlocs and
                (redsplit.red_edge in splitlocs or
                 redsplit.red_edge == 0 or
                 redsplit.red_edge == len(construction)) and
                not set(redsplit.new_badsplits).intersection(splitlocs))]
        # Make sure forcing remains the same
        for redsplit in recapitulated_redsplits:
            redsplit.forced = splits.get(redsplit.splitloc).forced
        return recapitulated_redsplits

    def get_subtrees(self):
        """Returns a tuple of the two subtrees, with R_subtree reindexed
        according to the current split"""
        if self.is_branching:
            R_subtree = None if self.R_subtree is None else self.R_subtree.reindex(self.split)
            return (self.L_subtree, R_subtree)


class RedSpan(object):
    """Data class for a potential instance of reduplication"""
    __slots__ = ('red_start', 'red_end', 'minbase_edge', 'reduplicant', 'minbase', 'red_label')

    def __init__(self, source, red_start, red_end, minbase_edge, label_by_kind=False):
        """Initialize the object.
        From the endpoints of RED and the minimum permissible base,
        calculate all other required attributes.

        Arguments:
            source: source of potential instance of reduplication
            red_start: location of reduplication beginning
            red_end: location of reduplication endpoint
            minbase_edge: minimum permissible base
            label_by_kind: label for RedSpan (default = False)
        """
        self.red_start = red_start
        self.red_end = red_end
        self.minbase_edge = minbase_edge
        # If generated from RedSpan source, inherit reduplicant, base, and label
        if isinstance(source, RedSpan):
            self.reduplicant = source.reduplicant
            self.minbase = source.minbase
            self.red_label = source.red_label
        # Otherwise, calculate properties from source construction
        else:
            self.reduplicant = self._get_reduplicant(source)
            self.minbase = self._get_minbase(source)
            if label_by_kind:
                self.red_label = self.kind
            else:
                self.red_label = "<RED>"

    def __str__(self):
        """Pretty-printing function"""
        return 'RedSpan(red_start={}, red_end={}, minbase_edge={})'.format(
            *(self.__getattribute__(attr) for attr in ["red_start", "red_end", "minbase_edge"]))

    def __repr__(self):
        """Full-printing function"""
        return 'RedSpan({})'.format(
            ", ".join("{}={}".format(attr, self.__getattribute__(attr)) for attr in self.__slots__))

    def __eq__(self, other):
        """For equality, use red_start, reduplicant and minbase"""
        return (type(self) == type(other) and
                (self.red_start, self.reduplicant, self.minbase) ==
                (other.red_start, other.reduplicant, other.minbase))

    def __hash__(self):
        """For hashing, use red_start, reduplicant and minbase"""
        return hash((self.red_start, self.reduplicant, self.minbase))

    @property
    def attachment(self):
        """Gets the side to which RED attaches: L or R"""
        if self.minbase_edge > self.red_end:
            return ("L")
        elif self.minbase_edge < self.red_start:
            return ("R")

    @property
    def red_edge(self):
        """Gets the far edge of the redspan from the base"""
        if self.attachment == "L":
            return self.red_start
        elif self.attachment == "R":
            return self.red_end

    @property
    def left_edge(self):
        """Gets the left edge of the redspan"""
        return min(self.red_start, self.minbase_edge)

    @property
    def right_edge(self):
        """Gets the right edge of the redspan"""
        return max(self.red_end, self.minbase_edge)

    @property
    def splitloc(self):
        """Gets the location of the split associated with this redspan.
        For L-attaching RED, splitloc is at red_end;
        for R-attaching RED, splitloc is at red_start."""
        if self.attachment == "L":
            return self.red_end
        elif self.attachment == "R":
            return self.red_start

    @property
    def base_portion(self):
        """Gets the portion of the base that the reduplicant is assumed
        to have copied."""
        if self.attachment == "L":
            return self.minbase[:len(self.reduplicant)]
        elif self.attachment == "R":
            return self.minbase[-len(self.reduplicant):]

    def _get_reduplicant(self, construction):
        """Gets the part of the construction representing the reduplicant"""
        return construction[self.red_start:self.red_end]

    def _get_minbase(self, construction):
        """Gets the part of the construction that is the minimal base"""
        if self.attachment == "L":
            return construction[self.red_end:self.minbase_edge]
        elif self.attachment == "R":
            return construction[self.minbase_edge:self.red_start]

    def reindex(self, newsplit):
        """Returns a new RedSpan, reindexed by subtracting a given newsplit from the RED and base indices.
        Validity checking of reindexing is assumed to occur elsewhere (so this can give negative indices)."""
        newsplitloc = newsplit.splitloc
        return RedSpan(self, self.red_start - newsplitloc, self.red_end - newsplitloc, self.minbase_edge - newsplitloc)

    def is_left_of(self, newsplit):
        """Checks whether the RedSpan is entirely left of a provided newsplit."""
        newsplitloc = newsplit.splitloc
        return self.right_edge <= newsplitloc

    def is_right_of(self, newsplit):
        """Checks whether the RedSpan is entirely right of a provided newsplit."""
        newsplitloc = newsplit.splitloc
        return self.left_edge >= newsplitloc

    def is_at_edge_of(self, construction):
        """Checks whether RED in the redspan is at the edge of the construction
        that is consistent with its attachment side (L-attaching RED at the left
        edge, R-attaching RED at the right edge)"""
        red_edge = self.red_edge
        return red_edge == 0 or red_edge == len(construction)

    def has_base_at_edge_of(self, construction):
        """Checks whether the minbase of the redspan is at the edge
        of the construction"""
        base_edge = self.minbase_edge
        return base_edge == 0 or base_edge == len(construction)

    @property
    def badsplits(self):
        """Gets the badsplits associated with the current redspan.
        These are all the integer splits falling between left_edge and right_edge,
        exclusive, and excluding the current splitloc"""
        return (tuple(range(self.left_edge + 1, self.splitloc)) +
                tuple(range(self.splitloc + 1, self.right_edge)))

    def to_redsplit(self, base_newform=None, forced=False):
        """Converts a RedSpan object into a RedSplit object.
        If label_by_kind is True, gives the label of RED as the
        kind of reduplication; otherwise, gives it as <RED>.
        Changes to the form of the base (e.g. initial lengthening)
        are represented in base_newform.

        Arguments:
            base_newform: newform representing changes to the form of the base (?)
            forced: boolean indicating if split is forced (?) (default = False)
        """
        # Get the new forms of children
        if self.attachment == "L":
            L_newform = Reduplicant(self.kind, self.red_label)
            R_newform = base_newform
        elif self.attachment == "R":
            L_newform = base_newform
            R_newform = Reduplicant(self.kind, self.red_label)
        return RedSplit(self.splitloc, self.red_edge, self.minbase_edge, self.badsplits,
                        L_newform=L_newform, R_newform=R_newform, forced=forced)

    @property
    def kind(self):
        """Gets a string representation of the kind of RED.
        e.g. <RED-2-R> is right-reduplication of a bimoraic base portion,
        <RED-1+-L> is left-reduplication of a monomoraic base portion, with lengthening"""
        weight = self.base_portion.weight
        lengthened = (self.reduplicant != self.base_portion and
                      self.base_portion.lengthen_initial() == self.reduplicant)
        return "<RED-{}{}-{}>".format(weight, "+" if lengthened else "", self.attachment)


class AssociativeStore(dict):
    """Class for storing objects (redspans or badsplits) in association
    with a motivating construction.
    The basic structure is a dict mapping from object to motivators,
    where motivators are represented as strings.
    All of the public methods of this class return new instances,
    rather than modifying the existing instance in-situ."""
    __slots__ = ()

    def _motivators(self, obj):
        """Get a set of motivators for the given object"""
        return self.get(obj, set())

    def add(self, new_objs, motivator):
        """Creates a new AssociativeStore in which a provided tuple of new_objs
        has been added, with its corresponding motivator.
        If motivator is a Construction, it is converted to str

        Arguments:
            new_objs: tuple of objects
            motivator: associated motivating construction (?) (are all motivators constructions?)
        """
        other_store = self.__class__((obj, {motivator.label}) for obj in new_objs)
        return self + other_store

    def remove(self, motivator):
        """Creates a new AssociativeStore in which the objects associated with
        the provided motivator have been removed"""
        other_store = self.__class__((obj, {motivator.label}) for obj in self)
        return self - other_store

    def __add__(self, other_store):
        """Creates a new AssociativeStore that has objs from both the current
        store and a provided other store"""
        modified_store = self.__class__()
        for obj in set(self).union(other_store):
            modified_store[obj] = self._motivators(obj).union(other_store._motivators(obj))
        return modified_store

    def __sub__(self, other_store):
        """Creates a new AssociativeStore that has all of the original objs, except
        those contained in the other_store object"""
        modified_store = self.__class__()
        for obj in self:
            modified_store[obj] = self[obj] - other_store._motivators(obj)
        modified_store._trim_empty()
        return modified_store

    def _trim_empty(self):
        """Removes entries for objs that no longer have any motivators"""
        unmotivated = [obj for obj, motivators in self.items() if not motivators]
        for obj in unmotivated:
            del self[obj]


class RedSpanStore(AssociativeStore):
    """Class for storing all RedSpans associated with a construction,
    and methods for reindexing and partitioning these RedSpans.
    The basic structure is a dict, mapping from RedSpan objects to
    a set of strings representing surface ancestral constructions where
    the redspans were originally found."""
    __slots__ = ()

    def reindex(self, split):
        """Returns a new RedSpanStore in which all of the redspans
        have been reindexed by subtracting the splitloc of the given Split."""
        modified_redspans = RedSpanStore(
            (redspan.reindex(split), ancestors) for redspan, ancestors in self.items())
        return modified_redspans

    def slice(self, start_split, end_split=None):
        """Creates a new RedSpanStore that is a slice of the current one,
        including all redspans that are right of start_split and left of
        end_split, reindexed according to start_split

        Arguments:
            start_split: Split to begin slice
            end_split: Split to end slice (default = None)
        """
        # If there are no redspans, return an empty RedSpanStore
        if not self:
            return RedSpanStore()
        # Otherwise, slice as required
        sliced_redspans = ((redspan, ancestors) for redspan, ancestors in self.items() if
                           redspan.is_right_of(start_split))
        if end_split is not None:
            sliced_redspans = ((redspan, ancestors) for redspan, ancestors in sliced_redspans if
                               redspan.is_left_of(end_split))
        modified_redspans = RedSpanStore(sliced_redspans).reindex(start_split)
        return modified_redspans

    def partition(self, split):
        """Partition the redspans according to a provided split.
        Returns a tuple of RedSpanStore objects (L_redspans, R_redspans),
        where a given RedSpan will be in L_redspans if it falls entirely
        left of the splitloc and in R_redspans if it falls entirely right.
        The redspans in R_redspans will be reindexed by split"""
        L_redspans = self.slice(Split(0), end_split=split)
        R_redspans = self.slice(split)
        return (L_redspans, R_redspans)


class BadSplitStore(AssociativeStore):
    """Class for storing the badsplits and their motivations for a particular
    ConstrNode, as well as methods for updating these structures.
    The basic structure is a dict mapping from a badsplit index to the set
    of motivators for that badsplit."""
    __slots__ = ()

    def reindex(self, split):
        """Creates a new BadSplits object that is a copy of the current one,
        except with splits reindexed by subtracting the splitloc of a Split"""
        modified_badsplits = BadSplitStore(
            (badsplitloc - split.splitloc, motivators) for badsplitloc, motivators in self.items())
        return modified_badsplits

    def slice(self, start_split, end_split=None):
        """Creates a new BadSplits object that is a slice of the current
        badsplits occurring between start_split and end_split,
        reindexed according to start_split

        Arguments:
            start_split: Split to begin slice
            end_split: Split to end slice (default = None)
            """
        # If there are no badsplits, return an empty BadSplitStore
        if not self:
            return BadSplitStore()
        # Otherwise, slice as required
        sliced_badsplits = ((badsplitloc, motivators) for badsplitloc, motivators in self.items() if
                            badsplitloc > start_split.splitloc)
        if end_split is not None:
            sliced_badsplits = ((badsplitloc, motivators) for badsplitloc, motivators in sliced_badsplits if
                                badsplitloc < end_split.splitloc)
        modified_badsplits = BadSplitStore(sliced_badsplits).reindex(start_split)
        return modified_badsplits

    def partition(self, split, construction, add_new=True):
        """Partitions the badsplits based on splitloc from a Split object.
        If add_new is True, new badpslits introduced by the Split are added.
        Returns a tuple of BadSplits objects, (L_badsplits, R_badsplits)
        L inherits badsplits s.t. badsplit < splitloc
        R inherits badsplits s.t. badsplit > splitloc, reindexed by splitloc

        Arguments:
            split: Split object with splitloc to base partition on
            construction: construction object (?)
            add_new: boolean to indicate if new badsplits introduced
                        by the Split are added (default = True)

        """
        # Set up master badsplits for inheritance
        master_badsplits = self
        # Add new badsplits to master if they exist and add_new is True
        if add_new and hasattr(split, "new_badsplits") and split.new_badsplits:
            master_badsplits = master_badsplits.add(split.new_badsplits, construction)
        L_badsplits = master_badsplits.slice(Split(0), end_split=split)
        R_badsplits = master_badsplits.slice(split)

        return (L_badsplits, R_badsplits)


class ForcedSplitlocStore(AssociativeStore):
    """Class for storing all forced splitlocs associated with a construction,
    and methods for reindexing and partitioning these splitlocs.
    The basic structure is a dict, mapping from splitlocs (ints) to
    a set of strings representing surface ancestral constructions where
    the splitlocs were originally forced."""
    __slots__ = ()

    def reindex(self, split):
        """Returns a new ForcedSplitlocStore in which all of the splitlocs
        have been reindexed by subtracting the splitloc of the given Split."""
        modified_splitlocs = ForcedSplitlocStore(
            (splitloc - split.splitloc, ancestors) for splitloc, ancestors in self.items())
        return modified_splitlocs

    def slice(self, start_split, end_split=None):
        """Creates a new ForcedSplitlocStore that is a slice of the current one,
        including all splitlocs that are right of start_split and left of
        end_split, reindexed according to start_split

        Arguments:
            start_split: Split to begin slice
            end_split: Split to end slice (default = None)
        """
        # If there are no forced splitlocs, return an empty ForcedSplitlocStore
        if not self:
            return ForcedSplitlocStore()
        # Otherwise, slice as required
        sliced_splitlocs = ((splitloc, ancestors) for splitloc, ancestors in self.items() if
                            splitloc > start_split.splitloc)
        if end_split is not None:
            sliced_splitlocs = ((splitloc, ancestors) for splitloc, ancestors in sliced_splitlocs if
                                splitloc < end_split.splitloc)
        modified_splitlocs = ForcedSplitlocStore(sliced_splitlocs).reindex(start_split)
        return modified_splitlocs

    def partition(self, split):
        """Partition the splitlocs according to a provided split.
        Returns a tuple of ForcedSplitlocStore objects (L_splitlocs, R_splitlocs),
        where a given splitloc will be in L_splitlocs if it falls left of
        the split and in R_splitlocs if it falls right of it.
        The splitlocs in R_splitlocs will be reindexed by split"""
        L_splitlocs = self.slice(Split(0), end_split=split)
        R_splitlocs = self.slice(split)
        return (L_splitlocs, R_splitlocs)


class Construction(object):
    """A container for a construction object.
    Internally, a construction looks like a tuple of atoms,
    with additional properties for r_count, count, splitlist, badsplits, and redspans.
    It prints like a string, joining all the atoms."""
    __slots__ = ('_atoms', 'r_count', 'count', 'splitlist', 'badsplits', 'redspans', 'forced_splitlocs', 'label')

    def __init__(self, atoms, r_count=0, count=0, splitlist=None, badsplits=None, redspans=None, forced_splitlocs=None, label=None):
        """Initialize the object, storing the atoms

        Arguments:
            atoms: atoms in the construction object
            r_count: (?) (default = 0)
            count: (?) (default = 0)
            splitlist: list of splits for the construction (default = None)
            badsplits: list of bad splits for the construction (default = None)
            redspans: list of potential instances of reduplication (default = None)
            forced_splitlocs: all forced splitlocs associated with a construction (default = None)
            label: string form of joined atoms (default = None)
        """
        if label is None:
            label = "".join(atoms)
        # Correct for NoneTypes (avoid pooled referencing)
        if splitlist is None:
            splitlist = SplitStore()
        if badsplits is None:
            badsplits = BadSplitStore()
        if redspans is None:
            redspans = RedSpanStore()
        if forced_splitlocs is None:
            forced_splitlocs = ForcedSplitlocStore()
        # Store properties
        self._atoms = atoms
        self.r_count = r_count
        self.count = count
        self.splitlist = splitlist
        self.badsplits = badsplits
        self.redspans = redspans
        self.forced_splitlocs = forced_splitlocs
        self.label = label

    def __eq__(self, other):
        """For equality, use the atoms"""
        return type(self) == type(other) and self._atoms == other._atoms

    def __hash__(self):
        """For hashing, use the atoms"""
        return hash(self._atoms)

    def __repr__(self):
        """For representations, use the attributes"""
        cls_name = type(self).__name__
        return "{}({}, {})".format(cls_name, self._atoms, ", ".join(
            "{}={}".format(attr, self.__getattribute__(attr)) for cls in type(self).mro() for attr in
            getattr(cls, '__slots__', ()) if attr != "_atoms"))

    def __str__(self):
        """For strings, join the atoms with no delimiter"""
        return self.label

    def __getitem__(self, idx):
        """When getting (or slicing), use the indices from the atoms.
        If the result has a single atom, return the atom string.
        If the result has no atoms, return an empty Construction object.
        Otherwise, return a new Construction object that has a slice of atoms,
        but which does not inherit any attributes."""
        result = self._atoms[idx]
        if isinstance(result, str):
            return result
        else:
            return Construction(result)

    def __len__(self):
        """Get the number of atoms"""
        return len(self._atoms)

    def __iter__(self):
        """For iteration, use the atoms"""
        return iter(self._atoms)

    def __lt__(self, other):
        """For sorting, use the atoms"""
        return isinstance(other, Construction) and self._atoms < other._atoms

    def __copy__(self):
        """A method for copying, to improve speed"""
        return Construction(self._atoms, self.r_count, self.count, self.splitlist, self.badsplits, self.redspans, self.forced_splitlocs, self.label)

    def populate_redspans(self, redFinder):
        """Uses a provided ReduplicationFinder object to find the
        RedSpans for the construction and store them in the RedSpanStore."""
        new_redspans = redFinder.find_redspans(self)
        self.redspans += new_redspans

    def populate_badsplits(self, badsplit_re, atom_separator=" "):
        """Uses a provided regular expression to find the badsplits for
        the construction and store in the BadSplitStore.

        Arguments:
            badsplit_re: regex expression to find badsplits
            atom_separator: string used to separate atoms in construction
                            (default = " ")
        """
        for i in range(1, len(self)):
            split_environ = atom_separator.join(self._atoms[i - 1:i + 1])
            if badsplit_re.search(split_environ):
                self.badsplits = self.badsplits.add([i], self)

    def populate_forced_splitlocs(self, forcesplit_re, atom_separator=" "):
        """Uses a provided regular expression to find the forced splitlocs
        for the construction and store in the ForcedSplitlocStore.

        Arguments:
            badsplit_re: regex expression to find forced splitlocs
            atom_separator: string used to separate atoms in construction
                            (default = " ")
        """
        for i in range(1, len(self)):
            split_environ = atom_separator.join(self._atoms[i - 1:i + 1])
            if forcesplit_re.search(split_environ):
                self.forced_splitlocs = self.forced_splitlocs.add([i], self)

    @property
    def forced_splits(self):
        """Returns a list of the valid forced splits for the current construction,
        which are Splits with splitlocs in forced_splitlocs, as well as any
        valid RedSplits that have the same splitloc"""
        forced_splitlocs = set(self.forced_splitlocs)
        splits = [Split(splitloc, forced=True) for splitloc in forced_splitlocs]
        splits += [redspan.to_redsplit(forced=True) for redspan in self.redspans if
                   redspan.splitloc in forced_splitlocs and
                   not forced_splitlocs.intersection(redspan.badsplits)]
        return splits

    def split(self, split, add_new_badsplits=True, inherit=True):
        """Split a construction using a Split object.
        Returns a list of new Construction objects, [L_child, R_child],
        which inherit the count, badsplits and redspans of the construction
        if inherit is True.
        New badsplits are added when triggered if add_new_badsplits is True.
        There is no inheritance of splitlists.

        Arguments:
            split: Split object
            add_new_badsplits: boolean indicator to add new badsplits (default = True)
            inherit: boolean indicator to inherit count, badsplits
                    and redspans of the construction (default = True)
        """
        # Must be a Split instance in order to split
        if not isinstance(split, Split):
            raise Exception("Attempting to split %s with non-Split object %s" % (self, split))
        # Begin by assuming the standard split
        L_child = self[:split.splitloc]
        R_child = self[split.splitloc:]
        # Correct for newforms
        if isinstance(split, ChangeSplit):
            if split.L_newform:
                L_child = split.L_newform
            if split.R_newform:
                R_child = split.R_newform
        if inherit:
            # Inherit counts
            for child in [L_child, R_child]:
                child.count = self.count
            # Inherit badsplits, redspans, and forced splitlocs, provided the child is not RED
            L_badsplits, R_badsplits = self.badsplits.partition(split, self, add_new=add_new_badsplits)
            L_redspans, R_redspans = self.redspans.partition(split)
            L_forcedsplitlocs, R_forcedsplitlocs = self.forced_splitlocs.partition(split)
            if not isinstance(L_child, Reduplicant):
                L_child.badsplits = L_badsplits
                L_child.redspans = L_redspans
                L_child.forced_splitlocs = L_forcedsplitlocs
            if not isinstance(R_child, Reduplicant):
                R_child.badsplits = R_badsplits
                R_child.redspans = R_redspans
                R_child.forced_splitlocs = R_forcedsplitlocs

        return [L_child, R_child]

    def multi_split(self, splitlist, inherit=False):
        """Splits a Construction multiple times, in the order designated
        by the default SplitTree corresponding to the splitlist (so
        as not to mess up instances of reduplication)

        Arguments:
            splitlist: list of Split objects
            inherit: boolean indicator to inherit count, badsplits
                    and redspans of the construction (?) (default = False)
        """
        # If it is just a binary split, use normal split instead
        if len(splitlist) == 1:
            return self.split(splitlist[0], inherit=inherit)
        else:
            splittree = splitlist.to_splittree()
            return self._split_by_tree(splittree, inherit=inherit)

    def _split_by_tree(self, splittree, inherit=False):
        """Splits the Construction according to the provided SplitTree.
        Returns a list of terminal child Constructions.

        Arguments:
            splittree: SplitTree object
            inherit: boolean indicator to inherit count, badsplits
                    and redspans of the construction (?) (default = False)
        """
        if not splittree.is_branching:
            return [self]
        else:
            L_child, R_child = self.split(splittree.split, inherit=inherit)
            L_splittree = splittree.L_subtree
            R_splittree = splittree.R_subtree.reindex(splittree.split)
            return L_child._split_by_tree(L_splittree, inherit=inherit) + R_child._split_by_tree(R_splittree, inherit=inherit)

    def slice(self, start_split, end_split=None):
        """Returns a slice of the construction between start and end
        Split objects, with attributes copied accross.
        If end_split is None, carries on to the end of the Construction

        Arguments:
            start_split: Split with location to begin slice
            end_split: Split with location to end slice (default = None)
        """
        # Begin by assuming the slice with no attributes copied
        if end_split is None:
            construction = self[start_split.splitloc:]
        else:
            construction = self[start_split.splitloc:end_split.splitloc]
        # Inherit counts
        construction.count = self.count
        # Inherit badsplits, redspans, and forced splitlocs
        construction.badsplits = self.badsplits.slice(start_split, end_split)
        construction.redspans = self.redspans.slice(start_split, end_split)
        construction.forced_splitlocs = self.forced_splitlocs.slice(start_split, end_split)

        return construction

    @property
    def has_children(self):
        """Checks if the construction has children"""
        return bool(self.splitlist)

    def get_children(self, add_new_badsplits=True):
        """Gets the children of the Construction, by splitting
        at all of the splits in the splitlist, in the order
        designated by the default splittree (so as not to
        mess up reduplicants).
        Returns a list of Construction objects, in linear order.
        If add_new_badsplits is True, the badsplits of children
        are updated as new badsplits are introduced by Splits."""
        if self.has_children:
            if len(self.splitlist) == 1:
                return self.split(self.splitlist[0], add_new_badsplits=add_new_badsplits)
            else:
                splittree = self.splitlist.to_splittree()
                L_child, R_child = self.split(splittree.split, add_new_badsplits=add_new_badsplits)
                L_splitlist, R_splitlist = self.splitlist.partition(splittree.split)
                L_child.splitlist = L_splitlist
                R_child.splitlist = R_splitlist
                return L_child.get_children(add_new_badsplits=add_new_badsplits) + R_child.get_children(
                    add_new_badsplits=add_new_badsplits)
        else:
            return [self]

    def _get_valid_split_subset(self, possible_splits):
        """Given an iter of possible splits, returns a generator
         over the subset of those splits that are valid, i.e. that
         do not conflict with the construction's badsplits, and that
         do not create badsplits at the construction's forced splits"""
        forced_splitlocs = set(self.forced_splitlocs)
        for split in possible_splits:
            if split.splitloc not in self.badsplits:
                if not (hasattr(split, "new_badsplits") and forced_splitlocs.intersection(
                        split.new_badsplits)):
                    yield split

    def _get_valid_redspan_subset(self, possible_redspans):
        """Given an iter of possible RedSpans, returns a generator
         over the subset of those redspans that are valid, i.e. that
         do not conflict with the construction's badsplits, and that
         do not create badsplits at the construction's forced splits"""
        forced_splitlocs = set(self.forced_splitlocs)
        for redspan in possible_redspans:
            if not (redspan.splitloc in self.badsplits or redspan.red_edge in self.badsplits):
                if not forced_splitlocs.intersection(redspan.badsplits):
                    yield redspan

    @property
    def full_redspan(self):
        """Returns a RedSpan that separates the construction into
        an edge-aligned reduplicant and base at its halfway point,
        or None if no such RedSpan exists."""
        if len(self) % 2 == 0:
            valid_redspans = self._get_valid_redspan_subset(self.redspans)
            for redspan in valid_redspans:
                if redspan.is_at_edge_of(self) and redspan.splitloc == len(self) / 2:
                    return redspan
        return None

    @property
    def full_redsplit(self):
        """Returns a Forced RedSplit that separates the construction into
        an edge-aligned reduplicant and base at its halfway point,
        or None if no such split exists."""
        redspan = self.full_redspan
        if redspan is None:
            return None
        else:
            return redspan.to_redsplit(forced=True)

    @property
    def valid_splits(self):
        """Returns a generator over the valid Splits for the Construction
        (i.e. the possible splits, minus the badsplits). Does not include
        a 'zero split' or redsplits."""
        possible_splits = (Split(splitloc) for splitloc in range(1, len(self)))
        yield from self._get_valid_split_subset(possible_splits)

    def get_valid_redsplits(self, base_at_edge=False):
        """Returns a pair of lists of valid RedSplits for the Construction
        (i.e. the RedSplits that do not conflict with existing badsplits
        or forced splits).
        The first list contains the RedSplits that are at the Construction edge.
        If base_at_edge is True, the first list also contains any RedSplits
        with the same splitloc that have their base at the Construction edge.
        The second list contains the remaining RedSplits."""
        # First pass: get the valid redspans
        valid_redspans = list(self._get_valid_redspan_subset(self.redspans))
        # Second pass: sort into edge-aligned redsplits, and remainder
        edge_redsplits = []
        remainder = []
        for redspan in valid_redspans:
            if redspan.is_at_edge_of(self):
                edge_redsplits.append(redspan.to_redsplit())
            else:
                if base_at_edge:
                    remainder.append(redspan)
                else:
                    remainder.append(redspan.to_redsplit())
        # Third pass, if base_at_edge is True: sort remainder into edge-equivalent, and non-edge, and convert to RedSplits
        if base_at_edge:
            nonedge_redsplits = []
            edge_splitlocs = {redsplit.splitloc for redsplit in edge_redsplits}
            for redspan in remainder:
                if redspan.has_base_at_edge_of(self) and redspan.splitloc in edge_splitlocs:
                    edge_redsplits.append(redspan.to_redsplit())
                else:
                    nonedge_redsplits.append(redspan.to_redsplit())
        else:
            nonedge_redsplits = remainder
        return (edge_redsplits, nonedge_redsplits)

    def lengthen_initial(self):
        """Returns a version of the Construction in which the first
        syllable has been lengthened"""
        # Lengthening is just capitalization of the vowel (last character)
        first_syll = self._atoms[0]
        vowel = first_syll[-1]
        vowel_long = vowel.upper()
        if vowel == vowel_long:
            # Already long; return original construction
            return self
        else:
            return Construction((first_syll[:-1] + vowel_long,) + self._atoms[1:])

    @property
    def weight(self):
        """Calculate the weight of the construction.
        Here, weight is measured in morae: an atom ending in
        a long vowel (AEIOU) gets 2 morae, and any other
        atom gets 1 mora."""
        weight = 0
        for atom in self:
            if atom[-1] == atom[-1].upper():
                weight += 2
            else:
                weight += 1
        return weight

    @property
    def penalties(self):
        """Calculate a list of penalties incurred by this construction"""
        penalties = []
        # Penalty for being monomoraic
        if self.weight == 1:
            penalties.append("monomoraic")
        return penalties

    def segmentation_to_splitlist(self, parts):
        """Given a list of parts in atom form, returns the splitlist
        that can turn the current Construction into those parts"""
        red_starts_to_redspans = {}
        for redspan in self.redspans:
            if redspan.red_start in red_starts_to_redspans:
                red_starts_to_redspans[redspan.red_start].append(redspan)
            else:
                red_starts_to_redspans[redspan.red_start] = [redspan]

        splitlist = []
        i = 0  # represents the start index of the current part
        for j, part in enumerate(parts):
            if len(part) == 1 and part[0].startswith("<RED"):
                # Reduplication. Find the corresponding redspan
                red_start = i
                candidate_redspans = red_starts_to_redspans[red_start]
                # If there's just one candidate, take it
                if len(candidate_redspans) == 1:
                    redspan = candidate_redspans[0]
                # Otherwise, figure out which candidate is right
                else:
                    for redspan in candidate_redspans:
                        # If we're on the final part, RED must be R-attaching
                        if j == len(parts) - 1:
                            if redspan.attachment == "R": break
                        # If we're on the initial part, RED must be L-attaching
                        elif j == 0:
                            if redspan.attachment == "L": break
                        # If RED is L-attaching, the first atom of RED must be the first atom of the next part
                        elif redspan.attachment == "L":
                            if self[red_start] == parts[j + 1][0]: break
                        # And if RED is R-attaching, the last atom of RED must be the last atom of the previous part
                        elif redspan.attachment == "R":
                            if self[redspan.red_end] == parts[j - 1][-1]: break
                # If we have L-attaching RED, we can just append the redspan
                # (since the prefix split is already handled)
                if redspan.attachment == "L":
                    splitlist.append(redspan.to_redsplit())
                # If we have R-attaching RED, we have to replace the previous split
                # with the redspan, then append the new index as a suffix split
                elif redspan.attachment == "R":
                    splitlist = splitlist[:-1] + [redspan.to_redsplit(), Split(redspan.red_end)]
                # Update the start index and continue to the next part
                i = redspan.red_end
            else:
                i += len(part)
                splitlist.append(Split(i))

        # Exclude the final "split" corresponding to the end of the construction
        return SplitStore(splitlist[:-1])

    @property
    def initial_segment(self):
        """Returns a string representation of the initial segment in the Construction"""
        return self._atoms[0][0]

    @property
    def red_splitlocs_to_redspans(self):
        """Gets a dict mapping from the splitloc of RedSpans in the Construction
        to a list of the corresponding RedSpans"""
        splitlocs_to_redspans = {}
        for redspan in self.redspans:
            splitloc = redspan.splitloc
            if splitloc in splitlocs_to_redspans:
                splitlocs_to_redspans[splitloc].append(redspan)
            else:
                splitlocs_to_redspans[splitloc] = [redspan]
        return splitlocs_to_redspans

    @property
    def potential_fullred_right_to_left_edges(self):
        """Gets a dict mapping from the splitloc of the right edge to the left edge
        of RedSpans in the Construction that potentially correspond to full reduplication"""
        valid_redspans = self._get_valid_redspan_subset(self.redspans)
        full_redspans = (redspan for redspan in valid_redspans if redspan.reduplicant == redspan.minbase)
        return {redspan.right_edge: redspan.left_edge for redspan in full_redspans}


class Reduplicant(Construction):
    """A container for a reduplicant that has been identified.
    Simply stores the label and kind.
    This object behaves as though it is a construction consisting
    just of the label, but the kind is also accessible"""
    __slots__ = ('kind',)

    def __init__(self, kind, label=None, count=0):
        """Initializes the object.
        If no label is provided, kind is used as a the label

        Arguments:
            kind: string representation of type of rediplicant
            label: reduplicant label (default = None)
            count: (?) (default = 0)
            """
        if isinstance(kind, tuple):
            kind = kind[0]
        if label is None:
            label = kind
        # Initialize the construction
        super().__init__((label,), count=count, label=label)
        # Add the reduplication-specific attributes
        self.kind = kind

    def __copy__(self):
        """A method for copying, to improve speed"""
        return Reduplicant(self.kind, self.label, self.count)

    @staticmethod
    def lengthen_initial():
        """RED instances have None for initial lengthened versions"""
        return None

    @property
    def weight(self):
        """RED instances have None for weight"""
        return None

    @property
    def penalties(self):
        """Add reduplication-specific penalties"""
        penalties = super().penalties
        # Penalty for reduplicating a monomoraic part of the base
        if "1" in self.kind:
            penalties.append("monored")
        return penalties


class ConstructionStore(object):
    """A container for storing Construction object.
    It is like a dict, mapping from strings to Constructions.
    Also contains methods for updating Construction counts and
    splitting Constructions with their attributes."""
    __slots__ = ('_constructions',)

    def __init__(self):
        """Initialize the object"""
        self._constructions = {}

    def __repr__(self):
        """Return a representation of the dict"""
        return repr(self._constructions)

    def __str__(self):
        """Pretty-printing"""
        return "\n".join("{} ({}): {}".format(construction.label, construction.count,
                                              " + ".join(child.label for child in construction.get_children())) for
                         construction in self)

    def __hash__(self):
        """This object is mutable, and hence unhashable"""
        return None

    def __len__(self):
        """Return the number of stored Construction objects"""
        return len(self._constructions)

    def __iter__(self):
        """Iterate over the Construction objects"""
        return iter(self._constructions.values())

    def __contains__(self, construction):
        """Check whether a given construction is in the store"""
        return construction.label in self._constructions

    def __getitem__(self, construction):
        """Gets a Construction object"""
        return self._constructions[construction.label]

    def _add_ref(self, construction):
        """Adds a reference for a Construction to the store.
        This is a blank Construction; attributes are not copied."""
        if isinstance(construction, Reduplicant):
            blank_entry = Reduplicant(construction.label)
        else:
            blank_entry = Construction(construction._atoms)
        self._constructions[construction.label] = blank_entry

    def get_ref(self, construction):
        """Gets a stored Construction with the same string as the probe.
        If there is no stored Construction, makes an empty one and returns it."""
        if construction not in self:
            self._add_ref(construction)
        return self[construction]

    def _del_ref(self, construction):
        """Removes a reference to a Construction from the store"""
        if construction in self:
            del self._constructions[construction.label]

    def _get_contributions(self, construction, operation):
        """Returns a generator over the contribution of the provided construction
        to all other constructions in the store, meaning both its descendants
        and the construction itself.
        Descendants are based on splits from the store.
        Contributions are Constructions with only those attributes (counts,
        badsplits, redspans) that can be traced to the original construction.
        Parameter operation takes string values "add" or "remove", saying
        whether the obtained contributions are going to be added to or removed
        from the store. Operation affects the creation of new badsplits:
        new badsplits are always created when adding (so that the child
        badsplits can be traced to the parent), but are only created when
        removing if the entirety of the parent's stored count is attributed
        to the original construction (so that new badsplits are not removed
        when only a partial contribution of the parent is removed).
        Still yields the construction if it is not in the store.

        Arguments:
            construction: Construction object to retrieve contribution
            operation: operation for creation of new badsplits (ex: "add" or "remove")
        """
        stored_construction = self.get_ref(construction)
        modified_construction = copy.copy(construction)
        modified_construction.splitlist = stored_construction.splitlist
        yield modified_construction
        if modified_construction.has_children:
            # Adding new badsplits is based on operation
            add_new_badsplits = ((operation == "add") or
                                 (operation == "remove" and modified_construction.count == stored_construction.count))
            for child in modified_construction.get_children(add_new_badsplits=add_new_badsplits):
                yield from self._get_contributions(child, operation)

    def _get_descendants(self, construction, operation):
        """Gets a list of the contributions of a construction to all of its
        descendants, based on splits provided in the store.
        See method get_contributions (this just removes the provided construction
        from the generator returned by get_contributions).
        If the construction is not in the store, _get_contributions will not be
        called, and thus a new entry for the construction will not be added.

        Arguments:
            construction: Construction object to get descendants contribution
            operation: operation for creation of new badsplits (ex: "add" or "remove")
        """
        if construction in self:
            contributions = self._get_contributions(construction, operation)
            # The construction itself will always be the first element in the generator
            own_contribution = next(contributions)
            return list(contributions)

    def get_history(self, construction):
        """Returns a list of the contributions that the construction makes
        to the store that are solely reliant upon the construction for
        existence. This forms a history of the state of the store."""
        return [contribution
                for contribution in self._get_contributions(construction, "remove")
                if contribution.count == self[contribution].count]

    def restore_history(self, contributions):
        """Restores the contributions that a construction makes to the store,
        which means resetting the store according to the history that the
        contributions represent."""
        for construction in contributions:
            self.change_splitlist(construction, construction.splitlist, branching=False)

    def _remove_isolated(self, construction, change_rcount=False):
        """Removes the contribution of a single construction to the store,
        without removing its contributions to descendants

        Arguments:
            construction: Construction object to remove isolated contribution
            change_rcount: (?) (default = False)
        """
        stored_construction = self.get_ref(construction)
        if change_rcount:
            stored_construction.r_count -= construction.r_count
        stored_construction.count -= construction.count
        if stored_construction.count == 0:
            self._del_ref(construction)
        # Don't bother updating badsplits or redspans for reduplicants
        elif not isinstance(construction, Reduplicant):
            stored_construction.badsplits -= construction.badsplits
            stored_construction.redspans -= construction.redspans
            stored_construction.forced_splitlocs -= construction.forced_splitlocs

    def _remove_descendants(self, construction):
        """Removes a Construction's contributions to the counts, badsplits,
        and redspans of its descendants, as they exist in the store."""
        for descendant_contribution in self._get_descendants(construction, "remove"):
            self._remove_isolated(descendant_contribution)

    def remove(self, construction):
        """Removes the contribution of a single construction to the store,
        together with its contribution to all of its descendants.
        The provided Construction object is assumed to contain all of
        attributes to be removed; it is not cross-referenced with
        constructions in the store (because that would overwrite the
        set of attributes to be removed)"""
        self._remove_descendants(construction)
        self._remove_isolated(construction, change_rcount=True)

    def _add_isolated(self, construction, overwrite_splitlist=False, change_rcount=False):
        """Adds a new contribution for a single construction to the store,
        without adding contributions to descendants.
        Overwrites the splitlist in the store if desired.

        Arguments:
            construction: Construction object to add an isolated new contribution to
            overwrite_splitlist: boolean to allow for splitlist in store
                                    to be overwritten (default = False)
            change_rcount: (?) (default = False)
        """
        stored_construction = self.get_ref(construction)
        if overwrite_splitlist:
            self.change_splitlist(stored_construction, construction.splitlist)
        if change_rcount:
            stored_construction.r_count += construction.r_count
        stored_construction.count += construction.count
        stored_construction.badsplits += construction.badsplits
        stored_construction.redspans += construction.redspans
        stored_construction.forced_splitlocs += construction.forced_splitlocs

    def _add_descendants(self, construction):
        """Adds a Construction's contributions to the counts, badsplits,
        and redspans of its descendants, as they exist in the store.
        If descendants are not in the store, creates them."""
        for descendant_contribution in self._get_descendants(construction, "add"):
            self._add_isolated(descendant_contribution)

    def add(self, construction, overwrite_splitlist=False):
        """Adds a new contribution for a construction to the store,
        and adds contributions for all of its descendants.
        Descendants are added to the store if they aren't already in it.
        Overwrites the splitlist in the store if desired (for the main
        construction only, not for descendants).

        Arguments:
            construction: Construction object to add contribution to
            overwrite_splitlist: boolean to allow for splitlist in store
                                    to be overwritten (default = False)
        """
        self._add_isolated(construction, overwrite_splitlist=overwrite_splitlist, change_rcount=True)
        self._add_descendants(construction)

    def change_splitlist(self, construction, new_splitlist, branching=True, red_delayed=False):
        """Changes the splitlist of the stored construction,
        moving associated counts, badsplits, and redspans
        from previous children to new children.
        If multiple splits are provided, setting branching to True
        will distribute these splits hierarchically among successive
        children (following the default branching scheme laid out
        in SplitStore.to_splittree), such that each construction only
        has a single split, whereas setting branching to False will
        store all of the splits at the current construction, in a flat fashion.
        Raises an error if the construction is not in the store.

        Arguments:
            construction: Construction object to change splitlist of
            new_splitlist: new splitlist to use for construction
            branching: boolean to distribute multuple splits hierarchially
                        among successive children (default = True)
            red_delayed: boolean to delay Split objects so tha they
                            introduce terminal nodes (?) (Default = False)
            """
        if construction in self:
            # Check for multiple splits to be distributed hierarchically
            if len(new_splitlist) > 1 and branching:
                self.set_splittree(construction, new_splitlist.to_splittree(red_delayed=red_delayed))
            else:
                # Work off the construction in the store
                construction = self[construction]
                # First check that the new splitlist is actually new!
                if construction.splitlist == new_splitlist:
                    return
                # Remove the construction's contribution to its children
                self._remove_descendants(construction)
                # Set the new splitlist
                construction.splitlist = new_splitlist
                # Create contributions to the new descendants
                self._add_descendants(construction)
        else:
            raise Exception("Attempting to change splits for unknown word: {}".format(construction))

    def split(self, construction, new_split):
        """Forms a binary Split in a Construction in the store,
        overwriting the existing splitlist.
        If the construction is not in the store, raises an error.

        Arguments:
            construction: Construction object to Split
            new_split: new Split to store
        """
        self.change_splitlist(construction, SplitStore([new_split]))

    def segment(self, construction):
        """Gets the segmentation of a Construction, i.e. the decomposition
        into terminal (real) constructions.
        Returns a list of Construction objects, in linear order.
        Each Construction object represents a contribution to the store,
        not necessarily the entirety of the construction's representation
        in the store."""
        # Quick check that the construction is in the store,
        # to avoid adding new constructions during the check
        if construction not in self:
            return None
        return [contribution for contribution in self._get_contributions(construction, "remove") if
                not contribution.has_children]

    def get_compounds(self):
        """Returns a list of the compounds in the store,
        i.e. the constructions with non-zero r_count"""
        return [construction for construction in self if construction.r_count > 0]

    def get_constructions(self):
        """Returns a list of the terminal constructions (morphs) in the store,
        sorted alphabetically"""
        return sorted(construction for construction in self if not construction.splitlist)

    def get_segmentations(self, get_trees=False):
        """Retrieve segmentations for all compounds encoded by the model.

        Arguments:
            get_trees: boolean to yield string form of
                        the given construction according
                        to its splittree (default = False)
        """
        for compound in sorted(self.get_compounds()):
            constructions = self.segment(compound)
            if get_trees:
                tree = self.get_tree_str(compound)
            else:
                tree = None
            yield compound, constructions, tree

    def get_tree_str(self, construction):
        """Returns a string form of the given construction, bracketed
        according to its splittree.
        Reduplicants retain their original atoms, but are enclosed in <>."""
        splittree = self.get_splittree(construction)
        brackets = [""] * (len(construction) + 1)
        brackets = splittree.to_brackets(brackets)
        # Interleave the brackets with the atoms
        combined = "".join(symbol for pair in zip(brackets, list(construction) + [""]) for symbol in pair)
        # Add spaces
        combined = re.sub(r"\s*(\[+)", r" \1", combined)
        combined = re.sub(r"(\]+)\s*", r"\1 ", combined)
        combined = re.sub(r"\[([^\s]+)\]", r"\1", combined)
        combined = combined.strip()

        return combined

    def get_splittree(self, construction, offset=Split(0)):
        """Returns a SplitTree for the given Construction,
        based on the splits in the store

        Arguments:
            construction: Construction object to retrieve SplitTree
            offset: (?) (default = Split(0))
        """
        # Get the stored version of the construction
        if construction not in self:
            return SplitTree(terminal=True)
        construction = self[construction]
        if construction.has_children:
            # Convert SplitStore to default SplitTree
            tree = construction.splitlist.to_splittree()
            tree = tree.reindex(offset)
            children = construction.get_children()
            # Slot subtrees corresponding to children into terminals
            for child, (parent_tree, side) in zip(children, tree.get_terminal_parents()):
                if side == "L":
                    parent_tree.L_subtree = self.get_splittree(child, offset=offset)
                elif side == "R":
                    new_offset = Split(-parent_tree.split.splitloc)
                    parent_tree.R_subtree = self.get_splittree(child, offset=new_offset)
            return tree
        else:
            return SplitTree(terminal=True)

    def set_splittree(self, construction, splittree):
        """Sets the splittree for a construction in the store to the
        provided SplitTree. This sets the splitlist for the construction
        to the highest binary split, and then recurses down through children,
        setting splitlists to the remaining binary splits.

        Arguments:
            construction: Construction object to set provided SplitTree
            splittree: SplitTree object to set with construction
        """
        # Raise an exception if the construction is unknown
        if construction not in self:
            raise Exception("Setting split tree for unknown construction: {}".format(construction))
        # Work off the version of the construction in the store
        construction = self[construction]
        if not splittree.is_branching:
            self._remove_descendants(construction)
        else:
            self.split(construction, splittree.split)
            L_child, R_child = construction.split(splittree.split)
            self.set_splittree(L_child, splittree.L_subtree)
            self.set_splittree(R_child, splittree.R_subtree.reindex(splittree.split))

    def get_red_enforced_splittree(self, construction, allow_restructuring=False, full_red_only=False):
        """Gets a SplitTree that enforces RedSplits wherever known reduplication
        has been recapitulated. That is, if the segmentation of a construction
        isolates the reduplicant and leaves the minimal base intact,
        without making use of a RedSplit, this method returns a tree that
        uses a RedSplit (provided the Construction has a corresponding RedSpan).
        If allow_restructuring is False, a reduplicant must be at the
        construction edge in order for the corresponding RedSplit to be
        enforced; thus, a structure like [[mana ti] tia] or [[mana ti][tia Na]]
        would not be considered an instance of reduplication, but something like
        [ti [tia Na]] would be. Conversely, if allow_restructuring is
        True, any reduplicant that is linearly adjacent to its minimal
        base would have the corresponding RedSplit enforced, and the tree
        would be restructured to ensure that the RedSplit is at the edge.
        Note: if full_red_only is True, restructuring is also allowed for
        the purposes of delaying the reduplication

        Arguments:
            construction: Construction object to retrieve SplitTree with enforcded RedSplits
            allow_restructuring: boolean to allow restructuring of the SplitTree to ensure
                                    that the RedSplit is at the edge (default = False)
            full_red_only: boolean to also allow restructuring for delaying
                            the reduplication (default = False)
        """
        # Make sure to use the version of the construction in the store
        if construction not in self:
            return SplitTree(terminal=True)
        construction = self[construction]
        tree = self.get_splittree(construction)
        splits = tree.to_splitlist()
        candidate_redsplits = tree.get_recapitulated_redsplits(construction)
        # If only checking for full reduplication, remove any candidate redsplits
        # that don't have a split at the minbase edge
        if full_red_only:
            candidate_redsplits = [redsplit for redsplit in candidate_redsplits if (redsplit.has_base_at_edge_of(construction) or redsplit.minbase_edge in splits.splitlocs)]
        # Get the edgesplits corresponding to each candidate redsplit
        edgesplits = [splits.get(redsplit.red_edge) for redsplit in candidate_redsplits]
        # Consider each candidate in turn
        modified_tree = tree
        for redsplit, edgesplit in zip(candidate_redsplits, edgesplits):
            # If allow_restructuring is True, relabel every split
            # Otherwise, relabel only if edgesplit is None (i.e. at
            # the construction edge) or is an ancestor of redsplit
            if allow_restructuring or edgesplit is None or tree.is_ancestor(edgesplit, redsplit):
                modified_tree = modified_tree.update_split_type(redsplit)
        # If checking for full reduplication only, ensure that RED is delayed
        if full_red_only:
            modified_tree = modified_tree.enforce_delayed_red()
        # If allow_restructuring is True, restructure to put RED at the edge
        elif allow_restructuring:
            modified_tree = modified_tree.enforce_edge_red()

        return modified_tree

    def remove_nonterminals(self):
        """Returns a new version of the store in which all Constructions
        with children have been removed"""
        new_store = ConstructionStore()
        for construction in self:
            if not construction.has_children:
                new_store.add(construction)


class ReduplicationFinder(object):
    """Class containing methods for identifying reduplication in a construction"""
    __slots__ = (
    'Lred_minbase_weight', 'Rred_minbase_weight', 'check_left_red', 'check_right_red', 'default_red_attachment',
    'full_red_only', 'label_by_kind', 'disable_fullred', 'eliminate_conflict')

    def __init__(self, Lred_minbase_weight, Rred_minbase_weight, check_left_red=True, check_right_red=False,
                 default_red_attachment="L", full_red_only=False, disable_fullred=False, 
                 label_by_kind=False, eliminate_conflict=True):
        """Initialize with preferences for RED identification.

        Arguments:
            Lred_minbase_weight: minimum permissible base weight for reduplication on left (?)
            Rred_minbase_weight: minimum permissible base weight for reduplication on right (?)
            check_left_red: Bool to check for left reduplication (?) (default = True)
            check_right_red: (?) (default = False)
            default_red_attachment: (?) (default = "L")
            full_red_only: (?) (default = False)
            disable_fullred: (?) (default = False)
            label_by_kind: (?) (default = False)
            eliminate_conflict: (?) (default = True)
        """
        self.Lred_minbase_weight = Lred_minbase_weight
        self.Rred_minbase_weight = Rred_minbase_weight
        self.check_left_red = check_left_red
        self.check_right_red = check_right_red
        self.default_red_attachment = default_red_attachment
        self.full_red_only = full_red_only
        self.disable_fullred = disable_fullred
        self.label_by_kind = label_by_kind
        self.eliminate_conflict = eliminate_conflict

    def __copy__(self):
        """A method for creating a copy with the same attributes, for speed"""
        return ReduplicationFinder(self.Lred_minbase_weight, self.Rred_minbase_weight, self.check_left_red,
                                   self.check_right_red, self.default_red_attachment, self.full_red_only,
                                   self.disable_fullred, self.label_by_kind, self.eliminate_conflict)

    def _get_left_reduplication(self, construction, splitloc):
        """Gets the left-attaching RedSpan with longest reduplicant
        that can be formed at a given split location.
        If no valid RedSpan can be formed, returns None.

        Arguments:
              construction: Construction object to retrieve left-attaching RedSpan
              splitloc: split location
        """
        # Fix red_end at splitloc and move red_start from 0 to splitloc
        red_start = 0
        red_end = splitloc
        while red_start < red_end:
            # Get the minimum base end
            minbase_end = self._get_minbase_end(construction, red_start, red_end)
            # Only proceed if a base can be formed
            if minbase_end <= len(construction):
                # Make a RedSpan object and return if valid
                redspan = RedSpan(construction, red_start, red_end, minbase_end, label_by_kind=self.label_by_kind)
                if self._is_valid_redspan(redspan):
                    return redspan
            # Otherwise, try moving red_start
            red_start += 1

        # If no RedSpan can be formed, return None
        return None

    def _get_right_reduplication(self, construction, splitloc):
        """Gets the right-attaching RedSpan with longest reduplicant
        that can be formed at a given split location.
        If no valid RedSpan can be formed, returns None.

        Arguments:
              construction: Construction object to retrieve right-attaching RedSpan
              splitloc: split location
        """
        # Fix red_start at splitloc and move red_end from len(construction) to splitloc
        red_start = splitloc
        red_end = len(construction)
        while red_start < red_end:
            # Get the minimum base start
            minbase_start = self._get_minbase_start(construction, red_start, red_end)
            # Only proceed if a base can be formed
            if minbase_start >= 0:
                # Make a RedSpan object and return if valid
                redspan = RedSpan(construction, red_start, red_end, minbase_start, label_by_kind=self.label_by_kind)
                if self._is_valid_redspan(redspan):
                    return redspan
            # Otherwise, try moving red_end
            red_end -= 1

        # If no RedSpan can be formed, return None
        return None

    def _are_equivalent(self, redspan1, redspan2):
        """Checks if two redspans are equivalent.
        Equivalent means that they just swap the role of RED and base"""
        return (redspan1.attachment != redspan2.attachment and
                redspan1.splitloc == redspan2.splitloc and
                redspan1.red_edge == redspan2.minbase_edge and
                redspan1.minbase_edge == redspan2.red_edge)

    def _get_nondefault_equivalent(self, redspan1, redspan2):
        """Gets the non-default of two equivalent RedSpans."""
        if redspan1.attachment != self.default_red_attachment:
            return redspan1
        elif redspan2.attachment != self.default_red_attachment:
            return redspan2
        return None

    def _are_conflicting(self, redspan1, redspan2):
        """Checks if two redspans are conflicting.
        Conflicting means that they have the same attachment side,
        and the splitloc of one is a badsplit of the other."""
        return (redspan1.attachment == redspan2.attachment and
                (redspan1.splitloc in redspan2.badsplits or
                 redspan2.splitloc in redspan1.badsplits))

    def _get_conflict_loser(self, redspan1, redspan2):
        """Gets the loser of a conflict between two RedSpans.
        The loser is the one with the shortest reduplicant, in syllables.
        If both have the same length reduplicant, return None."""
        if len(redspan1.reduplicant) < len(redspan2.reduplicant):
            return redspan1
        elif len(redspan2.reduplicant) < len(redspan1.reduplicant):
            return redspan2
        return None

    def find_redspans(self, construction, badsplits=None):
        """Finds all of the RedSpans that are consistent with a construction,
        and returns them in a RedSpanTuple.
        If self.eliminate_conflict is True, conflicting RedSpans are removed.
        This means:
        1. When two redspans with the same attachment overlap, only the largest
        is kept as a candidate (or both if they are the same size).
        2. When there are two equivalent redspans (with different attachments),
        only the one with the default attachment is kept.
        Returns a RedSpanTuple.

        Arguments:
            construction: Construction object
            badsplits: list of badsplits for the construction
        """
        # If not analyzing reduplication, there are no possible instances of reduplication
        if not (self.check_left_red or self.check_right_red):
            return (RedSpanStore())
        # For a length 1 construction, there are no possible instances of reduplication
        if len(construction) == 1:
            return (RedSpanStore())
        # For a length 2 construction, there are no possible instances of reduplication unless both vowels are long
        if len(construction) == 2 and construction.weight < 4:
            return (RedSpanStore())

        # Basic idea: iterate through possible split locations,
        # getting RED instances as they are valid.
        # Assume that impossible split locations are in badsplits
        potential_L_redspans = []
        potential_R_redspans = []
        for splitloc in range(len(construction) - 1, 0, -1):
            # Ignore cases where the split is invalid
            if splitloc in construction.badsplits:
                continue

            # Get valid RedSpans
            if self.check_left_red:
                L_redspan = self._get_left_reduplication(construction, splitloc)
                if L_redspan is not None:
                    potential_L_redspans.append(L_redspan)
            if self.check_right_red:
                R_redspan = self._get_right_reduplication(construction, splitloc)
                if R_redspan is not None:
                    potential_R_redspans.append(R_redspan)

        # Find RedSpans to eliminate, if eliminating
        removed_redspans = set()
        if self.eliminate_conflict:
            # Sort the potential redspan lists from longest to shortest RED,
            # to assist in conflict removal
            potential_L_redspans.sort(key=lambda redspan: len(redspan.reduplicant), reverse=True)
            potential_R_redspans.sort(key=lambda redspan: len(redspan.reduplicant), reverse=True)

            # Remove conflict losers, starting from those with the largest REDs.
            # Going in this order means that we will avoid over-removal when a RedSpan
            # conflicts with a RedSpan to either side that *do not* conflict with each
            # other, where one has a longer RED and the other has a shorter RED.
            # If we removed the short-RED RedSpan first (through conflict with the medium-RED
            # one), we would still have to remove the medium-RED RedSpan (through conflict
            # with the long-RED one). By removing the medium-RED RedSpan first, we can
            # keep the short-RED RedSpan, because it does not conflict with the long-RED RedSpan.
            for redspan_list in [potential_L_redspans, potential_R_redspans]:
                for i in range(len(redspan_list) - 1):
                    redspan1 = redspan_list[i]
                    if redspan1 not in removed_redspans:
                        for j in range(i + 1, len(redspan_list)):
                            redspan2 = redspan_list[j]
                            if redspan2 not in removed_redspans:
                                if self._are_conflicting(redspan1, redspan2):
                                    conflict_loser = self._get_conflict_loser(redspan1, redspan2)
                                    if conflict_loser is not None:
                                        removed_redspans.add(conflict_loser)

            # Also remove equivalent full-RED spans from the non-default attachment
            if self.check_left_red and self.check_right_red:
                for L_redspan in potential_L_redspans:
                    if L_redspan not in removed_redspans:
                        for R_redspan in potential_R_redspans:
                            if R_redspan not in removed_redspans:
                                if self._are_equivalent(L_redspan, R_redspan):
                                    nondefault_redspan = self._get_nondefault_equivalent(L_redspan, R_redspan)
                                    removed_redspans.add(nondefault_redspan)

        # Return a RedSpanStore of the found RedSpans
        return RedSpanStore(
            (redspan, {construction.label}) for redspan in (potential_L_redspans + potential_R_redspans) if
            redspan not in removed_redspans)

    def _get_minbase_end(self, construction, red_start, red_end):
        """Gets the location of the end of the minimal base,
        assuming left-attaching reduplication.

        Arguments:
            construction: Construction object
            red_start: start of the reduplicant
            red_end: end of the reduplicant
        """
        base_start = red_end
        base_end = base_start + (red_end - red_start)
        while (base_end <= len(construction) and 
            (construction[base_start:base_end].weight < self.Lred_minbase_weight or 
            (self.disable_fullred and construction[red_start:red_end] == construction[base_start:base_end]))):
            base_end += 1
        return base_end

    def _get_minbase_start(self, construction, red_start, red_end):
        """Gets the location of the start of the minimal base,
        assuming right-attaching reduplication.

        Arguments:
            construction: Construction object
            red_start: start of the reduplicant
            red_end: end of the reduplicant
        """
        base_end = red_start
        base_start = base_end - (red_end - red_start)
        while (base_start >= 0 and 
            (construction[base_start:base_end].weight < self.Rred_minbase_weight or
            (self.disable_fullred and construction[red_start:red_end] == construction[base_start:base_end]))):
            base_start -= 1
        return base_start

    def _is_valid_redspan(self, redspan):
        """Checks whether a RedSpan object is valid."""
        # Disabled full-reduplication (for the case where adjusting the minbase was not possible)
        if self.disable_fullred:
            # It is NOT reduplication if the reduplicant matches the minbase
            if redspan.reduplicant == redspan.minbase:
                return False
        # Full-reduplication only
        if self.full_red_only:
            # It is only valid if the reduplicant matches the minbase
            return redspan.reduplicant == redspan.minbase
        # Left-attaching (partial) RED
        elif redspan.attachment == "L":
            # It can only be reduplication if the reduplicant is polysyllabic or monosyllabic CV
            if len(redspan.reduplicant) > 1 or redspan.reduplicant.initial_segment in "ptkmnNwrfh":
                # It IS reduplication if red is the same as the base portion
                if redspan.reduplicant == redspan.base_portion:
                    return True
                # It IS reduplication if red is monosyllabic and the same as initial-lengthened base portion
                if (len(redspan.reduplicant) == 1 and
                        redspan.reduplicant == redspan.base_portion.lengthen_initial()):
                    return True
        # Right-attaching RED
        elif redspan.attachment == "R":
            # It is NOT reduplication if RED is less than 2 morae
            if redspan.reduplicant.weight < 2:
                return False
            # It is NOT reduplication if a minbase of weight 4+ does not start with a long vowel
            if redspan.minbase.weight >= 4 and redspan.minbase != redspan.minbase.lengthen_initial():
                return False
            # It IS reduplication if red is the same as the base portion
            if redspan.reduplicant == redspan.base_portion:
                return True
        # In all other cases, it is not reduplication
        return False