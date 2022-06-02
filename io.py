import bz2
import codecs
import datetime
import gzip
import locale
import logging
import re
import sys
import copy

from . import get_version
from .representations import Construction, ReduplicationFinder

try:
    # In Python2 import cPickle for better performance
    import cPickle as pickle
except ImportError:
    import pickle

PY3 = sys.version_info[0] == 3

_logger = logging.getLogger(__name__)


class MorfessorIO(object):
    """Definition for all input and output files. Also handles all
    encoding issues.

    This class has states for the separators used in the data,
    the patterns of reduplication, and the places splits should be
    forced or avoided.
    If multiple input files are from the same language and have the same
    separators, the same class instance can be used for initializing them.
    """

    def __init__(self, encoding=None, comment_start='#', lowercase=False,
                 compound_separator=r'\s+', construction_separator=' + ',
                 atom_separator=None, strip_atom_sep=True,
                 nosplit_re=None, split_re=None, forcesplit_chars=None,
                 check_left_red=False, Lred_minbase_weight=1,
                 check_right_red=False, Rred_minbase_weight=1,
                 default_red_attachment="L", full_red_only=False,
                 disable_fullred=False, separate_red=False):
        """
            encoding: a string representation of the encoding used for files.
                      If None, the system encoding is used for writing files,
                      and encoding is inferred when reading files.
            comment_start: a string that starts comment lines, which are skipped
                           when reading files.
            lowercase: a Boolean flag for whether to convert to lowercase
                       when reading text files.
            compound_separator: a regular expression matching the separator of
                                compounds in input files.
            construction_separator: a string that is used to separate constructions
                                    in input and output files.
            atom_separator: a character that is used to separate atoms in the input,
                            and in the output (if strip_atom_sep is False).
            strip_atom_sep: a Boolean flag for whether to remove the atom separator
                            when writing output.
            nosplit_re: regular expression string for preventing splitting
                        in certain contexts. The expression should match an
                        atom separator that should not be converted to a split,
                        with up to one atom either side as environment.
            split_re: a regex for the atom boundaries that should be pre-split
                      into constructions. The expression should match an
                      atom separator that should be converted to a forced split,
                      with up to one atom either side as environment.
            forcesplit_chars: force segmentations on the characters in
                              the given str. These split characters are
                              not represented in the store and trigger one
                              input to be represented as multiple compounds.
            check_left_red: Boolean flag for whether or not to consider
                            possible left-attaching reduplication
            Lred_minbase_weight: the minimum weight of a base required for valid
                                 left-reduplication (using the weight property
                                 of a Construction, specified in
                                 representations.py)
            check_right_red: Boolean flag for whether or not to consider
                             possible right-attaching reduplication
            Rred_minbase_weight: the minimum weight of a base required for
                                 right-reduplication (using the weight property
                                 of a Construction, specified in
                                 representations.py)
            default_red_attachment: default side for attachment of full red
                                    when checking reduplication on both sides.
            full_red_only: Boolean. If True, all cases of reduplication will be coerced
                           to full reduplication.
            disable_fullred: Boolean. If True, full reduplication is not recognized as
                             reduplication.
            separate_red: Boolean flag for whether or not RED counts should be
                          separated based on reduplicant length and final lengthening
        """
        self.encoding = encoding
        self.construction_separator = construction_separator
        self.comment_start = comment_start
        if compound_separator == "":
            self.compound_sep_re = None
        else:
            self.compound_sep_re = re.compile(compound_separator, re.UNICODE)
        if atom_separator is None:
            self.atom_separator = ""
            self.strip_atom_sep = True
            self._atom_sep_re = None
        else:
            self.atom_separator = atom_separator
            self.strip_atom_sep = strip_atom_sep
            self._atom_sep_re = re.compile(atom_separator, re.UNICODE)
        self.lowercase = lowercase
        self._version = get_version()

        # If only looking for full reduplication, and no side has been given,
        # operationalize as the default side
        if full_red_only and not (check_left_red or check_right_red):
            check_left_red = (default_red_attachment == "L")
            check_right_red = (default_red_attachment == "R")

        # Properties for the population of Construction fields
        if not (check_left_red or check_right_red):
            self._redFinder = None
        else:
            if not check_left_red:
                default_red_attachment = "R"
            elif not check_right_red:
                default_red_attachment = "L"
            self._redFinder = ReduplicationFinder(Lred_minbase_weight, Rred_minbase_weight,
                                                  check_left_red=check_left_red, check_right_red=check_right_red,
                                                  default_red_attachment=default_red_attachment,
                                                  full_red_only=full_red_only, disable_fullred=disable_fullred, label_by_kind=separate_red)
        if forcesplit_chars:
            self._subcompound_re = re.compile("(?:{})*[{}](?:{})*".format(self.atom_separator, "".join(forcesplit_chars), self.atom_separator), re.UNICODE)
        else:
            self._subcompound_re = None
        if nosplit_re:
            self._nosplit_re = re.compile(nosplit_re, re.UNICODE)
        else:
            self._nosplit_re = None
        if split_re:
            self._split_re = re.compile(split_re, re.UNICODE)
        else:
            self._split_re = None

    def read_segmentation_file(self, file_name, has_counts=True, **kwargs):
        """Read segmentation file.

        File format:
        <count>\t<compound>\t<construction1><sep><construction2><sep>...<constructionN>

        """
        # Find all possible reduplications, in case attested ones were conflict losers
        redFinder = copy.copy(self._redFinder)
        if redFinder is not None:
            redFinder.eliminate_conflict = False
        _logger.info("Reading segmentations from '%s'...", file_name)
        for line in self._read_text_file(file_name):
            if has_counts:
                count, compound_str, compound_analysis = line.split('\t')
                count = int(count)
            else:
                count = 1
                compound_str, compound_analysis = line.split('\t')
            for subcompound_str, subcompound_analysis in self._split_into_subcompounds(compound_str, compound_analysis):
                subcompound = self._make_construction(subcompound_str, r_count=count, redFinder=redFinder)
                constructions = self._get_parts(subcompound, subcompound_analysis)
                yield subcompound, constructions
        _logger.info("Done.")

    def write_segmentation_file(self, file_name, segmentations, redup_info="", **kwargs):
        """Write segmentation file.

        File format:
        <count>\t<compound>\t<construction1><sep><construction2><sep>...<constructionN>

        if --get-trees was provided at cmd, adds an extra column containing the parse trees
        """
        _logger.info("Saving segmentations to '%s'...", file_name)
        with self._open_text_file_write(file_name) as file_obj:
            d = datetime.datetime.now().replace(microsecond=0)
            file_obj.write("# Output from Morfessor Baseline %s%s, %s\n" %
                           (self._version, redup_info, d))
            for compound, constructions, tree in segmentations:
                count = compound.r_count
                if self.strip_atom_sep:
                    compound_str = compound.label
                    s = self.construction_separator.join(
                        constr.label
                        for constr in constructions)
                else:
                    compound_str = self.atom_separator.join(compound)
                    s = self.construction_separator.join(
                        (self.atom_separator.join(constr)
                         for constr in constructions))
                if tree is not None:
                    file_obj.write("%d\t%s\t%s\t%s\n" % (count, compound_str, s, tree))
                else:
                    file_obj.write("%d\t%s\t%s\n" % (count, compound_str, s))
        _logger.info("Done.")

    def read_corpus_files(self, file_names):
        """Read one or more corpus files.

        Yield for each compound found a Construction object.

        """
        for file_name in file_names:
            for item in self.read_corpus_file(file_name):
                yield item

    def read_corpus_list_files(self, file_names):
        """Read one or more corpus list files.

        Yield for each compound found a Construction object.

        """
        for file_name in file_names:
            for item in self.read_corpus_list_file(file_name):
                yield item

    def read_corpus_file(self, file_name):
        """Read one corpus file.

        For each compound, yield a Construction object.
        After each line, yield None.

        """
        _logger.info("Reading corpus from '%s'...", file_name)
        for line in self._read_text_file(file_name, raw=True):
            if self.compound_sep_re is None:
                for subcompound in self._split_into_subcompounds(line):
                    if len(subcompound) > 0:
                        yield self._make_construction(subcompound, r_count=1)
            else:
                for compound in self.compound_sep_re.split(line):
                    for subcompound in self._split_into_subcompounds(compound):
                        if len(subcompound) > 0:
                            yield self._make_construction(subcompound, r_count=1)
            yield None
        _logger.info("Done.")

    def read_corpus_list_file(self, file_name):
        """Read a corpus list file.

        Each line has the format:
        <count> <compound>

        Yield a Construction object for each compound.

        """
        _logger.info("Reading corpus from list '%s'...", file_name)
        for line in self._read_text_file(file_name):
            try:
                count, compound = line.split(None, 1)
                for subcompound in self._split_into_subcompounds(compound):
                    yield self._make_construction(subcompound, r_count=int(count))
            except ValueError:
                for subcompound in self._split_into_subcompounds(line):
                    yield self._make_construction(subcompound, r_count=1)
        _logger.info("Done.")

    def read_annotations_file(self, file_name, analysis_sep=','):
        """Read a annotations file.

        Each line has the format:
        <compound>\t<constr1> + <constr2>... + <constrN>, <constr1>...<constrN>, ...

        Yield a dict mapping from compound to list(analyses).

        """
        # Find all possible reduplications, in case attested ones were conflict losers
        redFinder = copy.copy(self._redFinder)
        if redFinder is not None:
            redFinder.eliminate_conflict = False
        annotations = {}
        _logger.info("Reading annotations from '%s'...", file_name)
        for line in self._read_text_file(file_name):
            compound_str, analyses_line = line.split("\t", 1)
            if analysis_sep is None or analysis_sep not in analyses_line:
                analyses = [analyses_line]
            else:
                analyses = analyses_line.split(analysis_sep)
            for analysis in analyses:
                analysis = analysis.strip()
                for subcompound_str, subcompound_analysis in self._split_into_subcompounds(compound_str, analysis):
                    subcompound = self._make_construction(subcompound_str, redFinder=redFinder)
                    if subcompound not in annotations:
                        annotations[subcompound] = []
                    constructions = self._get_parts(subcompound, subcompound_analysis)
                    annotations[subcompound].append(constructions)

        _logger.info("Done.")
        return annotations

    def write_lexicon_file(self, file_name, lexicon):
        """Write to a Lexicon file all constructions and their counts."""
        _logger.info("Saving model lexicon to '%s'...", file_name)
        with self._open_text_file_write(file_name) as file_obj:
            for construction in lexicon:
                count = construction.count
                if self.strip_atom_sep:
                    construction_str = construction.label
                else:
                    construction_str = self.atom_separator.join(construction)
                file_obj.write("%d %s\n" % (count, construction_str))
        _logger.info("Done.")

    def read_binary_model_file(self, file_name):
        """Read a pickled model from file."""
        _logger.info("Loading model from '%s'...", file_name)
        model = self.read_binary_file(file_name)
        _logger.info("Done.")
        return model

    @staticmethod
    def read_binary_file(file_name):
        """Read a pickled object from a file."""
        with open(file_name, 'rb') as fobj:
            obj = pickle.load(fobj)
        return obj

    def write_binary_model_file(self, file_name, model):
        """Pickle a model to a file."""
        _logger.info("Saving model to '%s'...", file_name)
        self.write_binary_file(file_name, model)
        _logger.info("Done.")

    @staticmethod
    def write_binary_file(file_name, obj):
        """Pickle an object into a file."""
        with open(file_name, 'wb') as fobj:
            pickle.dump(obj, fobj, pickle.HIGHEST_PROTOCOL)

    def write_parameter_file(self, file_name, params):
        """Write learned or estimated parameters to a file"""
        with self._open_text_file_write(file_name) as file_obj:
            d = datetime.datetime.now().replace(microsecond=0)
            file_obj.write(
                '# Parameters for Morfessor {}, {}\n'.format(
                    self._version, d))
            for (key, val) in params.items():
                file_obj.write('{}:\t{}\n'.format(key, val))

    def read_parameter_file(self, file_name):
        """Read learned or estimated parameters from a file"""
        params = {}
        line_re = re.compile(r'^(.*)\s*:\s*(.*)$')
        for line in self._read_text_file(file_name):
            m = line_re.match(line.rstrip())
            if m:
                key = m.group(1)
                val = m.group(2)
                try:
                    val = float(val)
                except ValueError:
                    pass
                params[key] = val
        return params

    def read_any_model(self, file_name):
        """Read a file that is either a binary model or a Morfessor 1.0 style
        model segmentation. This method can not be used on standard input as
        data might need to be read multiple times"""
        try:
            model = self.read_binary_model_file(file_name)
            _logger.info("%s was read as a binary model", file_name)
            return model
        except BaseException:
            pass

        from morfessor import BaselineModel
        model = BaselineModel()
        model.load_segmentations(self.read_segmentation_file(file_name))
        _logger.info("%s was read as a segmentation", file_name)
        return model

    def format_constructions(self, constructions):
        """Return a formatted string for a list of constructions."""
        if isinstance(constructions[0], str):
            # Constructions are strings
            return self.construction_separator.join(constructions)
        else:
            # Constructions are not strings (should be tuples of strings)
            if self.strip_atom_sep:
                return self.construction_separator.join(
                    map(lambda x: "".join(x), constructions))
            else:
                return self.construction_separator.join(
                    map(lambda x: self.atom_separator.join(x), constructions))

    def _split_into_subcompounds(self, compound_str, analysis_str=None):
        """Forcibly split string representations of a compound and
        corresponding analysis based on characters that mark
        clear construction boundaries (e.g. hyphens), as provided in
        the original forcesplit_chars argument."""
        if self._subcompound_re is None:
            if analysis_str is None:
                return [compound_str]
            else:
                return [(compound_str, analysis_str)]
        else:
            subcompounds = self._subcompound_re.split(compound_str)
            if analysis_str is None:
                return subcompounds
            else:
                subanalyses = self._subcompound_re.split(analysis_str)
                return list(zip(subcompounds, subanalyses))

    def _make_construction(self, construction_str, r_count=0, count=0, redFinder=None):
        """Convert a construction string into a Construction object"""
        # Allow for a special RedFinder to be provided, for reading of segmentations
        if redFinder is None:
            redFinder = self._redFinder
        if r_count > 0 and count == 0:
            count = r_count
        atoms = self._split_atoms(construction_str)
        construction = Construction(atoms, r_count=r_count, count=count)
        if redFinder is not None:
            construction.populate_redspans(redFinder)
        if self._nosplit_re is not None:
            construction.populate_badsplits(self._nosplit_re, self.atom_separator)
        if self._split_re is not None:
            construction.populate_forced_splitlocs(self._split_re, self.atom_separator)
        return construction

    def _split_atoms(self, construction):
        """Split construction to its atoms."""
        if self.atom_separator == "":
            return tuple(construction)
        else:
            return tuple(self._atom_sep_re.split(construction))

    def _get_parts(self, construction, analysis_str):
        """Gets Construction objects representing the given analysis
        of a construction"""
        modified_construction = Construction(construction._atoms, redspans=construction.redspans)
        parts = analysis_str.split(self.construction_separator)
        part_atoms = [self._split_atoms(part) for part in parts if len(parts) > 0]
        splitlist = modified_construction.segmentation_to_splitlist(part_atoms)
        modified_construction.splitlist = splitlist
        return modified_construction.get_children()

    def _open_text_file_write(self, file_name):
        """Open a file for writing with the appropriate compression/encoding"""
        if file_name == '-':
            file_obj = sys.stdout
            if PY3:
                return file_obj
        elif file_name.endswith('.gz'):
            file_obj = gzip.open(file_name, 'wb')
        elif file_name.endswith('.bz2'):
            file_obj = bz2.BZ2File(file_name, 'wb')
        else:
            file_obj = open(file_name, 'wb')
        if self.encoding is None:
            # Take encoding from locale if not set so far
            self.encoding = locale.getpreferredencoding()
        return codecs.getwriter(self.encoding)(file_obj)

    def _open_text_file_read(self, file_name):
        """Open a file for reading with the appropriate compression/encoding"""
        if file_name == '-':
            if PY3:
                inp = sys.stdin
            else:
                class StdinUnicodeReader:
                    def __init__(self, encoding):
                        self.encoding = encoding
                        if self.encoding is None:
                            self.encoding = locale.getpreferredencoding()

                    def __iter__(self):
                        return self

                    def next(self):
                        l = sys.stdin.readline()
                        if not l:
                            raise StopIteration()
                        return l.decode(self.encoding)

                inp = StdinUnicodeReader(self.encoding)
        else:
            if file_name.endswith('.gz'):
                file_obj = gzip.open(file_name, 'rb')
            elif file_name.endswith('.bz2'):
                file_obj = bz2.BZ2File(file_name, 'rb')
            else:
                file_obj = open(file_name, 'rb')
            if self.encoding is None:
                # Try to determine encoding if not set so far
                self.encoding = self._find_encoding(file_name)
            inp = codecs.getreader(self.encoding)(file_obj)
        return inp

    def _read_text_file(self, file_name, raw=False):
        """Read a text file with the appropriate compression and encoding.

        Comments and empty lines are skipped unless raw is True.

        """
        inp = self._open_text_file_read(file_name)
        try:
            for line in inp:
                line = line.rstrip()
                if not raw and \
                   (len(line) == 0 or line.startswith(self.comment_start)):
                    continue
                if self.lowercase:
                    yield line.lower()
                else:
                    yield line
        except KeyboardInterrupt:
            if file_name == '-':
                _logger.info("Finished reading from stdin")
                return
            else:
                raise

    @staticmethod
    def _find_encoding(*files):
        """Test default encodings on reading files.

        If no encoding is given, this method can be used to test which
        of the default encodings would work.

        """
        test_encodings = ['utf-8', locale.getpreferredencoding()]
        for encoding in test_encodings:
            ok = True
            for f in files:
                if f == '-':
                    continue
                try:
                    if f.endswith('.gz'):
                        file_obj = gzip.open(f, 'rb')
                    elif f.endswith('.bz2'):
                        file_obj = bz2.BZ2File(f, 'rb')
                    else:
                        file_obj = open(f, 'rb')

                    for _ in codecs.getreader(encoding)(file_obj):
                        pass
                except UnicodeDecodeError:
                    ok = False
                    break
            if ok:
                _logger.info("Detected %s encoding", encoding)
                return encoding

        raise UnicodeError("Can not determine encoding of input files")
