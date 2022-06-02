# morfessoRED: Morfessor 2.0 with reduplication
*Simon Todd, 1 June 2022*

This code accompanies the following paper:

> Todd, Huang, Needle, Hay, and King (to appear). Unsupervised morphological segmentation in a language with reduplication. In Proceedings of the 19th SIGMORPHON Workshop on Computational Research in Phonetics, Phonology, and Morphology. Stroudsburg, PA: Association for Computational Linguistics.

**Please note:**  
- Documentation and tidying of the code is ongoing. Flags may change during this process.  
- We will add detailed description of the model adjustments when we are able.  
- The data used to train models in the paper cannot be provided due to licensing restrictions.  
- For now, this code is a patch that uses the infrastructure of Morfessor 2.0 and replaces some of the source files of that package. We will not distribute it as a standalone Python package until documentation is complete and all features have been thoroughly tested.  
- The reduplication templates are defined in the function `ReduplicationFinder._is_valid_redspan()` at the bottom of `representations.py`. Phonological weight is defined in the `Construction.weight` property in `representations.py`. Both are currently specific to Māori; we will remove this hard-coding and provide a more user-friendly interface when we are able.  

## Requirements

- Install Morfessor 2.0 (https://github.com/aalto-speech/morfessor)
- Copy the .py files in this repo to the source location of Morfessor 2.0, overwriting source files (`baseline.py, cmd.py, io.py`, `evaluation.py`, `representations.py`)  
  > The source files are located in the package repository for the current python environment  

*Note: source files were edited in python 3.5 and may not work with earlier (or later) versions*

## Input

A text file, with one word per line, split into atoms (phonemes, syllables, etc.) by the space character

## Outputs

- An analyses text file, tab-delimited with a comment on the first line.  
  > Column 1 is word count in training data, column 2 is word, and column 3 is analysis (morphs split by `+`)

## Sample usage

Training a model with reduplication templates:  
```
python morfessor -t training_input.txt -S training_output.txt --atom-separator " " --strip-atom-sep --compound-separator "" --forcesplit "" --regexsplit "(?<=(\S)) (?=\1)" --randseed 123 --finish-threshold 0.00005 --progressbar --Lred-minbase-weight 2 --Rred-minbase-weight 4 --left-reduplication --right-reduplication --skippable-full-red --delay-nonedge-red --backup
```

Flags in this example:
- `-t <TRAIN_INPUT>` gives the path to the input training file
- `-S <TRAIN_OUTPUT>` gives the path to the output training file
- `--atom-separator <SEP>` gives the separator used to separate atoms in the input  
- `--strip-atom-sep` is a flag that removes the atom separator from output files  
  > Use this to facilitate manual inspection  
  > Avoid if wanting to save the segmentations so that they can be read back into Morfessor later  
- `--compound-separator <SEP>` gives the separator used to separate words on the same line in the input  
  > Set to `""` when there is only one word per line  
- `--forcesplit <CHARS>` forces each atom in the provided list of characters to be its own construction.  
  > The default is `-`, which means that any words will be forcibly split on hyphens. When there are no hyphens, it provides a speed boost to set it to nothing.  
- `--regexsplit <REGEXP>` forces a split whenever it encounters an atom separator boundary that matches the given regular expression.  
  > The default is `None`, in which case this is ignored.  
  > The provided value of `"(?<=(\S)) (?=\1)"` forces a split between two adjacent identical characters (that are separated by a space).  
- `--randseed 123` sets the random seed to 123, for reproducible results (since words are shuffled in each epoch)  
- `--finish-threshold <THRESHOLD>` sets the stopping threshold: training stops when the average per-word-token decrease in cost is less than or equal to the threshold  
  > The default is 0.005 (=1/200, i.e. a loss of 100 points over 20000 words)  
- `--progressbar` triggers the display of a progress bar when running  
- `--Lred-minbase-weight` and `--Rred-minbase-weight` represent the minimum allowable phonological weight of a base for reduplication, for left- and right-attaching reduplication, respectively.  
- `--left-reduplication` and `--right-reduplication` instruct Morfessor to check for reduplication attaching to the left and right, respectively.  
  > Only one flag can be provided if only one side is to be checked.  
  > If neither flag is provided, and `--full-red-only` is not provided either, Morfessor will revert to its original formulation (without reduplication templates)  
  > In highly complex words, something that is identified as left- or right-reduplication early on may be reanalyzed as full reduplication later, if the reduplicant is at least as long as the relevant minimum base and `--disable-full-red` is not provided  
- `--skippable-full-red` tells Morfessor that it's OK to leave a full reduplication base and reduplicant together when isolated from the rest of the word (provided `--enforce-full-red` is not provided). Thus, *pakipaki* will not automatically be analyzed as *<RED> + paki* and *whaka + omaoma* will not automatically be analyzed as *whaka + <RED> + oma*, though either may be.  
  > If this flag is not provided, then whenever an apparent full reduplicant-base is isolated but not found to have a good further segmentation, it is reanalyzed posthoc as full reduplication. This is different to `--enforce-full-red` because that does the reanalysis without even considering alternative segmentations.  
  > If this flag is not provided, when assessing the probability of different segmentations or segmenting unseen items with a trained model, the probability mass of segmentations that isolate an apparent full reduplicant-base combination but do not further segment it will be reallocated to corresponding segmentations that treat the combination as full reduplication.  
  > If `--enforce-full-red` is provided, this flag is treated as though it were provided too. Its effect is to prevent the reallocation of probability mass described in the above point. It has no effect on training Morfessor models because enforcing full reduplication implies that it cannot be skipped (skipping is lazy; enforcing is greedy).  
- `--delay-nonedge-red` instructs Morfessor to look for reduplication where a reduplicant is *not* at the end of the construction only when it can't find a suitable split in the construction.  
  > Without the flag, reduplication anywhere in the word is considered at every stage.
  > The idea is that this should encourage consistency across paradigms that a base with RED participates in, without allowing that base to be split up before reduplication can be considered.  
- `--backup` will back up before each reanalysis, so that if the reanalysis would cause cost to *increase*, it will not be applied.  

Information about other flags can be found in the Morfessor 2.0 documentation. In particular, the following are needed to test the models:  
- `-T <TEST_INPUT>` gives the path to the input test file  
- `-o <TEST_OUTPUT>` gives the path to the output test file  

In addition, the following flags are unique to this extension and relevant to the above:  
- `--default-red-attachment` controls which kind of reduplication is treated as the default in cases of full reduplication.  
  > The default value is `L`.  
- `--disable-full-red` forces Morfessor not to treat full reduplication as a valid form of reduplication.  
  > This means that reduplicants are always required to be smaller than their base, no matter their absolute size.  
  > The end result of full reduplication can still be accomplished by ordinary splits: they just won't be reanalyzed as giving rise to <RED>.  
  > This allows left and/or right reduplication to be considered independently of full reduplication.  
  > When this flag is provided, `--skippable-full-red` is treated as though it were provided as well.  
- `--full-red-only` instructs Morfessor to identify full reduplication only, i.e. without also considering left- or right-reduplication  
  > If neither `--left-reduplication` nor `--right-reduplication` are provided, Morfessor will identify full reduplication based on the side determined by `--default-red-attachment`.  
  > Providing `--full-red-only` does not *enforce* full reduplication; this parameter is independent of `--enforce-full-red`/`--skippable-full-red`.
- `--enforce-full-red` instructs Morfessor to always make a split between a reduplicant and base when considering a whole wordform / morph that looks like full reduplication (e.g. *pakipaki* is automatically analyzed as *<RED> + paki*). Thus, if an apparent full reduplicant-base combination is ever isolated from the rest of the word, it is necessarily split into full reduplication (e.g. if*whakaomaoma* ever has the prefix split off, giving *whaka + omaoma*, the analysis automatically becomes *whaka + <RED> + oma*).  
  > When assessing the probability of different segmentations or segmenting unseen items with a trained model, this option will discard potential segmentations that have splits at the outer edges of the reduplicant-base combination but either leave it intact or split it in a way other than full reduplication.  
- `--delay-red-checking` is like `--delay-nonedge-red`, except reduplication where a reduplicant is at the edge of a construction is *also* delayed; Morfessor looks for any kind of reduplication only when it can't find an alternative split, i.e. after all instances of affixation and compounding have been accounted for.  
  > Without the flag, reduplication is considered at every stage, everywhere in the word.  
  > After splitting is complete, the analysis is re-evaluated and anything that recapitulates reduplication is converted to reduplication (where recapitulating reduplication is sensitive to the order in which the splits are made).  
  > This favors paradigm-uniformity like `--delay-nonedge-red`, but also permits a base to be split up before reduplication, which can prevent the identification of reduplication  
- `--backup-delay <N>` will wait N epochs before storing a history of removed split locations, and before backing up in case of increased cost (if `--backup` is also provided).  
  > The default is 1.  
  > Without random initializations, there is nothing to be gained from storing or backing up in the first epoch.  
  > With random initializations, storing or backing up in the first epoch would reinforce the initial conditions and thus be counterproductive.  
  > To disable storing a history of removed split locations (as in the original Morfessor), set this to a large number.  
  

## Usage in the paper

Model with no reduplication templates:  
```
python morfessor -t training_input.txt -S training_output.txt -T test_input.txt -o test_output.txt --atom-separator " " --strip-atom-sep --compound-separator "" --forcesplit "" --regexsplit "(?<=(\S)) (?=\1)" --randseed 123 --finish-threshold 0.00005 --progressbar --Lred-minbase-weight 2 --Rred-minbase-weight 4 --delay-nonedge-red --backup
```

Model with full-reduplication template only:  
```
python morfessor -t training_input.txt -S training_output.txt -T test_input.txt -o test_output.txt --atom-separator " " --strip-atom-sep --compound-separator "" --forcesplit "" --regexsplit "(?<=(\S)) (?=\1)" --randseed 123 --finish-threshold 0.00005 --progressbar --Lred-minbase-weight 2 --Rred-minbase-weight 4 --delay-nonedge-red --backup --full-red-only --skippable-full-red
```

Model with left-reduplication template only:  
```
python morfessor -t training_input.txt -S training_output.txt -T test_input.txt -o test_output.txt --atom-separator " " --strip-atom-sep --compound-separator "" --forcesplit "" --regexsplit "(?<=(\S)) (?=\1)" --randseed 123 --finish-threshold 0.00005 --progressbar --Lred-minbase-weight 2 --Rred-minbase-weight 4 --delay-nonedge-red --backup --left-reduplication --disable-full-red
```

Model with right reduplication template only:  
```
python morfessor -t training_input.txt -S training_output.txt -T test_input.txt -o test_output.txt --atom-separator " " --strip-atom-sep --compound-separator "" --forcesplit "" --regexsplit "(?<=(\S)) (?=\1)" --randseed 123 --finish-threshold 0.00005 --progressbar --Lred-minbase-weight 2 --Rred-minbase-weight 4 --delay-nonedge-red --backup --right-reduplication --disable-full-red
```

Model with full- and left-reduplication templates:  
```
python morfessor -t training_input.txt -S training_output.txt -T test_input.txt -o test_output.txt --atom-separator " " --strip-atom-sep --compound-separator "" --forcesplit "" --regexsplit "(?<=(\S)) (?=\1)" --randseed 123 --finish-threshold 0.00005 --progressbar --Lred-minbase-weight 2 --Rred-minbase-weight 4 --delay-nonedge-red --backup --left-reduplication --skippable-full-red
```

Model with full- and right-reduplication templates:  
```
python morfessor -t training_input.txt -S training_output.txt -T test_input.txt -o test_output.txt --atom-separator " " --strip-atom-sep --compound-separator "" --forcesplit "" --regexsplit "(?<=(\S)) (?=\1)" --randseed 123 --finish-threshold 0.00005 --progressbar --Lred-minbase-weight 2 --Rred-minbase-weight 2 --delay-nonedge-red --backup --right-reduplication --skippable-full-red
```

Model with left- and right-reduplication templates:  
```
python morfessor -t training_input.txt -S training_output.txt -T test_input.txt -o test_output.txt --atom-separator " " --strip-atom-sep --compound-separator "" --forcesplit "" --regexsplit "(?<=(\S)) (?=\1)" --randseed 123 --finish-threshold 0.00005 --progressbar --Lred-minbase-weight 2 --Rred-minbase-weight 4 --delay-nonedge-red --backup --left-reduplication --right-reduplication --disable-full-red
```

Model with all reduplication templates:  
```
python morfessor -t training_input.txt -S training_output.txt -T test_input.txt -o test_output.txt --atom-separator " " --strip-atom-sep --compound-separator "" --forcesplit "" --regexsplit "(?<=(\S)) (?=\1)" --randseed 123 --finish-threshold 0.00005 --progressbar --Lred-minbase-weight 2 --Rred-minbase-weight 4 --delay-nonedge-red --backup --left-reduplication --right-reduplication --skippable-full-red
```