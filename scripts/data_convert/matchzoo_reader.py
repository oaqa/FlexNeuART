# A base class that provides a MatchZoo reader from white-space tokenized text
import pandas as pd

from matchzoo.preprocessors.basic_preprocessor import BasicPreprocessor

class WhiteSpaceTokenize:
    """Process unit for text tokenization."""

    def transform(self, input_: str) -> list:
        """
        Process input data from raw terms to list of tokens.

        :param input_: raw textual input.

        :return tokens: tokenized tokens as a list.
        """
        return input_.strip().split()

class WhiteSpacePreprocessor(BasicPreprocessor):
    def _default_units(cls) -> list:
        """Prepare needed process units."""
        return [ WhiteSpaceTokenize() ]

def readWhiteSpacedMatchZooData(fileName):
  dtf = pd.read_csv(fileName, sep=',').astype(str)
  dtf['label'] = dtf['label'].astype(int)
  return dtf
  
