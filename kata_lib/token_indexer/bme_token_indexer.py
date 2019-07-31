from typing import Dict, List
from overrides import overrides

import itertools
import warnings

import torch
import torch.nn.functional as F

from allennlp.common.checks import ConfigurationError
from allennlp.common.util import pad_sequence_to_length
from allennlp.data.tokenizers.token import Token
from allennlp.data.token_indexers.token_indexer import TokenIndexer
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.tokenizers.character_tokenizer import CharacterTokenizer


@TokenIndexer.register("bme")
class BMETokenIndexer(TokenIndexer[List[int]]):
    """
    This :class:`TokenIndexer` represents tokens as lists of transformed character indices based on BME.
    BME (Beginning-Middle-End) is described in http://noisy-text.github.io/2018/pdf/W-NUT20188.pdf.

    This class inherits :class:`~allennlp.data.token_indexers.token_characters_indexer.TokenCharactersIndexer`
    due to some similarity in implementation. Note that ``min_padding_length`` argument for parent class
    is always set to 0.

    Cancel to inherit :class:`~allennlp.data.token_indexers.token_characters_indexer.TokenCharactersIndexer`
    due to `as_padded_tensor` can't be overridden. Trying to implement from scratch but similar to
    `TokenCharactersIndexer`
    Parameters
    ----------
    namespace : ``str``, optional (default=``token_characters``)
        We will use this namespace in the :class:`Vocabulary` to map the characters in each token
        to indices.
    character_tokenizer : ``CharacterTokenizer``, optional (default=``CharacterTokenizer()``)
        We use a :class:`CharacterTokenizer` to handle splitting tokens into characters, as it has
        options for byte encoding and other things.  The default here is to instantiate a
        ``CharacterTokenizer`` with its default parameters, which uses unicode characters and
        retains casing.
    start_tokens : ``List[str]``, optional (default=``None``)
        These are prepended to the tokens provided to ``tokens_to_indices``.
    end_tokens : ``List[str]``, optional (default=``None``)
        These are appended to the tokens provided to ``tokens_to_indices``.
    token_min_padding_length : ``int``, optional (default=``0``)
        See :class:`TokenIndexer`.
    begin_size: ``int``, optional(default=``3``)
        Size of B from BME. Please refer to http://noisy-text.github.io/2018/pdf/W-NUT20188.pdf
    end_size: ``int``, optional(default=```3``)
        Size of E from BME. Please refer to http://noisy-text.github.io/2018/pdf/W-NUT20188.pdf
    """
    def __init__(self,
                 namespace: str = 'bme_token_characters',
                 character_tokenizer: CharacterTokenizer = CharacterTokenizer(),
                 start_tokens: List[str] = None,
                 end_tokens: List[str] = None,
                 token_min_padding_length: int = 0,
                 begin_size: int = 3,
                 end_size: int = 3) -> None:
        super().__init__(token_min_padding_length)
        
        major, minor, patch = map(int, torch.__version__.split('.'))
        torch_version = major + 0.1*minor

        if torch_version < 1.1:
            raise Exception("BMETokenIndexer requires pytorch version >= 1.1 because it provides \
                torch.nn.functional.one_hot. Your version is {}".format(torch.__version__))

        self._namespace = namespace
        self._character_tokenizer = character_tokenizer

        self._start_tokens = [Token(st) for st in (start_tokens or [])]
        self._end_tokens = [Token(et) for et in (end_tokens or [])]

        self._begin_size = begin_size
        self._end_size = end_size

    @overrides
    def count_vocab_items(self, token: Token, counter: Dict[str, Dict[str, int]]):
        if token.text is None:
            raise ConfigurationError('BMETokenIndexer needs a tokenizer that retains text')
        for character in self._character_tokenizer.tokenize(token.text):
            # If `text_id` is set on the character token (e.g., if we're using byte encoding), we
            # will not be using the vocab for this character.
            if getattr(character, 'text_id', None) is None:
                counter[self._namespace][character.text] += 1

    @overrides
    def tokens_to_indices(self,
                          tokens: List[Token],
                          vocabulary: Vocabulary,
                          index_name: str) -> Dict[str, List[List[int]]]:
        vocab_size = vocabulary.get_vocab_size(self._namespace)

        # Initial steps are exactly the same as super().tokens_to_indices()
        indices: List[List[int]] = []

        for token in itertools.chain(self._start_tokens, tokens, self._end_tokens):
            token_indices: List[int] = []

            if token.text is None:
                raise ConfigurationError('BMETokenIndexer needs a tokenizer that retains text')

            for character in self._character_tokenizer.tokenize(token.text):
                if getattr(character, 'text_id', None) is not None:
                    # `text_id` being set on the token means that we aren't using the vocab, we just
                    # use this id instead.
                    index = character.text_id
                else:
                    index = vocabulary.get_token_index(character.text, self._namespace)

                token_indices.append(index)

            # Generating BME (steps that are different from super().tokens_to_indices())
            B = F.one_hot(torch.tensor(self._pad(token_indices[:self._begin_size], self._begin_size, True)), num_classes=vocab_size).reshape(-1)
            M = F.one_hot(torch.tensor([0] if len(token_indices) is 0 else token_indices), num_classes=vocab_size).sum(0)
            E = F.one_hot(torch.tensor(self._pad(token_indices[-self._end_size:], self._end_size)), num_classes=vocab_size).reshape(-1)

            indices.append(torch.cat((B, M, E)).tolist())
        return {index_name: indices}

    @overrides
    def get_padding_lengths(self, tokens: List[int]) -> Dict[str, int]:
        return {}

    @overrides
    #def as_padded_tensor(self,
    def pad_token_sequence(self,
                           tokens: Dict[str, List[List[int]]],
                           desired_num_tokens: Dict[str, int],
                           padding_lengths: Dict[str, int]) -> Dict[str, torch.Tensor]:
        return {key: torch.LongTensor(pad_sequence_to_length(val, desired_num_tokens[key], lambda: [0]*(len(tokens[key][0]))))
                for key, val in tokens.items()}

    # auxillary function to pad list whose length <= N
    # with padding_index (i.e. 0) so the list has length == N
    def _pad(self, arr: List[int], N: int, prepend: bool = False) -> List[int]:
        if prepend:
            return [0]*(N-len(arr)) + arr
        return arr + [0]*(N-len(arr))