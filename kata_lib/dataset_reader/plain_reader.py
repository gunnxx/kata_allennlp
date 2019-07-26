from typing import Dict, List
from overrides import overrides

from allennlp.data.tokenizers.token import Token
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.token_indexers.token_indexer import TokenIndexer
from allennlp.data.fields import TextField
from allennlp.data.instance import Instance

@DatasetReader.register("plain_reader")
class PlainReader(DatasetReader):
    def __init__(self,
                 token_indexers: Dict[str, TokenIndexer]) -> None:
        super().__init__(lazy=False)
        self.token_indexers = token_indexers

    def text_to_instance(self,
                         text: List[Token]) -> Instance:
        sentence = TextField(text, self.token_indexers)
        return Instance({"sentence": sentence})

    @overrides
    def _read(self, file_path: str):
        file_path = cached_path(file_path)
        with open(file_path, 'r') as f:
            for line in f:
                yield self.text_to_instance(list(map(Token, line.strip().split())))
