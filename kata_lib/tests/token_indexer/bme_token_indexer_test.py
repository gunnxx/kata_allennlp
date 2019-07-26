from allennlp.common.testing import AllenNlpTestCase
from allennlp.data import Token, Vocabulary

from kata_lib.token_indexer import BMETokenIndexer

class BMETokenIndexerTest(AllenNlpTestCase):
    def test_tokens_to_indices(self):
        vocab = Vocabulary()
        vocab.add_token_to_namespace("k", namespace='bme') # 2
        vocab.add_token_to_namespace("c", namespace='bme') # 3
        vocab.add_token_to_namespace("s", namespace='bme') # 4

        indexer = BMETokenIndexer("bme", begin_size=2, end_size=2)
        indices = indexer.tokens_to_indices([Token("kicks")], vocab, "char")
        '''
        B: k     -> [0, 0, 1, 0, 0]
           i     -> [0, 1, 0, 0, 0] #unknown
        M: kicks -> [0, 1, 2, 1, 1]
        E: k     -> [0, 0, 1, 0, 0]
           s     -> [0, 0, 0, 0, 1]
        '''
        assert indices == {"char": [[0, 0, 1, 0, 0,\
                                     0, 1, 0, 0, 0,\
                                     0, 1, 2, 1, 1,\
                                     0, 0, 1, 0, 0,\
                                     0, 0, 0, 0, 1]]}

        indexer = BMETokenIndexer("bme")
        indices = indexer.tokens_to_indices([Token("is")], vocab, "char")
        '''
        by default, begin_size and end_size both are 3
        B: pad -> [1, 0, 0, 0, 0]
           i   -> [0, 1, 0, 0, 0] #unknown
           s   -> [0, 0, 0, 0, 1]
        M: is  -> [0, 1, 0, 0, 1]
        B: i   -> [0, 1, 0, 0, 0] #unknown
           s   -> [0, 0, 0, 0, 1]
           pad -> [1, 0, 0, 0, 0]
        '''
        assert indices == {"char": [[1, 0, 0, 0, 0,\
                                     0, 1, 0, 0, 0,\
                                     0, 0, 0, 0, 1,\
                                     0, 1, 0, 0, 1,\
                                     0, 1, 0, 0, 0,\
                                     0, 0, 0, 0, 1,\
                                     1, 0, 0, 0, 0]]}

        indexer = BMETokenIndexer("bme", begin_size=2, end_size=2)
        indices = indexer.tokens_to_indices([Token("kicks"), Token("is")], vocab, "char")

        assert indices == {"char": [[0, 0, 1, 0, 0,\
                                     0, 1, 0, 0, 0,\
                                     0, 1, 2, 1, 1,\
                                     0, 0, 1, 0, 0,\
                                     0, 0, 0, 0, 1],
                                    [0, 1, 0, 0, 0,\
                                     0, 0, 0, 0, 1,\
                                     0, 1, 0, 0, 1,\
                                     0, 1, 0, 0, 0,\
                                     0, 0, 0, 0, 1]]}

    def test_start_and_end_tokens(self):
        vocab = Vocabulary()
        vocab.add_token_to_namespace("i", namespace='bme') # 2
        vocab.add_token_to_namespace("s", namespace='bme') # 3

        indexer = BMETokenIndexer("bme", start_tokens=["<s>"], end_tokens=["</s>"],
                                  begin_size=1, end_size=1)
        indices = indexer.tokens_to_indices([Token("is")], vocab, "char")

        assert indices == {"char": [[0, 1, 0, 0, 0, 2, 0, 1, 0, 1, 0, 0],   # <s>
                                    [0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1],   # is
                                    [0, 1, 0, 0, 0, 3, 0, 1, 0, 1, 0, 0]]}  # </s>

    def test_as_padded_tensor(self):
        indexer = BMETokenIndexer("bme")
        padded_tokens = indexer.as_padded_tensor({'key': [[0, 0, 1, 0, 0],
                                                          [0, 1, 0, 0, 0]]},
                                                 {'key': 4},
                                                 {})
        assert padded_tokens['key'].tolist() == [[0, 0, 1, 0, 0],
                                                 [0, 1, 0, 0, 0],
                                                 [0, 0, 0, 0, 0],
                                                 [0, 0, 0, 0, 0]]