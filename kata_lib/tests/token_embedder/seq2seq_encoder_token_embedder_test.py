import torch

from allennlp.common.testing import AllenNlpTestCase
from allennlp.modules.seq2seq_encoders import PytorchSeq2SeqWrapper

from kata_lib.token_embedder import Seq2SeqEncoderTokenEmbedder

class Seq2SeqEncoderTokenEmbedderTest(AllenNlpTestCase):
    def test_get_output_dim(self):
        input_dim = 10
        hidden_dim = 15

        lstm = PytorchSeq2SeqWrapper(torch.nn.LSTM(input_dim, hidden_dim, bidirectional=True, batch_first=True))
        embedder = Seq2SeqEncoderTokenEmbedder(lstm)
        assert embedder.get_output_dim() == hidden_dim*2

        lstm = PytorchSeq2SeqWrapper(torch.nn.LSTM(input_dim, hidden_dim, bidirectional=False, batch_first=True))
        embedder = Seq2SeqEncoderTokenEmbedder(lstm)
        assert embedder.get_output_dim() == hidden_dim

        lstm = PytorchSeq2SeqWrapper(torch.nn.LSTM(input_dim, hidden_dim, bidirectional=True, batch_first=True))
        embedder = Seq2SeqEncoderTokenEmbedder(lstm, 100)
        assert embedder.get_output_dim() == 100

    # checking output shape
    def test_forward(self):
        batch_size = 4
        seq_len    = 3
        feature    = 3
        hidden_dim = 20

        ins = torch.Tensor([[[1, 0, 0], [0, 0, 1], [0, 0, 0]],
                            [[1, 1, 1], [0, 0, 0], [0, 0, 0]],
                            [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                            [[1, 1, 1], [1, 1, 1], [1, 1, 1]]])

        lstm = PytorchSeq2SeqWrapper(torch.nn.LSTM(feature, hidden_dim, bidirectional=True, batch_first=True))
        embedder = Seq2SeqEncoderTokenEmbedder(lstm)
        assert embedder(ins).shape == torch.Size([batch_size, seq_len, hidden_dim*2])

        lstm = PytorchSeq2SeqWrapper(torch.nn.LSTM(feature, hidden_dim, bidirectional=False, batch_first=True))
        embedder = Seq2SeqEncoderTokenEmbedder(lstm)
        assert embedder(ins).shape == torch.Size([batch_size, seq_len, hidden_dim])

        lstm = PytorchSeq2SeqWrapper(torch.nn.LSTM(feature, hidden_dim, bidirectional=True, batch_first=True))
        embedder = Seq2SeqEncoderTokenEmbedder(lstm, 100)
        assert embedder(ins).shape == torch.Size([batch_size, seq_len, 100])