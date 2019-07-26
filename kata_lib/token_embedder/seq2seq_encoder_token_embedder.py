import torch
import torch.nn as nn

from allennlp.nn import util
from allennlp.modules import Seq2SeqEncoder
from allennlp.modules.token_embedders.token_embedder import TokenEmbedder


@TokenEmbedder.register("seq2seq_encoder")
class Seq2SeqEncoderTokenEmbedder(TokenEmbedder):
    """
    ``Seq2SeqEncoderTokenEmbedder`` takes input tensor of shape (batch_size, num_tokens, embedding_dim).
    """
    def __init__(self,
                 encoder: Seq2SeqEncoder,
                 projection_dim: int = None) -> None:
        super(Seq2SeqEncoderTokenEmbedder, self).__init__()

        self.encoder = encoder
        self.feedforward = nn.Linear(self.encoder.get_output_dim(), projection_dim) if projection_dim else None

        self.output_dim = projection_dim if projection_dim else self.encoder.get_output_dim()

    def get_output_dim(self) -> int:
        return self.output_dim

    def forward(self, token_characters: torch.Tensor) -> torch.Tensor:
        mask = util.get_text_field_mask({'_': token_characters})
        out = self.encoder(token_characters, mask)
        if self.feedforward: out = self.feedforward(out)
        return out