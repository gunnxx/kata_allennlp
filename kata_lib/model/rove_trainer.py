from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from allennlp.nn import util
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules import TextFieldEmbedder, Seq2SeqEncoder
from allennlp.models.model import Model


@Model.register("rove_trainer")
class Rove(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder,
                 projection_dim: int = None,
                 window_size: int = 3,
                 neg_sample: int = 5) -> None:
        super().__init__(vocab):

        self.encoder = encoder
        self.feedforward = nn.Linear(self.encoder.get_output_dim(), projection_dim) if projection_dim else None

        self.window_size = window_size
        self.neg_sample  = neg_sample

        self.metrics = {}   # will be filled by "loss"

    def forward(self,
                sentence: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        mask = util.get_text_field_mask(sentence)
        out  = self.encoder(ins, mask)
        if self.feedforward: out = self.feedforward(out)

        loss = torch.Tensor(0)

        # compute loss to minimize distance between center word and
        # context words based on params.context_window size
        for i in range(1, self.window_size + 1):
            loss += self.cos_embedding_loss(out[:, i:, :], out[:, :-i, :])

        # negative-samples loss
        # negative-samples are taken by randomly rolling batch
        for i in range(self.neg_sample):
            loss += self.cos_embedding_loss(out, torch.roll(out, randrange(32), 0), True)

        self.metrics["loss"] = loss.item()

        return {"loss": loss}

    def get_metrics(self) -> Dict[str, float]:
        return self.metrics

    def cos_embedding_loss(output: torch.Tensor,
                           target: torch.Tensor,
                           neg_samples: bool = False) -> torch.Tensor:
        '''Compute cosine similarity loss between output and target

        Args:
            output of shape (batch_size, num_tokens, embedding_dim)
            target of shape (batch_size, num_tokens, embedding_dim)
        '''
        if neg_samples:
            return torch.sum(torch.exp(F.relu(F.cosine_similarity(output, target, dim=2))))
        return torch.sum(torch.exp(1 - F.cosine_similarity(output, target, dim=2)))