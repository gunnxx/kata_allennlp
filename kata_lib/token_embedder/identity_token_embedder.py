import torch

from allennlp.modules.token_embedders.token_embedder import TokenEmbedder


@TokenEmbedder.register("identity")
class IdentityTokenEmbedder(TokenEmbedder):
    """
    ``IdentityTokenEmbedder`` just passes the input to output . It is needed when
    embedding is a deterministic processes and these processes is already handled
    by ``TokenIndexer``
    """
    def __init__(self) -> None:
        super(IdentityTokenEmbedder, self).__init__()

    def forward(self, token_characters: torch.Tensor) -> torch.Tensor:
        return token_characters