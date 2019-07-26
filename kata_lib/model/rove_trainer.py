@Model.register("rove_trainer")
class RoveTrainer(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 window_size: int,
                 neg_sample: int) -> None:
        super().__init__(vocab):

        self.window_size = window_size
        self.neg_sample  = neg_sample

        self.metrics = {}   # will be filled by "loss"

    def forward(self,
                sentence: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        loss = torch.Tensor(0)

        # take first value of the dictionary i.e. torch.Tensor
        # looks ugly but, yeah, works
        ins = list(sentence.values())[0]

        # compute loss to minimize distance between center word and
        # context words based on params.context_window size
        for i in range(1, self.window_size + 1):
            loss += self.cos_embedding_loss(ins[:, i:, :], ins[:, :-i, :])

        # negative-samples loss
        # negative-samples are taken by randomly rolling batch
        for i in range(self.neg_sample):
            loss += self.cos_embedding_loss(ins, torch.roll(ins, randrange(), 0), True)

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
