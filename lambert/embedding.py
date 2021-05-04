from transformers.models.roberta.modeling_roberta import RobertaEmbeddings
import torch
from torch import nn, Tensor


class LayoutEmbeddings(nn.Module):
    """
    Layout embeddings described in the paper 'LAMBERT: Layout-Aware Language Modeling for information extraction'.

    The scaling factors (defined in the paper) form a geometric sequence from 1 to `base` with `steps` elements.
    The dimension of resulting embeddings is 8 * steps.

    Args:
        base (int): the last scaling factor in the geometric sequence beginning with 1
        steps (int): length of the sequence of scaling factors
    """

    def __init__(self, base: int = 500, steps: int = 96) -> None:
        super().__init__()
        self.register_buffer('factors', torch.logspace(0, 1, steps, base), persistent=False)

    def forward(self, bboxes: Tensor) -> Tensor:
        """
        Compute the embeddings

        Args:
            bboxes (Tensor): tensor of bounding boxes, shape (..., 4)

        Returns:
            Tensor: layout embeddings of the given bounding boxes
        """
        thetas = bboxes.unsqueeze(-1) * self.factors
        return torch.cat([torch.sin(thetas), torch.cos(thetas)], dim=3).reshape(*bboxes.shape[:-1], -1)


class EmbeddingWrapper(nn.Module):
    """
    Wrapper over RobertaEmbeddings, combining it with LayoutEmbeddings, as described in the paper.

    Args:
        roberta_embeddings (RobertaEmbeddings): an instance of RobertaEmbeddings
        base (int): `base` parameter of LayoutEmbeddings
    """

    def __init__(self, roberta_embeddings: RobertaEmbeddings, base: int = 500) -> None:
        super().__init__()
        embedding_dim = roberta_embeddings.word_embeddings.embedding_dim
        steps = embedding_dim // 8
        self.roberta_embs = roberta_embeddings
        self.layout_embs = LayoutEmbeddings(base, steps)
        self.adapter_layer = nn.Linear(8 * steps, embedding_dim)

    def forward(self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None,
                past_key_values_length=0, bboxes=None):
        """
        Compute the combined embeddings.

        Args:
            input_ids: see `RobertaEmbeddings`
            token_type_ids: see `RobertaEmbeddings`
            position_ids: see `RobertaEmbeddings`
            inputs_embeds: see `RobertaEmbeddings`
            past_key_values_length: see `RobertaEmbeddings`
            bboxes: tensor of bounding boxes, shape (batch_size, sequence_length, 4)

        Returns:
            Tensor of computed embeddings
        """
        embs = self.roberta_embs(input_ids, token_type_ids, position_ids, inputs_embeds, past_key_values_length)
        if bboxes is not None:
            layout_embs = self.layout_embs(bboxes)
            embs += self.adapter_layer(layout_embs)
        return embs
