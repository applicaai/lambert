import torch
from transformers.models.roberta.modeling_roberta import RobertaModel

from lambert.model import LambertModel

BATCH_SIZE = 4
SEQUENCE_LENGTH = 32

roberta = RobertaModel.from_pretrained('roberta-base')
lambert = LambertModel(roberta)

input_ids = torch.randint(0, 100, (BATCH_SIZE, SEQUENCE_LENGTH))
bboxes = torch.rand((BATCH_SIZE, SEQUENCE_LENGTH, 4))

lambert_output = lambert(input_ids=input_ids, bboxes=bboxes)
lambert_encoding = lambert_output.last_hidden_state

assert lambert_encoding.shape == (BATCH_SIZE, SEQUENCE_LENGTH, roberta.config.hidden_size)
assert lambert_encoding.dtype == torch.float
