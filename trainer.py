from transformers import Trainer

from torch import nn


class CustomTrainer(Trainer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.kl_loss_func = nn.KLDivLoss(reduction="batchmean")

    def compute_loss(self, model, inputs, return_outputs=False, return_embeddings=False):
        target_p = inputs["labels"]
        outputs = model(inputs["input_ids"], attention_mask=inputs["attention_mask"], output_hidden_states=return_embeddings)
        logits = outputs[0]
        loss = self.kl_loss_func(logits.log_softmax(dim=-1), target_p)  # todo add case where target p isnt a dist (add softmax)

        if return_embeddings:
            last_layer_hidden_state = outputs['hidden_states'][-1]

            # Multiplying with the attention mask to zero out the pads' embeddings
            last_layer_hidden_state = (inputs['attention_mask'] * last_layer_hidden_state.permute([2, 0, 1])).permute([1, 2, 0])

            # Dividing each input with its length to get the mean
            lengths = inputs['attention_mask'].sum(dim=1)
            embeddings = (last_layer_hidden_state.sum(dim=1).t() / lengths).t()
            return loss, outputs, embeddings

        if return_outputs:
            return loss, outputs

        return loss