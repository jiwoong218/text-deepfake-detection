
import torch
from torch.nn import functional as F
from typing import Optional, Union
from transformers import ElectraForSequenceClassification
from transformers.modeling_outputs import SequenceClassifierOutput
from superloss import HardFirstSuperLoss

class SuperLossElectraForSequenceClassification(ElectraForSequenceClassification):
    hfs = HardFirstSuperLoss(
        lam = 1.65,
        tau = .7,
        mom = .1
    )
    
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple[torch.Tensor], SequenceClassifierOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        discriminator_hidden_states = self.electra(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        sequence_output = discriminator_hidden_states[0]
        
        loss = None
        logits = None

        if labels is not None:
            logits = self.classifier(sequence_output)
            
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"
            
            if self.config.problem_type == "regression":
                raise NotImplementedError()
            elif self.config.problem_type == "single_label_classification":
                base_loss = torch.nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1),
                    reduction='none'
                )
                spl, sig = self.hfs(base_loss)
                loss = spl.mean()
            elif self.config.problem_type == "multi_label_classification":
                raise NotImplementedError()
        else:
            logits = self.classifier(sequence_output)

        if not return_dict:
            output = (logits,) + discriminator_hidden_states[1:]
            return ((loss,) + output) if loss is not None else output
        
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=discriminator_hidden_states.hidden_states,
            attentions=discriminator_hidden_states.attentions,
        )
