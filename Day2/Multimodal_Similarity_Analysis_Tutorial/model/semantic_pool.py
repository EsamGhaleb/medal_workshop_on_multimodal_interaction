from pytorch_lightning import LightningModule
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from transformers import RobertaTokenizer, RobertaForSequenceClassification

from model.attentive_pooler import AttentivePooler


class BertjePoolingModule(LightningModule):
    def __init__(self, freeze_bertje=True, use_attentive_pooling=True, use_robbert=False, dont_pool=False):
        super().__init__()
        # Load the tokenizer and model
        if use_robbert:
            self.tokenizer = RobertaTokenizer.from_pretrained("DTAI-KULeuven/robbert-2023-dutch-large")
            self.bertje_model = RobertaForSequenceClassification.from_pretrained("DTAI-KULeuven/robbert-2023-dutch-large")
        else:
            self.tokenizer = AutoTokenizer.from_pretrained("jegormeister/bert-base-dutch-cased-snli")
            self.bertje_model = AutoModel.from_pretrained("jegormeister/bert-base-dutch-cased-snli")
        if freeze_bertje:
            for param in self.bertje_model.parameters():
                param.requires_grad = False
            self.bertje_model.eval()
        self.attentive_pooling = use_attentive_pooling
        self.use_robbert = use_robbert
        if self.attentive_pooling:
            self.attentive_pooler = AttentivePooler(
                num_queries=1,
                embed_dim=self.bertje_model.config.hidden_size,
                num_heads=self.bertje_model.config.num_attention_heads,
                mlp_ratio=4.0,
                depth=1,
                norm_layer=nn.LayerNorm,
                init_std=0.02,
                qkv_bias=True,
                complete_block=True
            )

    def forward(self, sentences):
        # get model device
        outputs = {}
        device = next(self.parameters()).device
        inputs = self.tokenizer(
            sentences,
            return_tensors="pt",
            truncation=False,
            padding=True,
            max_length=512
        ).to(device)
        if not self.use_robbert:
            embeddings = self.bertje_model(**inputs)

            # Use the last hidden layer as the embedding, or average across layers if desired
            # Here, we're taking the last hidden layer's embeddings
            outputs["local"] = embeddings.last_hidden_state
            outputs['attention_mask'] = inputs['attention_mask']  
            if self.attentive_pooling:
                outputs["global"] = self.attentive_pooler(embeddings.last_hidden_state).squeeze(1)
            # Alternatively, you can average the embeddings across all tokens
            else:
                outputs["global"] = embeddings['pooler_output']# torch.mean(embeddings.last_hidden_state, dim=1)
        else:
            embeddings = self.bertje_model(**inputs, output_hidden_states=True)
            # Extract the hidden states (a list of tensors for each layer)
            hidden_states = embeddings.hidden_states

            # Use the last hidden layer as the embedding, or average across layers if desired
            # Here, we're taking the last hidden layer's embeddings
            last_hidden_layer = hidden_states[-1]  # shape: (batch_size, sequence_length, hidden_size)
            outputs["local"] = last_hidden_layer
            outputs['attention_mask'] = inputs['attention_mask'] 
            if self.attentive_pooling:
                outputs["global"] = self.attentive_pooler(last_hidden_layer).squeeze(1)
            else:
                # Alternatively, you can average the embeddings across all tokens
                outputs["global"] = embeddings['pooler_output'] # last_hidden_layer.mean(dim=1)  # shape: (batch_size, hidden_size)

        return outputs
