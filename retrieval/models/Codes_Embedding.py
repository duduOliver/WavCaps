#!/usr/bin/env python3
# coding: utf-8
# @Author  : Xinhao Mei @CVSSP, University of Surrey
# @E-mail  : x.mei@surrey.ac.uk

import torch
import torch.nn as nn
from transformers import (
    BertModel,
    BertTokenizer,
    GPT2Model,
    GPT2Tokenizer,
    RobertaModel,
    RobertaTokenizer,
    DistilBertModel,
    DistilBertTokenizer,
    CLIPTokenizer,
    CLIPTextModel,
)
from transformers import AutoConfig


MODELS = {
    'openai/clip-vit-base-patch32': (CLIPTextModel, CLIPTokenizer, 512),
    'prajjwal1/bert-tiny': (BertModel, BertTokenizer, 128),
    'prajjwal1/bert-mini': (BertModel, BertTokenizer, 256),
    'prajjwal1/bert-small': (BertModel, BertTokenizer, 512),
    'prajjwal1/bert-medium': (BertModel, BertTokenizer, 512),
    'gpt2': (GPT2Model, GPT2Tokenizer, 768),
    'distilgpt2': (GPT2Model, GPT2Tokenizer, 768),
    'bert-base-uncased': (BertModel, BertTokenizer, 768),
    'bert-large-uncased': (BertModel, BertTokenizer, 1024),
    'roberta-base': (RobertaModel, RobertaTokenizer, 768),
    'roberta-large': (RobertaModel, RobertaTokenizer, 1024),
    'distilbert-base-uncased': (DistilBertModel, DistilBertTokenizer, 768),
    "distilroberta-base": (RobertaModel, RobertaTokenizer, 768),
}


class LongCodesEmbedder(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.tokenizer = MODELS[config["text_encoder_args"]["type"]][1].from_pretrained(
            config["text_encoder_args"]["type"])
        self.text_encoder = MODELS[config["text_encoder_args"]["type"]][0].from_pretrained(
            config["text_encoder_args"]["type"],
            add_pooling_layer=False)

        # Load the model configuration
        self.model_config = AutoConfig.from_pretrained(config["text_encoder_args"]["type"])
        # Get the maximum number of tokens the model can handle
        self.max_length = self.model_config.max_position_embeddings
        print("The maximum input size for ", {MODELS[config["text_encoder_args"]["type"]][0]}, " is ", {self.max_length}, " tokens.")
        # self.max_length = 512
        
        if config["audio_encoder_args"]["freeze"]:
            for name, param in self.text_encoder.named_parameters():
                param.requires_grad = False

        self.text_width = MODELS[config["text_encoder_args"]["type"]][-1]

    @property
    def device(self):
        return list(self.parameters())[0].device

    def forward(self, codes):
        # Tokenize texts without truncation
        # tokens = self.tokenizer(texts, return_tensors='pt', truncation=False, padding=True)
        input_ids = codes # tokens['input_ids']
        # attention_mask = tokens['attention_mask']

        batch_size, seq_len = input_ids.size()

        # Prepare list to collect chunk outputs
        all_chunks = []

        for batch_idx in range(batch_size):
            input_ids_batch = input_ids[batch_idx]
            # attention_mask_batch = attention_mask[batch_idx]

            input_chunks = []
            attention_chunks = []

            # Split input_ids into chunks of max_length with overlap
            chunk_size = self.max_length #- 2  # account for [CLS] and [SEP] tokens
            overlap_size = 50
            
            for i in range(0, len(input_ids_batch), chunk_size - overlap_size):
                chunk_ids = input_ids_batch[i:i + chunk_size]
                # chunk_mask = attention_mask_batch[i:i + chunk_size]
                chunk_mask = torch.tensor([1] * len(chunk_ids)).to(self.device)

                # Add [CLS] and [SEP] tokens
                # chunk_ids = torch.cat([torch.tensor([self.tokenizer.cls_token_id]), chunk_ids, torch.tensor([self.tokenizer.sep_token_id])])
                # chunk_mask = torch.cat([torch.tensor([1]), chunk_mask, torch.tensor([1])])

                # Pad chunks to max_length
                padding_length = self.max_length - chunk_ids.size(0)
                if padding_length > 0:
                    chunk_ids = torch.cat([chunk_ids, torch.tensor([0] * padding_length).to(self.device)])
                    chunk_mask = torch.cat([chunk_mask, torch.tensor([0] * padding_length).to(self.device)])

                input_chunks.append(chunk_ids.unsqueeze(0))
                attention_chunks.append(chunk_mask.unsqueeze(0))

            # Stack chunks into a batch
            input_chunks = torch.cat(input_chunks, dim=0).to(self.device)
            # print(input_chunks.shape)
            attention_chunks = torch.cat(attention_chunks, dim=0).to(self.device)
            # print(attention_chunks.shape)

            # Process each chunk and collect the hidden states
            outputs = []
            for i in range(input_chunks.size(0)):
                chunk_output = self.text_encoder(input_ids=input_chunks[i:i+1], attention_mask=attention_chunks[i:i+1])[0]
                outputs.append(chunk_output)

            # Concatenate all chunk outputs along the sequence dimension
            outputs = torch.cat(outputs, dim=1)
            
            all_chunks.append(outputs.unsqueeze(0))  # Keep batch dimension

        # Stack all batch outputs
        all_chunks = torch.cat(all_chunks, dim=0)

        # Optionally: apply some combination method here (e.g., mean pooling, attention)
        # combined_output = torch.mean(all_chunks, dim=1)  # example of mean pooling
        # Average Pooling
        pooled_embeddings = torch.mean(all_chunks, dim=2, keepdim=True)  # Shape: [1, 1, 768]
        # pooled_embeddings = pooled_embeddings.repeat(1, 1, 4, 1)            # Shape: [1, 4, 768]

        return pooled_embeddings
