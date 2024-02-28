#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

from transformers import AutoConfig, AutoModelForCausalLM
import torch.nn.functional as F
from llava.transformers.modeling_outputs import CausalLMOutputWithPast

from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM
from .configuration_codeshell import CodeShellConfig
from .modeling_codeshell import CodeShellModel,  CodeShellForCausalLM

class LlavaCodeShellConfig(CodeShellConfig):
    model_type = "llava_codeshell"


class LlavaCodeShellModel(LlavaMetaModel, CodeShellModel):
    config_class = LlavaCodeShellConfig

    def __init__(self, config: CodeShellConfig):
        super(LlavaCodeShellModel, self).__init__(config)


class LlavaCodeShellForCausalLM(CodeShellForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaCodeShellConfig

    def __init__(self, config):
        super(CodeShellForCausalLM, self).__init__(config)
        self.transformer = LlavaCodeShellModel(config)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.transformer
    def return_last_token_hidden(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        use_eos: Optional[bool] = None,
        captions: Optional[torch.LongTensor] = None,
        hallucination_caption: Optional[torch.LongTensor] = None,
        ) -> Union[Tuple, CausalLMOutputWithPast]:

        # print('input_ids.shape, labels.shape ', input_ids.shape, labels.shape)
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=True,
            output_hidden_states=True,
            return_dict=return_dict,
        )
        
        last_hidden_state = outputs.last_hidden_state
    
        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]
            
        if input_ids is not None:
             
            if use_eos:
                
                sequence_lengths = torch.where(input_ids == self.tokenizer.eos_token_id)[1][0].to(last_hidden_state.device)
            else:
                sequence_lengths = (torch.ne(input_ids, self.tokenizer.pad_token_id).sum(-1) - 1).to(last_hidden_state.device)
        else:
            sequence_lengths = -1
        # print(last_hidden_state.shape,sequence_lengths)
        global_representation = last_hidden_state[:, sequence_lengths, :]
        return global_representation, outputs

    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
        captions: Optional[torch.LongTensor] = None,
        captions_attention_mask: Optional[torch.LongTensor] = None,
        hallucination_caption: Optional[torch.LongTensor] = None
        ) -> Union[Tuple, CausalLMOutputWithPast]:
        # print('input_ids.shape, labels.shape ', input_ids.shape, labels.shape)
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # print(attention_mask.shape)
        input_ids, attention_mask, past_key_values, inputs_embeds, labels, image_features  = self.prepare_inputs_labels_for_multimodal(input_ids, attention_mask, past_key_values, labels, images)
        # print('---',attention_mask.shape)
        # print('input_ids.shape', input_ids.shape)
        # print('inputs_embeds', inputs_embeds.shape)
        # print('image_features', image_features.shape)
        # print('label', labels.shape)

        if captions is not None:
            if self.transformer.config.add_eos:
                eos_embeds =  self.get_model().get_input_embeddings()(torch.tensor([[self.tokenizer.eos_token_id]]).to(images.device))
                eos_embeds =  eos_embeds.expand(image_features.shape[0], -1, -1)
                image_features = torch.cat([ image_features,eos_embeds], dim=1)
            
            image_last_token_hidden,_ = self.return_last_token_hidden(input_ids=None, inputs_embeds=image_features, use_eos=self.transformer.config.add_eos)
            text_last_token_hidden,outputs = self.return_last_token_hidden(input_ids=captions, inputs_embeds=None, use_eos =self.transformer.config.add_eos )
             
            image_global_embedding = F.normalize(self.transformer.vision_projector(image_last_token_hidden))
            text_global_embedding = F.normalize(self.transformer.text_projector(text_last_token_hidden))
            
            if self.transformer.config.use_queue:
                text_queue = self.transformer.text_queue.clone().detach()
                image_queue = self.transformer.image_queue.clone().detach()
                text_feat_all = torch.cat([text_global_embedding.t(),text_queue], dim=1)
                image_feat_all = torch.cat([image_global_embedding.t(), image_queue], dim=1)
            else:
                if self.transformer.config.gather_all:
                    image_feat_all = allgather(image_global_embedding, dist.get_rank(), dist.get_world_size()).t()
                    text_feat_all = allgather(text_global_embedding, dist.get_rank(), dist.get_world_size()).t()
                else:
                    image_feat_all = image_global_embedding.t()
                    if hallucination_caption is not None:
                        hallucination_last_token_hidden,_ = self.return_last_token_hidden(input_ids=hallucination_caption, inputs_embeds=None, use_eos =self.transformer.config.add_eos )
                        hallucination_global_embedding = F.normalize(self.transformer.text_projector(hallucination_last_token_hidden))
                        # print(hallucination_global_embedding.shape, text_global_embedding.shape)
                        text_feat_all = torch.cat([text_global_embedding.t(), hallucination_global_embedding.t()], dim=1)
                        # print(text_feat_all.shape)
                    else:   
                        text_feat_all = text_global_embedding.t()
             
            with torch.no_grad():
                self.transformer.itc_temp.clamp_(0.001, 0.5)
           
            
            sim_i2t = image_global_embedding @ text_feat_all / self.transformer.itc_temp
            sim_t2i = text_global_embedding @ image_feat_all / self.transformer.itc_temp
            
            sim_i2t_targets = torch.zeros(sim_i2t.size()).to(inputs_embeds.device)
            sim_i2t_targets.fill_diagonal_(1)

            sim_t2i_targets = torch.zeros(sim_t2i.size()).to(inputs_embeds.device)
            sim_t2i_targets.fill_diagonal_(1)
                
            loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1) * sim_i2t_targets, dim=1).mean()
            loss_t2i = -torch.sum(F.log_softmax(sim_t2i, dim=1) * sim_t2i_targets, dim=1).mean()
            
            loss_ita = (loss_i2t + loss_t2i) / 2
            loss = None
            loss_generation=None
            
            if self.config.do_generation:
                outputs = self.transformer(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict
                )

                hidden_states = outputs[0]
                logits = self.lm_head(hidden_states)
                
                if labels is not None:
                    # Shift so that tokens < n predict n
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()
                    # Flatten the tokens
                    loss_fct = CrossEntropyLoss()
                    shift_logits = shift_logits.view(-1, self.config.vocab_size)
                    shift_labels = shift_labels.view(-1)
                    # Enable model/pipeline parallelism
                    shift_labels = shift_labels.to(shift_logits.device)
                    loss_generation = loss_fct(shift_logits, shift_labels)
                    loss = loss_generation + loss_ita * self.transformer.config.ita_weight

            else:
                loss = loss_ita 
                loss_generation = loss_ita
                logits = None
            if self.transformer.config.use_queue:
                self.transformer._dequeue_and_enqueue(image_global_embedding, text_global_embedding)
        else:     
            outputs = self.transformer(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict
            )

            hidden_states = outputs[0]
            logits = self.lm_head(hidden_states)
            
            loss = None
            if labels is not None:
                # Shift so that tokens < n predict n
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                # Flatten the tokens
                loss_fct = CrossEntropyLoss()
                shift_logits = shift_logits.view(-1, self.config.vocab_size)
                shift_labels = shift_labels.view(-1)
                # Enable model/pipeline parallelism
                shift_labels = shift_labels.to(shift_logits.device)
                loss_generation = loss_fct(shift_logits, shift_labels)
                loss_ita = loss_generation
                loss = loss_generation
                loss_dict = {'ce_loss': loss_generation.clone().detach(),'ita_loss': loss_ita.clone().detach()}
            else:
                loss_dict = None

        if not return_dict:
            output = (logits,) + outputs[1:] +[loss_dict]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            loss_dict = loss_dict
        )
         
        
    def add_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer
        print(self.tokenizer.vocab_size)
    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "images": kwargs.get("images", None),
            }
        )
        return model_inputs

AutoConfig.register("llava_codeshell", LlavaCodeShellConfig)
AutoModelForCausalLM.register(LlavaCodeShellConfig, LlavaCodeShellForCausalLM)
