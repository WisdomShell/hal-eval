from PIL import Image
from io import BytesIO
import base64
import json
import time
import re,os
import torch
import pandas as pd
from tqdm import tqdm

import transformers
import requests
from transformers import TextStreamer
import argparse

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria


def load_image(image_file):
    if image_file.startswith('http') or image_file.startswith('https'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image
# MODEL_PATH=
MODEL_PATH=["/hdd2/jcy/project/LLaVA_hal/output/llava-v1.5-7b-hal-SFT_only_use_coco_hal","/hdd2/jcy/project/LLaVA_hal/output/llava-v1.5-7b-hal-SFT_use_coco_share_gpt4_hal","/hdd2/jcy/project/LLaVA_hal/output/llava-v1.5-7b-hal-SFT_only_use_coco_hal","/hdd2/jcy/project/LLaVA_hal/output/llava-v1.5-7b-hal-SFT","/shd/jcy/project/LLaVA_itc_v3/output/llava-7b-v1.3-sft_itc",'/shd/jcy/project/LLaVA_itc_v3/output/llava-vicuna-7b-v1.5-finetune/checkpoint-1234','/shd/jcy/project/LLaVA_itc_v3/output/llava-vicuna-7b-v1.3-sft_itc_coco_no_eos/checkpoint-1000']
parser = argparse.ArgumentParser()
parser.add_argument("--model-path", type=str, default=MODEL_PATH[0])
parser.add_argument("--model-base", type=str, default=None)
# parser.add_argument("--image-file", type=str, required=True)
parser.add_argument("--num-gpus", type=int, default=1)
parser.add_argument("--device", type=str, choices=["cuda", "cpu"], default="cuda")
parser.add_argument("--conv-mode", type=str, default=None)
parser.add_argument("--temperature", type=float, default=0.2)
parser.add_argument("--max-new-tokens", type=int, default=512)
parser.add_argument("--debug", action="store_true")
parser.add_argument('--qdir', type=str, default= '/hdd2/jcy/LLaVA1.5/outdomain_short/result1_jhr_llava_select_short_out.json')
parser.add_argument('--odir', type=str, default='/hdd2/jcy/LLaVA1.5/outdomain_short/result_llava_jhr_select_evalres.json')
args = parser.parse_args()    


#也可以换成random、popular、adversarial
qdir = args.qdir
odir = args.odir

result = []

data_sum = 0
hallu_sum = 0
object_sum = 0
relation_sum = 0
attribute_sum = 0
additional_sum = 0
multitype_sum = 0
else_sum = 0

with open(qdir,'r') as f:
    disable_torch_init()
    model_name = get_model_name_from_path(args.model_path)
    print(model_name)
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name)

    #不使用，只用来提供stop_str
    if "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"
    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print('[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(conv_mode, args.conv_mode, args.conv_mode))
    else:
        args.conv_mode = conv_mode
    conv = conv_templates[args.conv_mode].copy()

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]


    data = json.load(f)
    #格式太诡异了，只能一次读一行
    for data_item in data:
        # question_id = data['question_id']
        data_sum += 1

        image = data_item['image']
        caption = data_item['predict']
        text = "Caption: {}\nQuestion: Based on the image, whether there is any hallucinatory content in the caption?".format(caption)
        # label = data['label']

        image = load_image(image)
        image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'].half().cuda()

        system = "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions."
        
        question = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + text # caption:{}\n Question:{}
        prompt = system + ' ' + 'USER: ' + question + '</s>' + 'ASSISTANT: '

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
        # streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                do_sample=True,
                temperature=0.2,
                max_new_tokens=32,
                # streamer=streamer,
                use_cache=True,
                stopping_criteria=[stopping_criteria])
        
        outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
        # if yes: --> line 87 
        # object  / relationship / attribute / additional 
        first_str = outputs.strip()
        yes_index = first_str.find('Yes')
        no_index = first_str.find('No')
        first_answer = ''
        if yes_index == -1 and no_index == -1:
            first_answer = 'No'
        elif yes_index == -1:
            first_answer = 'No'
        elif no_index == -1:
            first_answer = 'Yes'
        elif yes_index < no_index:
            first_answer = 'Yes'
        else:
            first_answer = 'No'
        data_item['first_answer'] = first_str
        if first_answer == 'Yes':
            hallu_sum += 1
            # prompt = system + ' ' + 'USER: ' + question + '</s>' + 'ASSISTANT: '
            prompt_add = prompt + first_str + 'USER: What types of hallucinatory information are present?' + '</s>' + 'ASSISTANT: The types of hallucinations contained in the description are as follows: \n Hallucination Type: '
            input_ids = tokenizer_image_token(prompt_add, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
            stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
            # streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor,
                    do_sample=True,
                    temperature=0.2,
                    max_new_tokens=32,
                    # streamer=streamer,
                    use_cache=True,
                    stopping_criteria=[stopping_criteria])
            
            outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
            # if yes: --> line 87 
            # object  / relationship / attribute / additional 
            second_str = outputs.strip()

            if second_str.find('object') != -1 or second_str.find('Object') != -1:
                object_sum += 1
            if second_str.find('relation') != -1 or second_str.find('Relation') != -1:
                relation_sum += 1
            if second_str.find('attribut') != -1 or second_str.find('Attribut') != -1:
                attribute_sum += 1
            if second_str.find('additional') != -1 or second_str.find('Additional') != -1 or second_str.find('aditional') != -1 or second_str.find('Aditional') != -1:
                additional_sum += 1
            if second_str.find('Multitype') != -1 or second_str.find('multitype') != -1:
                multitype_sum += 1
            if second_str.find('object') == -1 and second_str.find('Object') == -1 and second_str.find('relation') == -1 and second_str.find('Relation') == -1 and second_str.find('attribut') == -1 and second_str.find('Attribut') == -1 and second_str.find('additional') == -1 and second_str.find('Additional') == -1:
                else_sum += 1
            data_item['second_answer'] = second_str

        print("Finish: ", data_sum)
        result.append(data_item)

hallu_ratio = float(hallu_sum) / float(data_sum)
object_ratio = float(object_sum) / float(hallu_sum)
relation_ratio = float(relation_sum) / float(hallu_sum)
attribute_ratio = float(attribute_sum) / float(hallu_sum)
additional_ratio = float(additional_sum) / float(hallu_sum)
multitype_ratio = float(multitype_sum) / float(hallu_sum)
else_ratio = float(else_sum) / float(hallu_sum)
print("data_sum: ", data_sum)
print("hallu_sum: ", hallu_sum)
print("object_sum: ", object_sum)
print("relation_sum: ", relation_sum)
print("attribute_sum: ", attribute_sum)
print("additional_sum: ", additional_sum)
print("multitype_sum: ", multitype_sum)
print("else_sum: ", else_sum)
print("hallu_ratio: ", hallu_ratio)
print("object_ratio: ", object_ratio)
print("relation_ratio: ", relation_ratio)
print("attribute_ratio: ", attribute_ratio)
print("additional_ratio: ", additional_ratio)
print("multitype_ratio: ", multitype_ratio)
print("else_ratio: ", else_ratio)
with open(odir,'w') as output:
    json.dump(result,output)