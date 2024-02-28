from PIL import Image
from io import BytesIO
import base64
import json
import time
import os
import torch
import pandas as pd
from tqdm import tqdm

import llava.transformers as transformers
import requests
from llava.transformers import TextStreamer
import argparse
import glob

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.model.language_model.llava_codeshell import LlavaCodeShellForCausalLM
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria


def load_image(image_file):
    if image_file.startswith('http') or image_file.startswith('https'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image_file = os.path.join('/hdd2/jcy/jcy/data/coco/val2014', image_file)
        image = Image.open(image_file).convert('RGB')
    return image
# MODEL_PATH=
MODEL_PATH=["/hdd2/jcy/ckpt/llava-codeshell-chat-hal-SFT-itc","/hdd2/jcy/project/LLaVA_hal/output/llava-codeshell-chat-hal-SFT","/hdd2/jcy/ckpt/llava-vicuna-7b-v1.3-finetune","/hdd2/jcy/jcy/project/output/llava-v1.5-7b-SFT","/hdd2/jcy/project/LLaVA_hal/output/llava-v1.5-7b-hal-SFT",'/shd/jcy/project/LLaVA_itc_v3/output/llava-vicuna-7b-v1.5-finetune/checkpoint-1234','/shd/jcy/project/LLaVA_itc_v3/output/llava-vicuna-7b-v1.3-sft_itc_coco_no_eos/checkpoint-1000']
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
parser.add_argument('--qdir', type=str, default= '/hdd2/jcy/jcy/data/pope/coco_pope_adversarial.json')
parser.add_argument('--odir', type=str, default='./adversarial_llavahal.json')
args = parser.parse_args()    


#也可以换成random、popular、adversarial
qdir = args.qdir
odir = args.odir

disable_torch_init()
model_name = get_model_name_from_path(args.model_path)
# print(model_name)
# # tokenizer = transformers.AutoTokenizer.from_pretrained(
# #             "/hdd2/jcy/ckpt/codeshell-chat/checkpoint-1800",
# #             cache_dir=None,
# #             model_max_length=512,
# #             padding_side="right",
# #             use_fast=False,
# #         )
# tokenizer = transformers.AutoTokenizer.from_pretrained(
#             "/hdd2/jcy/ckpt/codeshell-chat/checkpoint-1800",
#             cache_dir=None,
#             model_max_length=512,
#             padding_side="right"
#         )

# bnb_model_from_pretrained_args = {}
# model = LlavaCodeShellForCausalLM.from_pretrained(
#     "/hdd2/jcy/ckpt/codeshell-chat/checkpoint-1800",
#     cache_dir=None,
#     **bnb_model_from_pretrained_args
# )

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

stop_str = '<lendoftext|>'
keywords = [stop_str]


# PROMPT = '{}. Does the description match the image exactly? If so, please tell me "yes". If not, please tell me "no".'
# PROMPT = '```{}```. Does the caption delimited by triple backticks match the image you saw exactly? Just tell me "yes" or "no".'
PROMPT = '```{}```. Does the caption delimited by triple backticks match the image exactly? Let\u2019s work this out in a step by step way to be sure we have the right answer. If so, return "yes". If not, return "no".'
PROMPT = '```{}```. Does the caption delimited by triple backticks match the image exactly? If so, return "yes". If not, return "no".'

result = []
cnt = 0

with open("/hdd2/jcy/project/coco_eval_hal_5k.json", "r") as f:
    data = json.load(f)
    for item in data:
        res = {}
        res['predict'] = []
        res['result'] = []
        gt = item["caption"]
        space = item["hal_caption"][0]["caption"]
        object = item["hal_caption"][1]["caption"]
        attributive = item["hal_caption"][2]["caption"]
        addistional = item["hal_caption"][3]["caption"]
        tmp = [gt, space, object, attributive, addistional]

        img_path = os.path.join('/hdd2/jcy/jcy/data/coco',item["image"])

        image = load_image(img_path)
        image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'].half().cuda()

        system = "## system:You are a helpful assistant, Please ensure that your responses are helpful, detailed, and polite. \n"


        for prompt in tmp:
            prompt = system + ' ' + '## huaman:' + DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '<lendoftext|>' + PROMPT.format(prompt) + '<lendoftext|>' + '## assistant:'
            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
            stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
            # streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
             
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor,
                    do_sample=True,
                    temperature=0.2,
                    max_new_tokens=64,
                    # streamer=streamer,
                    use_cache=True,
                    stopping_criteria=[stopping_criteria])
                
            outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()

            re = outputs.split(',')[0].strip()
            res['predict'].append(re)


            print(prompt)
            print(re)
            if 'no' in re.lower() or 'not' in re.lower():
                res['result'].append(0)
                print(0)
            elif 'yes' in re.lower() or 'match' in re.lower():
                res['result'].append(1)
                print(1)
            else:
                res['result'].append(2)
                print(2)

        result.append(res)
        print("finish ", cnt)
        cnt += 1

with open('/hdd2/jcy/5k_two/code2.json','w') as output:
    json.dump(result,output)