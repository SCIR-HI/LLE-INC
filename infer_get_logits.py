# -*- coding: utf-8 -*-
# author:Haochun Wang
import os.path
import sys
from tqdm import tqdm
import pickle
import fire
import torch
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer


if torch.cuda.is_available():
    device = "cuda"

def main(
        load_8bit: bool = False,
        base_model: str = "llama model path",
):

    tokenizer = LlamaTokenizer.from_pretrained(base_model)
    model = LlamaForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=load_8bit,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.config.pad_token_id = tokenizer.pad_token_id = 0
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2
    if not load_8bit:
        model.half()

    model.eval()

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    def evaluate(
            input=None,
            temperature=0.1,
            top_p=0.75,
            top_k=1,
            max_new_tokens=256,
            **kwargs,
    ):
        input_content = input[0]
        inputs = tokenizer(input_content, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            no_repeat_ngram_size=1,
            repetition_penalty=0.2,
            num_beams=1,
            do_sample=False,
            output_hidden_states=True,
            **kwargs,
        )
        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                no_repeat_ngram_size=1,
                repetition_penalty=0.2,
                max_new_tokens=max_new_tokens,
            )
        return generation_output.hidden_states[0][-1][0][-1].tolist()

    # sst2
    seed = 13
    data_dir = 'SST2/%s/train.tsv' % seed
    train_dataset = []
    with open(data_dir, 'r') as f:
        lines = f.readlines()
        for (i, line) in enumerate(lines[1:]):
            line = line.strip().split("\t")
            text_a = line[0] + ' It is '
            label = int(line[1])
            train_dataset.append([text_a, label])
    index = 0
    for j in tqdm(train_dataset):
        try:
            with open(os.path.join('SST2', '%s' % seed, 'train', '%s.pickle' % str(index)), 'wb') as fn:
                pickle.dump([j,evaluate(input=j)], fn)
            index += 1
        except:
            continue
if __name__ == "__main__":
    fire.Fire(main)

