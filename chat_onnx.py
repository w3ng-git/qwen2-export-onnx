import sys
import os
import json
import torch
import numpy as np
import onnxruntime as ort
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = sys.argv[1]  # tokenizer and *model* config dir(the config.json should be included!)
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

session = ort.InferenceSession("./output_onnx/language-model.onnx")
logits_session = ort.InferenceSession("./output_onnx/logits-model.onnx")

def load_json(file_path):   # for loading the config
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
        return data['num_hidden_layers'],\
            data['num_key_value_heads'],\
            data['hidden_size'] // data['num_attention_heads']


n_layer, n_kv_heads, dim_head = load_json(os.path.join(model_id, 'config.json'))

def generate(input_ids, past_key_values):   # call the onnx inference session, one forward
    inputs = {
        "input_ids": input_ids,
        "attention_mask": np.ones_like(input_ids),
    }
    for i in range(n_layer):
        key_index = 2 * i
        value_index = key_index + 1
        inputs["past_key_in" + str(i)] = past_key_values[key_index]
        inputs["past_value_in" + str(i)] = past_key_values[value_index]
    outputs = session.run(None, inputs)

    return logits_session.run(None, {
        "all_input_ids": input_ids,
        "logits": outputs[0][:, -1:, :],
    }), outputs[1:]


max_len = 300
chat_template = [   # also chat history, init sys msg
    {'role': 'system', 'content': 'You are a helpful assistant.'},
]


def get_reply_with_implicit_history(new_user_content):
    global chat_template
    chat_template.append({'role': 'user', 'content': new_user_content})
    input_ids = tokenizer(
        tokenizer.apply_chat_template(chat_template, tokenize=False, add_generation_prompt=False) + '<|im_start|>doctor\n',
        return_tensors='pt')['input_ids'].numpy()
    # Define(clear) the KV, because using the previous KV directly will lead to onnx err
    past_key_values = []
    for i in range(2*n_layer):
        past_key_values.append(np.zeros((1, n_kv_heads, 0, dim_head), np.float16))

    assistant_reply = ''
    for i in range(max_len):
        next_token_id, past_key_values = generate(input_ids, past_key_values)
        next_token_id_ = next_token_id[0].item()
        if next_token_id_ == 151643 or next_token_id_ == 151645:    # eos or im_end
            break
        next_token = tokenizer.decode(next_token_id_)
        assistant_reply = assistant_reply + next_token
        print(next_token, end='', flush=True)
        input_ids = next_token_id
    chat_template.append({'role': 'doctor', 'content': assistant_reply})


if __name__ == '__main__':
    import readline
    while True:     # multi-turn chat with history memory
        userin = input('\n[user]>')
        print('[assistant]>', end='', flush=True)
        get_reply_with_implicit_history(userin)
