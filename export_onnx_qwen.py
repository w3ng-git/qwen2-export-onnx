import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import os
import sys

def load_json(file_path):   # for loading the config
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            print("Loaded config, here's the info you may need later for inference!"+16*'=')
            print(f"[*] kv heads num={data['num_key_value_heads']}")
            print(f"[*] kv heads dim={data['hidden_size'] // data['num_attention_heads']}")
            print(80*"=")
        return data
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
    except json.JSONDecodeError:
        print(f"Error: Failed to decode JSON from the file {file_path}.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


# ================================= Export Language Model =================================
# load model and tokenizer
model_id = sys.argv[1]  # input your model id in the command line
model = AutoModelForCausalLM.from_pretrained(model_id).eval().half()
tokenizer = AutoTokenizer.from_pretrained(model_id)
dummy_model_input = tokenizer("This is a sample", return_tensors="pt")
model_config = load_json(os.path.join(model_id, 'config.json'))

dynamic_axes = {
    'input_ids': {1: 'N', },
    'attention_mask': {1: 'N'},
}
input_names = ['input_ids', 'attention_mask']
output_names = ['logits']
past_key_values = []
for i in range(model_config['num_hidden_layers']):     # num layers
    num_key_value_heads = model_config['num_key_value_heads']
    num_attention_heads = model_config['num_attention_heads']
    hidden_size = model_config['hidden_size']
    kv_dim = hidden_size // num_attention_heads

    # for example, [1, num_key_value_heads, 0, kv_dim] = [1, 2, 0, 64] for qwen2-0.5b
    past_key_in = torch.randn([1, num_key_value_heads, 0, kv_dim], dtype=torch.float16)
    past_value_in = torch.randn([1, num_key_value_heads, 0, kv_dim], dtype=torch.float16)
    input_names.extend([f"past_key_in{i}", f"past_value_in{i}"])
    output_names.extend([f"past_key_out{i}", f"past_value_out{i}"])
    dynamic_axes[f"past_key_in{i}"] = {2: "N"}    # The third dim
    dynamic_axes[f"past_value_in{i}"] = {2: "N"}    # The third dim
    past_key_values.append((past_key_in, past_value_in))

# export
torch.onnx.export(
    model,
    tuple(dummy_model_input.values()) + (None,) + (past_key_values,),   # position ids:None
    f="./output_onnx/language-model.onnx",
    input_names=input_names,    # ['input_ids', 'attention_mask' , <<PAST KVs>>]
    output_names=output_names,
    dynamic_axes=dynamic_axes,
    do_constant_folding=False,
    opset_version=14,   # for qwen models, opset version requires to be >= 14
)

# ================================= Export Logits Model ================================= #
def repetitionPenaltyLogitsProcess(input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
    penalty = 1.1
    score = torch.gather(scores, 1, input_ids)
    # if score < 0 then repetition penalty has to be multiplied to reduce the token probabilities
    score = torch.where(score < 0, score * penalty, score / penalty)
    scores.scatter_(1, input_ids, score)
    return scores


def topK(scores: torch.FloatTensor) -> torch.FloatTensor:
    top_k = 50
    filter_value = -float("Inf")
    top_k = min(top_k, scores.size(-1))  # Safety check
    # Remove all tokens with a probability less than the last token of the top-k
    indices_to_remove = scores < torch.topk(scores, top_k)[0][..., -1, None]
    scores = scores.masked_fill(indices_to_remove, filter_value)
    return scores


def topP(scores: torch.FloatTensor) -> torch.FloatTensor:
    top_p = 0.8
    filter_value = -float("Inf")
    min_tokens_to_keep = 1
    sorted_logits, sorted_indices = torch.sort(scores, descending=False)
    cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
    # Remove tokens with cumulative top_p above the threshold (token with 0 are kept)
    sorted_indices_to_remove = cumulative_probs <= (1 - top_p)
    # Keep at least min_tokens_to_keep
    sorted_indices_to_remove[..., -min_tokens_to_keep:] = 0
    # scatter sorted tensors to original indexing
    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
    scores = scores.masked_fill(indices_to_remove, filter_value)
    return scores


def processLogits(input_ids, next_token_logits):
    next_token_scores = repetitionPenaltyLogitsProcess(input_ids, next_token_logits)
    next_token_scores = topK(next_token_scores)
    next_token_scores = topP(next_token_scores)
    probs = torch.nn.functional.softmax(next_token_scores, dim=-1)
    next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
    return next_tokens


class LogitsModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, all_input_ids, logits):
        return processLogits(all_input_ids, logits[:, -1, :])

m = LogitsModel()   # topK -> topP -> sample
dynamic_axes = {
    'all_input_ids': {1: 'X', },
}
logits = torch.randn([1, 1, 151936])


torch.onnx.export(
    m,
    (dummy_model_input['input_ids'], logits),
    './output_onnx/logits-model.onnx',
    opset_version=14,
    do_constant_folding=False,
    input_names=["all_input_ids", "logits"],
    output_names=["token_id"],
    dynamic_axes=dynamic_axes,
    verbose=False,
)
