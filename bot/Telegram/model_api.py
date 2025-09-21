from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

MODEL_PATH = "Qwen/Qwen2.5-14B-Instruct"

print(f"Loading model from {MODEL_PATH} ...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

# Better device management to avoid meta/cuda device conflicts
if torch.cuda.is_available():
    load_kwargs = {
        "torch_dtype": torch.float16,
        "device_map": "auto",
        "trust_remote_code": True,
        "offload_folder": None  # Prevent CPU offloading that causes meta device issues
    }
else:
    load_kwargs = {
        "torch_dtype": torch.float16,
        "device_map": "cpu",
        "trust_remote_code": True
    }

model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, **load_kwargs)

def generate_reply(messages, max_new_tokens=128):
    try:
        input_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        )
        
        # Ensure consistent device placement
        if hasattr(model, 'device'):
            input_ids = input_ids.to(model.device)
        elif torch.cuda.is_available():
            input_ids = input_ids.to('cuda')

        outputs = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id  # Prevent padding issues
        )
        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return text
        
    except Exception as e:
        return f"⚠️ Generation error: {str(e)[:200]}..."
