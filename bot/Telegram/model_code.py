import os, torch
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")  # Avoid Dynamo/meta device issues
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from threading import Thread

MODEL_ID = os.getenv("MODEL_CODE", "Qwen/Qwen2.5-Coder-14B")

_tokenizer = None
_model = None

def _load():
    global _tokenizer, _model
    if _tokenizer is None or _model is None:
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

        # Prefer direct CUDA load to avoid accelerate/meta device sharding issues
        if torch.cuda.is_available():
            try:
                _model = AutoModelForCausalLM.from_pretrained(
                    MODEL_ID,
                    torch_dtype=torch.float16,
                    trust_remote_code=True,
                )
                _model.to("cuda")
                _model.eval()
            except RuntimeError as e:
                # Fallback to auto sharding if GPU OOM
                if "out of memory" in str(e).lower():
                    _model = AutoModelForCausalLM.from_pretrained(
                        MODEL_ID,
                        torch_dtype=torch.float16,
                        trust_remote_code=True,
                        device_map="auto",
                    )
                    _model.eval()
                else:
                    raise
        else:
            # CPU fallback
            _model = AutoModelForCausalLM.from_pretrained(
                MODEL_ID,
                torch_dtype=torch.float32,
                trust_remote_code=True,
            )
            _model.eval()
    return _tokenizer, _model

def stream_generate(messages, max_new_tokens: int = 1024, temperature: float = 0.2):
    """
    Character-based streaming generator for code model.
    Yields one character at a time.
    """
    try:
        tok, mdl = _load()
        
        # Extract the user's request (last user message)
        user_prompt = ""
        for m in messages:
            if m['role'] == 'user':
                user_prompt = m['content']
        
        # Construct a specialized coding prompt for qwen-coder
        prompt = f"""You are an expert programming assistant specializing in generating high-quality, well-documented code.

Please provide complete, working code solutions that follow best practices. Include:
- Clear comments explaining complex logic
- Proper error handling
- Efficient implementations
- Descriptive variable/function names

If the user's request is ambiguous, make reasonable assumptions and explain them.

REQUEST: {user_prompt}

CODE SOLUTION:
"""

        # Ensure inputs are on the correct device
        inputs = tok(prompt, return_tensors="pt")
        if hasattr(mdl, 'device'):
            inputs = inputs.to(mdl.device)
        elif torch.cuda.is_available():
            inputs = inputs.to('cuda')
            
        streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)
        gen_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=0.95,
            pad_token_id=tok.eos_token_id  # Prevent padding issues
        )

        thread = Thread(target=mdl.generate, kwargs=gen_kwargs)
        thread.start()

        buffer = ""
        for new_text in streamer:
            buffer += new_text
            while buffer:
                yield buffer[0]
                buffer = buffer[1:]
                
    except Exception as e:
        # Fallback response if generation fails
        error_msg = f"⚠️ Code generation error: {str(e)[:100]}..."
        for char in error_msg:
            yield char
