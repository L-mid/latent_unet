
import logging
from transformers import AutoTokenizer 
import open_clip     

logger = logging.getLogger(__name__)

# === NOTES
"""
This tokenizer has a lot of hidden dependencies and likely causes issues on import.

"""

# ---------------------------------------------------------------------------------
# Tokenizer Wrapper Interface
# ---------------------------------------------------------------------------------

class Tokenizer:
    # Unified tokenizer interface.

    def __init__(self, tokenizer, max_length: int, pad_token: str = None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_token = pad_token

    def encode(self, text: str):
        return self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
    
    def encode_batch(self, texts):
        return self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
    
    @property
    def vocab_size(self):
        return self.tokenizer.vocab_size
    

# --------------------------------------------------------------------------------
# Tokenizer Builders
# ------------------------------------------------------------------------------

def build_tokenizer(cfg) -> Tokenizer:
    # Build tokenizer from config.

    name = cfg.data.tokenizer.name.lower() 
    max_len = cfg.data.tokenizer.max_length

    if name == "bert":
        logger.info("[TOKENIZER] Using bert tokenizer")
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        return Tokenizer(tokenizer, max_length=max_len)
    
    if name == "clip":
        logger.info("[TOKENIZER] Using OpenCLIP tokenizer")
        tokenizer = open_clip.get_tokenizer("ViT-B-32")
        # OpenClip tokenizer has different interface - wrap it manually:
        return OpenCLIPTokenizerWrapper(tokenizer, max_length=max_len)
    
    if name == "t5":
        logger.info("[TOKENIZER] Using HuggingFace T5 tokenizer")
        tokenizer = AutoTokenizer.from_pretrained("t5-base")
        return Tokenizer(tokenizer, max_length=max_len)
    
    else:
        raise ValueError(f"Unknown tokenizer: {name}")
    

# -------------------------------------------------------------------------
# OpenCLIP Tokenizer Special Wrapper
# ------------------------------------------------------------------------------

class OpenCLIPTokenizerWrapper:
    # Special wrapper for OpenCLIP tokenizer interface.

    def __init__(self, clip_tokenizer, max_length: int):
        self.tokenizer = clip_tokenizer
        self.max_length = max_length

    def encode(self, text: str):
        tokens = self.tokenizer([text], context_length=self.max_length)
        return {"input_ids": tokens} 
    
    def encode_batch(self, texts):
        tokens = self.tokenizer(texts, context_length=self.max_length)
        return {"input_ids": tokens}
    
    @property
    def vocab_size(self):
        return self.tokenizer.vocab_size
    








