from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from typing import Optional, Dict, Any

def get_llm_pipeline(
    model_id: str,
    task: str = "text-generation",
    device_map: str = "auto",
    max_length: int = 2048,
    temperature: float = 0.7,
    top_p: float = 0.95,
    repetition_penalty: float = 1.15,
    model: Optional[AutoModelForCausalLM] = None,
    tokenizer: Optional[AutoTokenizer] = None,
    **kwargs: Dict[str, Any]
):
    """Create a text generation pipeline with the specified model and parameters."""
    if model is None or tokenizer is None:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map=device_map,
            trust_remote_code=True,
            **kwargs
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True,
            **kwargs
        )

    return pipeline(
        task,
        model=model,
        tokenizer=tokenizer,
        max_length=max_length,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        **kwargs
    ) 