from typing import Dict, Any
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.models.base import BaseModel

class EmotionLLaMAModel(BaseModel):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        self.model = AutoModelForCausalLM.from_pretrained(
            "ZebangCheng/Emotion-LLaMA",
            torch_dtype=torch.float16
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            "ZebangCheng/Emotion-LLaMA"
        )
        
    def _create_prompt(self, text: str) -> str:
        return f"""
        Analyze the emotional content in the following text:
        {text}
        Emotion:
        """
        
    def forward(self, batch):
        prompts = [self._create_prompt(text) for text in batch["text"]]
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(self.device)
        outputs = self.model.generate(**inputs)
        return self._process_outputs(outputs) 