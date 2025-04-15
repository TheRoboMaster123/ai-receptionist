from typing import Optional, Dict, Any
import torch
from pathlib import Path
import os
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
import logging
from datetime import datetime
import json
import shutil

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelManager:
    def __init__(
        self,
        model_id: str,
        cache_dir: str = "models",
        use_4bit: bool = False,
        use_8bit: bool = False,
        device_map: str = "auto",
        token: Optional[str] = None
    ):
        self.model_id = model_id
        self.cache_dir = cache_dir
        self.use_4bit = use_4bit
        self.use_8bit = use_8bit
        self.device_map = device_map
        self.token = token
        
        self.model = None
        self.tokenizer = None
        
        # Create cache directory if it doesn't exist
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        
        # Initialize metadata file
        self.metadata_file = Path(cache_dir) / "model_metadata.json"
        self.metadata = self._load_metadata()
    
    def _load_metadata(self) -> Dict[str, Any]:
        """Load or create model metadata"""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {
            "model_id": self.model_id,
            "last_updated": None,
            "download_status": "not_downloaded",
            "config": {
                "use_4bit": self.use_4bit,
                "use_8bit": self.use_8bit,
                "device_map": self.device_map
            }
        }
    
    def _save_metadata(self):
        """Save model metadata"""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def _get_quantization_config(self) -> Optional[BitsAndBytesConfig]:
        """Get quantization configuration based on settings"""
        if self.use_4bit:
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True
            )
        return None
    
    def initialize_model(self, force_reload: bool = False) -> None:
        """Initialize or reload the model."""
        try:
            logger.info(f"Initializing model {self.model_id}")
            
            if self.model is not None and not force_reload:
                logger.info("Model already loaded")
                return
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_id,
                cache_dir=self.cache_dir,
                trust_remote_code=True,
                token=self.token
            )
            
            # Configure quantization for A5000 GPU
            if self.use_8bit:
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_threshold=6.0,
                    llm_int8_has_fp16_weight=True,
                    llm_int8_enable_fp32_cpu_offload=False  # Disable CPU offload since we have enough VRAM
                )
            elif self.use_4bit:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True
                )
            else:
                quantization_config = None
            
            # Load model with optimized configuration for A5000
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                cache_dir=self.cache_dir,
                device_map=self.device_map,
                trust_remote_code=True,
                token=self.token,
                quantization_config=quantization_config,
                torch_dtype=torch.float16,  # Use FP16 for better memory efficiency
                max_memory={0: "22GB"}  # Reserve 22GB for model, leaving 2GB for overhead
            )
            
            # Optimize for inference
            self.model.eval()  # Set to evaluation mode
            if torch.cuda.is_available():
                torch.cuda.empty_cache()  # Clear any unused memory
                torch.backends.cudnn.benchmark = True  # Enable cudnn autotuner
            
            # Force model to GPU if specific device is set
            if self.device_map not in ["auto", None]:
                self.model.to(self.device_map)
            
            # Update metadata
            self.metadata.update({
                "last_updated": datetime.now().isoformat(),
                "download_status": "downloaded",
                "config": {
                    "use_4bit": self.use_4bit,
                    "use_8bit": self.use_8bit,
                    "device_map": self.device_map,
                    "gpu_mem_reserved": "22GB"
                }
            })
            self._save_metadata()
            
            logger.info(f"Model initialized successfully on {self.device_map} with optimized A5000 configuration")
            
        except Exception as e:
            logger.error(f"Error initializing model: {str(e)}")
            self.metadata["download_status"] = "failed"
            self._save_metadata()
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model state."""
        info = {
            "model_id": self.model_id,
            "is_loaded": self.model is not None,
            "device": self.device_map,
            "use_4bit": self.use_4bit,
            "use_8bit": self.use_8bit,
            "cache_dir": self.cache_dir
        }
        
        if torch.cuda.is_available():
            info.update({
                "gpu_memory_allocated": torch.cuda.memory_allocated(),
                "gpu_memory_reserved": torch.cuda.memory_reserved()
            })
        
        return info
    
    def cleanup(self) -> None:
        """Clean up model resources."""
        try:
            if self.model is not None:
                self.model = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            self.tokenizer = None
            logger.info("Model resources cleaned up")
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")
    
    def clear_cache(self, keep_metadata: bool = True) -> None:
        """Clear the model cache directory."""
        try:
            if self.cache_dir.exists():
                # Save metadata if needed
                metadata = None
                if keep_metadata and self.metadata_file.exists():
                    with open(self.metadata_file, 'r') as f:
                        metadata = json.load(f)
                
                # Remove cache directory
                shutil.rmtree(self.cache_dir)
                self.cache_dir.mkdir(parents=True)
                
                # Restore metadata if needed
                if keep_metadata and metadata:
                    with open(self.metadata_file, 'w') as f:
                        json.dump(metadata, f, indent=2)
                
                logger.info("Cache cleared successfully")
        except Exception as e:
            logger.error(f"Error clearing cache: {str(e)}")
            raise 