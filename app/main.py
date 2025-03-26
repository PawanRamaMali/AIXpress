from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel
from contextlib import asynccontextmanager
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging
import os
from typing import List, Optional

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define our models
class QueryRequest(BaseModel):
    prompt: str
    max_tokens: int = 100
    temperature: float = 0.7
    top_p: float = 0.9
    stop_sequences: Optional[List[str]] = None

class QueryResponse(BaseModel):
    response_text: str
    usage: dict

# Global variables for model and tokenizer
model = None
tokenizer = None

# Load model on startup and release on shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the model and tokenizer on startup
    load_llm()
    yield
    # Release resources on shutdown
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# Initialize FastAPI app
app = FastAPI(lifespan=lifespan)

def load_llm():
    """Load the Llama 3.2 model and tokenizer"""
    global model, tokenizer
    
    try:
        logger.info("Loading Llama 3.2 model and tokenizer...")
        
        # You can replace with the specific Llama 3.2 model you want to use
        model_id = "meta-llama/Meta-Llama-3.2-8B"
        
        # Check for CUDA availability
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        
        # Load model with optimizations if on CUDA
        if device == "cuda":
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16,  # Use mixed precision
                device_map="auto",           # Automatically determine device mapping
                low_cpu_mem_usage=True       # Optimize for low CPU memory
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(model_id)
            model.to(device)
            
        logger.info("Model loaded successfully!")
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise RuntimeError(f"Failed to load model: {str(e)}")

@app.get("/")
async def root():
    return {"message": "Llama 3.2 API is running. Send POST requests to /generate."}

@app.post("/generate", response_model=QueryResponse)
async def generate_text(request: QueryRequest = Body(...)):
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")
    
    try:
        # Prepare the inputs
        inputs = tokenizer(request.prompt, return_tensors="pt")
        
        # Move inputs to the same device as model
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Count input tokens
        input_token_count = inputs["input_ids"].shape[1]
        
        # Set generation parameters
        generation_config = {
            "max_new_tokens": request.max_tokens,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "do_sample": request.temperature > 0,
            "pad_token_id": tokenizer.eos_token_id
        }
        
        # Add stop sequences if provided
        if request.stop_sequences:
            stop_token_ids = [tokenizer.encode(seq, add_special_tokens=False) for seq in request.stop_sequences]
            flat_stop_token_ids = [item for sublist in stop_token_ids for item in sublist]
            generation_config["eos_token_id"] = flat_stop_token_ids
        
        # Generate text
        with torch.no_grad():
            output = model.generate(**inputs, **generation_config)
        
        # Decode the output, skipping the prompt tokens
        generated_tokens = output[0, inputs["input_ids"].shape[1]:]
        response_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        # Count output tokens
        output_token_count = len(generated_tokens)
        
        # Prepare the response
        usage = {
            "prompt_tokens": input_token_count,
            "completion_tokens": output_token_count,
            "total_tokens": input_token_count + output_token_count
        }
        
        return QueryResponse(response_text=response_text, usage=usage)
    
    except Exception as e:
        logger.error(f"Error generating text: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating text: {str(e)}")

# Add health check endpoint
@app.get("/health")
async def health_check():
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy", "model": "Llama 3.2", "device": next(model.parameters()).device.type}

if __name__ == "__main__":
    import uvicorn
    # Get port from environment variable or use default
    port = int(os.environ.get("PORT", 8000))
    
    # Start the server
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)