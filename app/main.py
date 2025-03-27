from fastapi import FastAPI, HTTPException, Body, UploadFile, File
from pydantic import BaseModel
from contextlib import asynccontextmanager
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging
import os
import shutil
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
model_path = os.environ.get("MODEL_PATH", "/app/models")

# Load model on startup and release on shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Check if model exists locally, if not use a default model
    yield
    # Release resources on shutdown
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# Initialize FastAPI app
app = FastAPI(lifespan=lifespan)

def load_model_from_path(model_path):
    """Load a model from a local path"""
    global model, tokenizer
    
    try:
        logger.info(f"Loading model and tokenizer from {model_path}...")
        
        # Check for CUDA availability
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        
        # Load model with optimizations if on CUDA
        if device == "cuda":
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,  # Use mixed precision
                device_map="auto",           # Automatically determine device mapping
                low_cpu_mem_usage=True,      # Optimize for low CPU memory
                local_files_only=True
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True)
            model.to(device)
            
        logger.info("Model loaded successfully!")
        return True
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return False

def try_load_default_model():
    """Try to load a small default model if no custom model is available"""
    global model, tokenizer
    
    try:
        logger.info("Loading a small default model...")
        # Use a small open-source model that doesn't require authentication
        model_id = "gpt2"  # Small model that's freely available
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id)
        model.to(device)
        
        logger.info("Default model loaded successfully!")
        return True
    except Exception as e:
        logger.error(f"Error loading default model: {str(e)}")
        return False

@app.get("/")
async def root():
    return {"message": "LLM API is running. Send POST requests to /generate."}

@app.post("/load_model")
async def load_model(model_directory: str = Body(..., embed=True)):
    """Load a model from a specified directory path"""
    if not os.path.exists(model_directory):
        raise HTTPException(status_code=404, detail=f"Model directory not found: {model_directory}")
    
    success = load_model_from_path(model_directory)
    if success:
        return {"status": "success", "message": f"Model loaded from {model_directory}"}
    else:
        raise HTTPException(status_code=500, detail="Failed to load model")

@app.post("/upload_model")
async def upload_model(file: UploadFile = File(...)):
    """Upload a model file (for small models or model files)"""
    # Create models directory if it doesn't exist
    os.makedirs(model_path, exist_ok=True)
    
    # Save the uploaded file
    file_path = os.path.join(model_path, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    return {"status": "success", "message": f"Model file uploaded to {file_path}"}

@app.post("/generate", response_model=QueryResponse)
async def generate_text(request: QueryRequest = Body(...)):
    global model, tokenizer
    
    # Try to load a model if none is loaded
    if model is None or tokenizer is None:
        model_loaded = False
        
        # First try to load from the configured model path
        if os.path.exists(model_path):
            model_loaded = load_model_from_path(model_path)
        
        # If that fails, try to load a default model
        if not model_loaded:
            model_loaded = try_load_default_model()
            
        if not model_loaded:
            raise HTTPException(status_code=503, detail="No model is loaded and couldn't load a default model")
    
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

@app.get("/models")
async def list_models():
    """List all available models in the model directory"""
    if not os.path.exists(model_path):
        return {"models": []}
    
    models = [d for d in os.listdir(model_path) if os.path.isdir(os.path.join(model_path, d))]
    return {"models": models}

# Add health check endpoint
@app.get("/health")
async def health_check():
    if model is None or tokenizer is None:
        return {"status": "no model loaded", "device": "unknown"}
    return {"status": "healthy", "device": next(model.parameters()).device.type}

if __name__ == "__main__":
    import uvicorn
    # Get port from environment variable or use default
    port = int(os.environ.get("PORT", 8000))
    
    # Start the server
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)