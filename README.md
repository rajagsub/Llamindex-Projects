# Llamindex-Projects

embed_model_id = 'sentence-transformers/all-MiniLM-L6-v2'

device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'

embed_model = HuggingFaceEmbeddings(
    model_name=r"../Models/all-MiniLM-L6-v2",
    model_kwargs={'device': device},
    encode_kwargs={'device': device, 'batch_size': 32}
)

from sentence_transformers import HuggingFaceEmbeddings
from pathlib import Path

# Replace 'path/to/all-MiniLM-L6-v2' with the actual path to your downloaded model
model_path = 'path/to/all-MiniLM-L6-v2'

# Ensure the model path is a pathlib.Path object
model_path = Path(model_path)

# Create an instance of HuggingFaceEmbeddings and load the model from the local path
embed_model = HuggingFaceEmbeddings(model_name=str(model_path), model_kwargs={'device': 'cuda' if cuda.is_available() else 'cpu'}, encode_kwargs={'device': 'cuda' if cuda.is_available() else 'cpu', 'batch_size': 32})

