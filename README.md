# Llamindex-Projects

embed_model_id = 'sentence-transformers/all-MiniLM-L6-v2'

device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'

embed_model = HuggingFaceEmbeddings(
    model_name=r"../Models/all-MiniLM-L6-v2",
    model_kwargs={'device': device},
    encode_kwargs={'device': device, 'batch_size': 32}
)
