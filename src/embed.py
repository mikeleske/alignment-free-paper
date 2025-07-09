import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path

from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM, AutoModelForCausalLM, AutoConfig
from torch import nn


class CustomEmbedding(nn.Module):
  def unembed(self, u):
    return u
  

def load_model_from_hf(model_name: str, maskedlm: bool = False, causallm: bool = False):
    """
    Loads a model and tokenizer from Hugging Face.

    Args:
        model_name (str): The model name or path in Hugging Face Hub.

    Returns:
        tuple: A tuple containing the loaded model and tokenizer.

    Raises:
        ValueError: If the model name is invalid or loading fails.
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        if maskedlm:
            model = AutoModelForMaskedLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                device_map={"": 0}
            )
        elif causallm:
            model_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True, revision="1.1_fix")
            model_config.use_cache = True
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                config=model_config,
                trust_remote_code=True,
                revision="1.1_fix",
                torch_dtype=torch.float16,
            )
        else:
            model = AutoModel.from_pretrained(
                model_name,
                trust_remote_code=True,
                device_map={"": 0}
            )
        return model, tokenizer
    except Exception as e:
        raise ValueError(f"Failed to load model/tokenizer: {e}")


def get_emb(model, tokenizer, seq: str, device: str) -> torch.Tensor:
    """
    Generates an embedding for a DNA sequence using a model with mean pooling.

    Args:
        model (torch.nn.Module): The model.
        tokenizer (PreTrainedTokenizer): The tokenizer corresponding to the model.
        seq (str): The DNA sequence to embed.
        device (str): Device to use ('cuda' or 'cpu'). Defaults to 'cuda'.

    Returns:
        torch.Tensor: Mean-pooled embedding of the input DNA sequence.

    Raises:
        ValueError: If the input sequence is invalid.
        RuntimeError: If CUDA is selected but unavailable.
    """
    if not isinstance(seq, str):
        raise ValueError("Input sequence must be a string.")

    device = torch.device(device)
    
    try:
        inputs = tokenizer(seq, return_tensors="pt")["input_ids"].to(device)
        with torch.no_grad():
            hidden_states = model(inputs)[0]
        embedding_mean = torch.mean(hidden_states[0], dim=0).detach().cpu()
        return embedding_mean
    except Exception as e:
        raise RuntimeError(f"Error during embedding generation: {e}")


def get_emb_nt(model, tokenizer, seq: str, device: str = "cuda") -> torch.Tensor:
    """
    Generates an embedding for a DNA sequence using a model with mean pooling.

    Args:
        model (torch.nn.Module): The model.
        tokenizer (PreTrainedTokenizer): The tokenizer corresponding to the model.
        seq (str): The DNA sequence to embed.
        device (str): Device to use ('cuda' or 'cpu'). Defaults to 'cuda'.

    Returns:
        torch.Tensor: Mean-pooled embedding of the input DNA sequence.

    Raises:
        ValueError: If the input sequence is invalid.
        RuntimeError: If CUDA is selected but unavailable.
    """
    if not isinstance(seq, str):
        raise ValueError("Input sequence must be a string.")

    device = torch.device(device)
    
    try:
        inputs = tokenizer(seq, return_tensors="pt")["input_ids"].to(device)
        attention_mask = inputs != tokenizer.pad_token_id
        torch_outs = model(
            inputs,
            attention_mask=attention_mask,
            encoder_attention_mask=attention_mask,
            output_hidden_states=True
        )
        embeddings = torch_outs['hidden_states'][-1].detach()#.numpy()
        attention_mask = torch.unsqueeze(attention_mask, dim=-1)
        mean_sequence_embeddings = (torch.sum(attention_mask*embeddings, axis=-2)/torch.sum(attention_mask, axis=1)).cpu().numpy()
        return mean_sequence_embeddings
    except Exception as e:
        raise RuntimeError(f"Error during embedding generation: {e}")


def get_emb_evo(model, tokenizer, seq: str, device: str = "cuda") -> np.ndarray:
    """
    Generates an embedding for a DNA sequence using a model with mean pooling.

    Args:
        model (torch.nn.Module): The model.
        tokenizer (PreTrainedTokenizer): The tokenizer corresponding to the model.
        seq (str): The DNA sequence to embed.
        device (str): Device to use ('cuda' or 'cpu'). Defaults to 'cuda'.

    Returns:
        torch.Tensor: Mean-pooled embedding of the input DNA sequence.

    Raises:
        ValueError: If the input sequence is invalid.
        RuntimeError: If CUDA is selected but unavailable.
    """
    if not isinstance(seq, str):
        raise ValueError("Input sequence must be a string.")

    device = torch.device(device)

    try:
        inputs = tokenizer(seq, return_tensors="pt")["input_ids"].to(device)
        embed = model(inputs)
        mean_sequence_embedding = embed.logits.mean(dim=1).cpu().detach().float().numpy()
        return mean_sequence_embedding
    except Exception as e:
        raise RuntimeError(f"Error during embedding generation: {e}")
 

def drop_duplicate_sequences(df: pd.DataFrame, column: str = "Seq") -> pd.DataFrame:
    """
    Removes duplicate sequences from a DataFrame based on 'Taxa' and a specified column.

    Args:
        df (pd.DataFrame): The DataFrame containing sequence data.
        column (str): Column name to consider for duplicates. Defaults to 'Seq'.

    Returns:
        pd.DataFrame: DataFrame with duplicates removed.

    Raises:
        ValueError: If required columns are missing in the DataFrame.
    """
    if column not in df.columns or "Taxa" not in df.columns:
        raise ValueError(f"DataFrame must contain columns 'Taxa' and '{column}'.")

    return df.drop_duplicates(subset=["Taxa", column], keep="first").reset_index(drop=True)


def vectorize(
    model_name,
    model,
    tokenizer,
    df: pd.DataFrame,
    column: str = "Seq",
    embeddings_numpy_file: str = None,
    batch_size: int = 10000
) -> None:
    """
    Generate embeddings for sequences in a DataFrame column and save to a NumPy file.

    Args:
        model (torch.nn.Module): The model used for generating embeddings.
        tokenizer (PreTrainedTokenizer): The tokenizer corresponding to the model.
        df (pd.DataFrame): The DataFrame containing sequences.
        column (str): The column containing sequences. Defaults to 'Seq'.
        embeddings_numpy_file (str): The file path for saving embeddings.
        batch_size (int): The number of vectors to process in each batch. Defaults to 10,000.

    Raises:
        ValueError: If required parameters are missing or column doesn't exist in DataFrame.
        RuntimeError: If embeddings saving/loading fails.
    """
    if embeddings_numpy_file is None:
        raise ValueError("embeddings_numpy_file cannot be None.")
    
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame.")
    
    embeddings_numpy_file = embeddings_numpy_file
    vectors = []
    embeddings = None

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device={device}')

    try:
        # Process rows in batches
        for i in tqdm(range(df.shape[0]), desc="Processing sequences"):
            if "DNABERT" in model_name:
                emb = get_emb(model, tokenizer, df.loc[i, column], device)
            elif "NT" in model_name:
                emb = get_emb_nt(model, tokenizer, df.loc[i, column], device)
            elif "Evo" in model_name:
                emb = get_emb_evo(model, tokenizer, df.loc[i, column], device)
            vectors.append(emb.reshape(1, -1))

            # Save batch to file when reaching batch_size
            if len(vectors) >= batch_size:
                vectors = np.vstack(vectors)
                embeddings = _save_embeddings_batch(vectors, embeddings_numpy_file)
                vectors = []

        # Save any remaining vectors after processing all rows
        if vectors:
            vectors = np.vstack(vectors)
            embeddings = _save_embeddings_batch(vectors, embeddings_numpy_file)

        print(f"Final embeddings shape: {embeddings.shape if embeddings is not None else (0,)}")
    except Exception as e:
        raise RuntimeError(f"Error during vectorization: {e}")


def _save_embeddings_batch(vectors: np.ndarray, file_path: str) -> np.ndarray:
    """
    Save a batch of embeddings to a NumPy file, appending if the file already exists.

    Args:
        vectors (np.ndarray): Batch of vectors to save.
        file_path (str): Path to the NumPy file.

    Returns:
        np.ndarray: The combined embeddings.
    """
    try:
        if _file_exists(file_path):
            existing_embeddings = np.load(file_path)
            embeddings = np.concatenate((existing_embeddings, vectors), axis=0)
        else:
            embeddings = vectors
        np.save(file_path, embeddings)
        return embeddings
    except Exception as e:
        raise RuntimeError(f"Failed to save embeddings to {file_path}: {e}")


def _file_exists(file_path: str) -> bool:
    """
    Check if a file exists.

    Args:
        file_path (str): Path to the file.

    Returns:
        bool: True if the file exists, False otherwise.
    """
    try:
        with open(file_path, "rb"):
            return True
    except FileNotFoundError:
        return False


def main() -> None:


    MODELS = {
        'DNABERT2': 'zhihan1996/DNABERT-2-117M', 
        'DNABERTS': 'zhihan1996/DNABERT-S',
        'NT2-500': 'InstaDeepAI/nucleotide-transformer-v2-500m-multi-species', 
        'Evo131K': 'togethercomputer/evo-1-131k-base'
    }
    SEQ_FILES = [
        '../data/gg2.2024.09.backbone.full-length.csv.gz',
        '../data/gg2.2024.09.backbone.full-length.V3V4.csv.gz',
        '../data/gtdb.r220.full-length.csv.gz',
        '../data/gtdb.r220.full-length.V3V4.csv.gz'
    ]

    for model_name, model_id in MODELS.items():
        print(f'Loading model {model_id}')
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        if 'DNABERT' in model_name:
            model, tokenizer = load_model_from_hf(model_id, maskedlm=False, causallm=False)
            model.to(device)
            model.eval()
        elif 'NT' in model_name:
            model, tokenizer = load_model_from_hf(model_id, maskedlm=True, causallm=False)
            model.to(device)
            model.eval()
        elif 'Evo' in model_name:
            model, tokenizer = load_model_from_hf(model_id, maskedlm=False, causallm=True)
            model.to(device)
            model.eval()
            model.backbone.unembed = CustomEmbedding()


        for seq_file in SEQ_FILES:
            print(f'Loading sequence file {seq_file}')
            df = pd.read_csv(seq_file, sep='\t')
            embed_column = 'V3V4' if 'V3V4' in seq_file else 'Seq'
            emb_file_path = Path('../data/embeddings') / seq_file.replace('csv.gz', model_name + '.npy')
            print(emb_file_path)

            vectorize(
                model_name=model_name,
                model=model, 
                tokenizer=tokenizer, 
                df=df, 
                column=embed_column, 
                embeddings_numpy_file=emb_file_path
            )


# --------------------------------------------------
if __name__ == '__main__':
    main()
