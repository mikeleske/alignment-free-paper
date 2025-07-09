import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import pandas as pd
import numpy as np
import random
from tqdm import tqdm
from pathlib import Path

from typing import Tuple, List

import faiss

import umap

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

logger = logging.getLogger()

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)


def create_faiss_db(dim: int):
    """
    Creates a FAISS index for inner product similarity search.

    Parameters:
    ----------
    dim : int
        Dimensionality of the vectors to be stored in the index.

    Returns:
    -------
    faiss.Index
        A FAISS index configured for inner product similarity.
    """
    return faiss.index_factory(dim, "Flat", faiss.METRIC_INNER_PRODUCT)

def add_vectors_to_db(embeddings: np.ndarray, vdb) -> None:
    """
    Normalizes embeddings and adds them to the FAISS index.

    Parameters:
    ----------
    embeddings : np.ndarray
        Array of vectors to be added to the index.
    vdb : faiss.Index
        FAISS index to which vectors will be added.
    """
    faiss.normalize_L2(embeddings)
    vdb.add(embeddings)

def get_knn(emb: np.ndarray, vdb, k: int):
    """
    Performs k-nearest neighbor search using the FAISS index.

    Parameters:
    ----------
    emb : np.ndarray
        Query embeddings.
    vdb : faiss.Index
        FAISS index containing the reference embeddings.
    k : int
        Number of nearest neighbors to retrieve.

    Returns:
    -------
    tuple[np.ndarray, np.ndarray]
        Tuple of distances and indices of the k nearest neighbors.
    """
    faiss.normalize_L2(emb)
    return vdb.search(emb, k=k)

def filter_dataset(
    df: pd.DataFrame, 
    embeddings: np.ndarray, 
    bp_range: tuple | None
) -> pd.DataFrame:
    """
    Filters a DataFrame and its corresponding embeddings by sequence length.

    Parameters:
    ----------
    df : pd.DataFrame
        DataFrame containing sequence database (accessions, sequence, length, etc.
    embeddings : np.ndarray
        NumPy array of embeddings aligned with the DataFrame rows.
    bp_range : tuple or None
        A (min, max) tuple defining the lower and upper bounds for 'SeqLen'.
        If None, no filtering is applied.

    Returns:
    -------
    tuple[pd.DataFrame, np.ndarray]
        The filtered DataFrame and the corresponding filtered embeddings.
    """
    if bp_range:
        df = df[(df['SeqLen'] > bp_range[0]) & (df['SeqLen'] < bp_range[1])]

    embeddings = embeddings[df.index]
    df = df.reset_index(drop=True)
    return df, embeddings

def run_faiss(
    df: pd.DataFrame, 
    embeddings: np.ndarray, 
    min_species_count: int, 
    vdb: faiss.IndexFlat, 
    n_samples: int, 
    k: int
    ) -> list:
    """
    Run FAISS k-NN search on a filtered subset of sequences.

    Parameters:
    ----------
    df : pd.DataFrame
        DataFrame containing taxonomic and sequence metadata.
    embeddings : np.ndarray
        NumPy array of sequence embeddings.
    min_species_count : int
        Minimum number of sequences required per species to be included in the search.
    vdb : faiss.IndexFlat
        Prebuilt FAISS index containing all sequence embeddings.
    n_samples : int
        Number of sequences to sample as queries.
    k : int
        Number of nearest neighbors to retrieve from the FAISS index.

    Returns:
    -------
    list
        A list containing the FAISS results and the selected query indices.
    """
    # Ensure our query sequence embeddings are
    # 1. Not the only sequence for a species (i.e. at least one other sequence available to match against)
    # 2. Names species
    #
    species_counts = df['Species'].value_counts()
    species_filtered = species_counts[species_counts >= min_species_count].index.values
    species_filtered = [species for species in species_filtered if (species != 's__' and ' sp' not in species)]
    idxs = df[df['Species'].isin(species_filtered)].index

    # Select n_samples query_ids
    # We purposely select query sequences from the database.
    # During the classification stage, we remove the query from the results before calculating accuracy scores.
    query_ids = random.sample(sorted(idxs), n_samples)
    return [ get_knn(emb=embeddings[query_ids], vdb=vdb, k=k), query_ids ]


def process_faiss_results(faiss_results: tuple, df: pd.DataFrame) -> list:
    """
    Processes FAISS results to evaluate taxonomic accuracy at each taxonomic level.

    Parameters:
    ----------
    faiss_results : tuple
        A tuple containing FAISS search results: ((distances, indices), query_ids).
    df : pd.DataFrame
        DataFrame containing a 'Taxon' column with semicolon-separated taxonomic strings.

    Returns:
    -------
    list
        A list of counts indicating how many times the top-1 result matched the query
        at each taxonomic level (Phylum, Class, Order, Family, Genus, Species).
    """
    levels = ['Phylum', 'Class', 'Order', 'Family', 'Genus', 'Species']
    results = [0 for _ in levels]

    df_taxons = df['Taxon'].to_list()

    (query_results_dis, query_results_ann), query_ids = faiss_results

    for i, query_id in enumerate(query_ids):
        query_data = df_taxons[query_id].split('; ')
        
        similarities = query_results_dis[i]
        anns = query_results_ann[i]

        # We remove the query item from the results. This correspond to how the --self flag works in VSEARCH.
        mask = anns != query_id
        similarities = np.round(similarities[mask], 5)
        anns = anns[mask]

        # Keep all entries with the highest similarity
        max_sim = np.max(similarities)
        top_mask = similarities == max_sim
        similarities = similarities[top_mask]
        anns = anns[top_mask]

        top_taxons = [df_taxons[ann] for ann in anns]

        for il, level in enumerate(levels):
            query_level_name_full = '; '.join(query_data[:il+2])
            top_names_full = ['; '.join(taxon.split('; ')[:il+2]) for taxon in top_taxons]

            if query_level_name_full in top_names_full:
                results[il] += 1
    
    return results


def load_data(
    dataset: str, 
    data_path: Path, 
    emb_path: Path,
    bp_range: List[int]
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Loads and preprocesses a dataset and its associated embeddings.

    Args:
        dataset (str): Identifier for the dataset (e.g., 'GG2', 'GTDB').
        data_path (Path): Path to the dataset file (CSV/TSV).
        emb_path (Path): Path to the .npy file containing embeddings.
        bp_range (List[int]): List with [min_bp, max_bp] to filter sequences by length.

    Returns:
        Tuple[pd.DataFrame, np.ndarray]: Filtered dataframe and corresponding embeddings.
    """
    delimiter = '\t' if dataset == 'GG2' else ';'

    try:
        df = pd.read_csv(data_path, sep=delimiter)
    except Exception as e:
        raise ValueError(f"Failed to load dataset from {data_path}: {e}")

    try:
        embeddings = np.load(emb_path)
    except Exception as e:
        raise ValueError(f"Failed to load embeddings from {emb_path}: {e}")

    # Normalize column names
    if 'V3V4Len' in df.columns:
        df = df.rename(columns={
            'V3V4': 'Seq',
            'V3V4Len': "SeqLen",
        })

    if dataset == 'GTDB':
        df = df.rename(columns={'Taxa': 'Taxon'})
        df['Taxon'] = df['Taxon'].astype(str).str.replace('~', '; ')

    # Filter based on sequence length or other criteria
    df_filtered, embeddings_filtered = filter_dataset(
        df=df, 
        embeddings=embeddings, 
        bp_range=bp_range
    )

    df_filtered.reset_index(drop=True, inplace=True)

    return df_filtered, embeddings_filtered


def print_results(
    res: list, 
    df: pd.DataFrame, 
    min_count: int
) -> None:
    """
    Processes and prints classification accuracy at each taxonomic level.

    Args:
        res: List of FAISS result for multiple runs.
        df (pd.DataFrame): Reference dataframe with taxonomic labels.
        min_count (int): Minimum count threshold used in the classification pipeline.
    """ 
    levels = ['Phylum', 'Class', 'Order', 'Family', 'Genus', 'Species']  
    all_results = []
    
    for faiss_result in res:
        result = process_faiss_results(faiss_results=faiss_result, df=df)
        all_results.append(result)

    mean_results = np.round(np.array(all_results).mean(axis=0) / 10, 2)

    levels = ['Phylum', 'Class', 'Order', 'Family', 'Genus', 'Species']
    results_df  = pd.DataFrame([mean_results], columns=levels)

    logger.info(f'Classification results for min_count = {min_count}:\n{results_df .to_markdown()}')


def classify_with_default_embeddings(
    dataset: str,
    data_path: Path, 
    emb_path: Path,
    n_runs: int,
    n_samples: int,
    k: int,
    bp_range: List[int],
) -> None:
    """
    Runs KNN-based classification using default embeddings and evaluates accuracy
    across multiple runs and minimum species count thresholds.

    Args:
        dataset (str): Dataset identifier (e.g., 'GTDB', 'GG2').
        data_path (Path): Path to the input data file.
        emb_path (Path): Path to the .npy file with precomputed embeddings.
        n_runs (int): Number of repeated classification runs for averaging.
        n_samples (int): Number of random samples to classify per run.
        k (int): Number of nearest neighbors (K) to retrieve in FAISS.
        bp_range (List[int]): Base-pair length range for filtering sequences.
    """
    min_counts = [2, 3, 5]

    logger.info("Loading data...")
    df, embeddings = load_data(
        dataset=dataset,
        data_path=data_path,
        emb_path=emb_path,
        bp_range=bp_range
    )
    logger.info("Data loading completed.")

    for min_count in min_counts:
        logger.info(f"Evaluating with min_count = {min_count}")

        vector_dim = embeddings.shape[1]
        vdb = create_faiss_db(vector_dim)
        add_vectors_to_db(embeddings, vdb)

        random.seed(10)
        logger.info(f"Running {n_runs} classification runs...")

        results = []
        for _ in tqdm(range(n_runs), desc=f"min_count={min_count}"):
            faiss_results = run_faiss(
                df=df, 
                embeddings=embeddings, 
                min_species_count=min_count, 
                vdb=vdb, 
                n_samples=n_samples, 
                k=k
            )
            results.append(faiss_results)

        print_results(res=results, df=df, min_count=min_count)


def classify_with_umap_embeddings(
    dataset: str,
    data_path: Path, 
    emb_path: Path,
    n_runs: int,
    n_samples: int,
    k: int,
    bp_range: List[int],
) -> None:
    """
    Applies UMAP dimensionality reduction to embeddings and performs classification
    across multiple taxonomic levels and dimensionalities.

    Args:
        dataset (str): Dataset identifier (e.g., 'GTDB', 'GG2').
        data_path (Path): Path to the input data file.
        emb_path (Path): Path to the .npy file with precomputed embeddings.
        n_runs (int): Number of repeated classification runs for averaging.
        n_samples (int): Number of random samples to classify per run.
        k (int): Number of nearest neighbors (K) to retrieve in FAISS.
        bp_range (List[int]): Base-pair length range for filtering sequences.
    """
    min_counts = [2, 3, 5]
    umap_dims = [8, 16, 32, 64, 128, 256]

    for umap_dim in umap_dims:
        logger.info(f"=== UMAP Dimension: {umap_dim} ===")

        logger.info("Loading data...")
        df, embeddings = load_data(
            dataset=dataset,
            data_path=data_path,
            emb_path=emb_path,
            bp_range=bp_range
        )
        logger.info("Data loading complete.")

        logger.info("Applying UMAP transformation...")
        embeddings = umap.UMAP(
            random_state=42, 
            n_neighbors=15, 
            min_dist=0.25, 
            metric='braycurtis', 
            n_components=umap_dim
        ).fit_transform( embeddings )
        logger.info("UMAP transformation complete.")

        for min_count in min_counts:
            logger.info(f"Evaluating with min_count = {min_count}")

            vector_dim = embeddings.shape[1]
            vdb = create_faiss_db(vector_dim)
            add_vectors_to_db(embeddings, vdb)

            random.seed(10)
            logger.info(f"Running {n_runs} classification runs...")

            results = []
            for _ in tqdm(range(n_runs), desc=f"UMAP={umap_dim}, min_count={min_count}"):
                faiss_results = run_faiss(
                    df=df,
                    embeddings=embeddings,
                    min_species_count=min_count,
                    vdb=vdb,
                    n_samples=n_samples,
                    k=k
                )
                results.append(faiss_results)

            print_results(res=results, df=df, min_count=min_count)

def main():
    #
    # Persistent parameters
    #
    N_RUNS = 100
    N_SAMPLES = 1_000
    MODELS = ['DNABERT2', 'DNABERTS', 'NT2-500', 'Evo131K']

    #
    # Greengenes2 Full 16S - Default embeddings
    #
    DATASET = 'GG2'
    DATA_PATH = Path('../../Greengenes2/2024.09/df.2024.09.backbone.full-length.csv.gz')
    BASE_EMB_PATH = '../../Greengenes2/2024.09/embeddings/df.2024.09.backbone.full-length.MODEL.npy'
    BP_RANGE = [1450, 1550]
    K = 11

    for model in MODELS:
        logger.info("=" * 80)
        logger.info(
            f"Experiment: Greengenes2 Full16S | Default | Model: {model} | BP_RANGE: {BP_RANGE} | K: {K}"
        )

        emb_path = Path(BASE_EMB_PATH.replace("MODEL", model))

        if not emb_path.exists():
            logger.warning(f"Skipping model `{model}`: embedding file not found at {emb_path}")
            continue

        try:
            classify_with_default_embeddings(
                dataset=DATASET,
                data_path=DATA_PATH,
                emb_path=emb_path,
                bp_range=BP_RANGE,
                k=K,
                n_runs=N_RUNS,
                n_samples=N_SAMPLES,
            )
        except Exception as e:
            logger.error(f"Error during classification for model `{model}`: {e}", exc_info=True)

    #
    # Greengenes2 Full V3V4 - Default embeddings
    #
    DATASET = 'GG2'
    DATA_PATH = Path('../Greengenes2/2024.09/df.2024.09.backbone.full-length.V3V4.csv.gz')
    BASE_EMB_PATH = '../Greengenes2/2024.09/embeddings/df.2024.09.backbone.full-length.V3V4.MODEL.npy'
    BP_RANGE = [390, 440]
    K = 51

    for model in MODELS:
        logger.info("=" * 80)
        logger.info(
            f"Experiment: Greengenes2 V3V4 | Default | Model: {model} | BP_RANGE: {BP_RANGE} | K: {K}"
        )

        emb_path = Path(BASE_EMB_PATH.replace("MODEL", model))

        if not emb_path.exists():
            logger.warning(f"Skipping model `{model}`: embedding file not found at {emb_path}")
            continue

        try:
            classify_with_default_embeddings(
                dataset=DATASET,
                data_path=DATA_PATH,
                emb_path=emb_path,
                bp_range=BP_RANGE,
                k=K,
                n_runs=N_RUNS,
                n_samples=N_SAMPLES,
            )
        except Exception as e:
            logger.error(f"Error during classification for model `{model}`: {e}", exc_info=True)

    #
    # GTDB Full 16S - Default embeddings
    #
    DATASET = 'GTDB'
    DATA_PATH = Path('../GTDB/GTDB_r220-DNABERTS-COUNTS-IDList_1400_1600.csv.gz')
    BASE_EMB_PATH = '../GTDB/embeddings/GTDB_r220-MODEL-EMB-1400-1600.npy'
    BP_RANGE = [1450, 1550]
    K = 11

    for model in MODELS:
        logger.info("=" * 80)
        logger.info(
            f"Experiment: GTDB Full16S | Default | Model: {model} | BP_RANGE: {BP_RANGE} | K: {K}"
        )

        emb_path = Path(BASE_EMB_PATH.replace("MODEL", model))

        if not emb_path.exists():
            logger.warning(f"Skipping model `{model}`: embedding file not found at {emb_path}")
            continue

        try:
            classify_with_default_embeddings(
                dataset=DATASET,
                data_path=DATA_PATH,
                emb_path=emb_path,
                bp_range=BP_RANGE,
                k=K,
                n_runs=N_RUNS,
                n_samples=N_SAMPLES,
            )
        except Exception as e:
            logger.error(f"Error during classification for model `{model}`: {e}", exc_info=True)

    #
    # GTDB Full V3V4 - Default embeddings
    #
    DATASET = 'GTDB'
    DATA_PATH = Path('../GTDB/GTDB_r220-DNABERTS-COUNTS-IDList_390_440.csv.gz')
    BASE_EMB_PATH = '../GTDB/embeddings/GTDB_r220-MODEL-EMB-V3V4-390-440.npy'
    BP_RANGE = [390, 440]
    K = 51

    for model in MODELS:
        logger.info("=" * 80)
        logger.info(
            f"Experiment: GTDB V3V4 | Default | Model: {model} | BP_RANGE: {BP_RANGE} | K: {K}"
        )

        emb_path = Path(BASE_EMB_PATH.replace("MODEL", model))

        if not emb_path.exists():
            logger.warning(f"Skipping model `{model}`: embedding file not found at {emb_path}")
            continue

        try:
            classify_with_default_embeddings(
                dataset=DATASET,
                data_path=DATA_PATH,
                emb_path=emb_path,
                bp_range=BP_RANGE,
                k=K,
                n_runs=N_RUNS,
                n_samples=N_SAMPLES,
            )
        except Exception as e:
            logger.error(f"Error during classification for model `{model}`: {e}", exc_info=True)


    #
    # Greengenes2 Full 16S - UMAP transformation
    #
    DATASET = 'GG2'
    DATA_PATH = Path('../Greengenes2/2024.09/df.2024.09.backbone.full-length.csv.gz')
    BASE_EMB_PATH = '../Greengenes2/2024.09/embeddings/df.2024.09.backbone.full-length.MODEL.npy'
    BP_RANGE = [1450, 1550]
    K = 11

    for model in MODELS:
        logger.info("=" * 80)
        logger.info(
            f"Experiment: Greengenes2 Full16S | UMAP | Model: {model} | BP_RANGE: {BP_RANGE} | K: {K}"
        )

        emb_path = Path(BASE_EMB_PATH.replace("MODEL", model))

        if not emb_path.exists():
            logger.warning(f"Skipping model `{model}`: embedding file not found at {emb_path}")
            continue

        try:
            classify_with_umap_embeddings(
                dataset=DATASET,
                data_path=DATA_PATH,
                emb_path=emb_path,
                bp_range=BP_RANGE,
                k=K,
                n_runs=N_RUNS,
                n_samples=N_SAMPLES,
            )
        except Exception as e:
            logger.error(f"Error during classification for model `{model}`: {e}", exc_info=True)

    #
    # Greengenes2 Full V3V4 - UMAP transformation
    #
    DATASET = 'GG2'
    DATA_PATH = Path('../Greengenes2/2024.09/df.2024.09.backbone.full-length.V3V4.csv.gz')
    BASE_EMB_PATH = '../Greengenes2/2024.09/embeddings/df.2024.09.backbone.full-length.V3V4.MODEL.npy'
    BP_RANGE = [390, 440]
    K = 51

    for model in MODELS:
        logger.info("=" * 80)
        logger.info(
            f"Experiment: Greengenes2 V3V4 | UMAP | Model: {model} | BP_RANGE: {BP_RANGE} | K: {K}"
        )

        emb_path = Path(BASE_EMB_PATH.replace("MODEL", model))

        if not emb_path.exists():
            logger.warning(f"Skipping model `{model}`: embedding file not found at {emb_path}")
            continue

        try:
            classify_with_umap_embeddings(
                dataset=DATASET,
                data_path=DATA_PATH,
                emb_path=emb_path,
                bp_range=BP_RANGE,
                k=K,
                n_runs=N_RUNS,
                n_samples=N_SAMPLES,
            )
        except Exception as e:
            logger.error(f"Error during classification for model `{model}`: {e}", exc_info=True)

    #
    # GTDB Full 16S - UMAP transformation
    #
    DATASET = 'GTDB'
    DATA_PATH = Path('../GTDB/GTDB_r220-DNABERTS-COUNTS-IDList_1400_1600.csv.gz')
    BASE_EMB_PATH = '../GTDB/embeddings/GTDB_r220-MODEL-EMB-1400-1600.npy'
    BP_RANGE = [1450, 1550]
    K = 11

    for model in MODELS:
        logger.info("=" * 80)
        logger.info(
            f"Experiment: GTDB Full16S | UMAP | Model: {model} | BP_RANGE: {BP_RANGE} | K: {K}"
        )

        emb_path = Path(BASE_EMB_PATH.replace("MODEL", model))

        if not emb_path.exists():
            logger.warning(f"Skipping model `{model}`: embedding file not found at {emb_path}")
            continue

        try:
            classify_with_umap_embeddings(
                dataset=DATASET,
                data_path=DATA_PATH,
                emb_path=emb_path,
                bp_range=BP_RANGE,
                k=K,
                n_runs=N_RUNS,
                n_samples=N_SAMPLES,
            )
        except Exception as e:
            logger.error(f"Error during classification for model `{model}`: {e}", exc_info=True)

    #
    # GTDB Full V3V4 - UMAP transformation
    #
    DATASET = 'GTDB'
    DATA_PATH = Path('../GTDB/GTDB_r220-DNABERTS-COUNTS-IDList_390_440.csv.gz')
    BASE_EMB_PATH = '../GTDB/embeddings/GTDB_r220-MODEL-EMB-V3V4-390-440.npy'
    BP_RANGE = [390, 440]
    K = 51

    for model in MODELS:
        logger.info("=" * 80)
        logger.info(
            f"Experiment: GTDB V3V4 | UMAP | Model: {model} | BP_RANGE: {BP_RANGE} | K: {K}"
        )

        emb_path = Path(BASE_EMB_PATH.replace("MODEL", model))

        if not emb_path.exists():
            logger.warning(f"Skipping model `{model}`: embedding file not found at {emb_path}")
            continue

        try:
            classify_with_umap_embeddings(
                dataset=DATASET,
                data_path=DATA_PATH,
                emb_path=emb_path,
                bp_range=BP_RANGE,
                k=K,
                n_runs=N_RUNS,
                n_samples=N_SAMPLES,
            )
        except Exception as e:
            logger.error(f"Error during classification for model `{model}`: {e}", exc_info=True)

if __name__ == "__main__":
    main()