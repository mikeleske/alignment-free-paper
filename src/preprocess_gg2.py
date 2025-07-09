import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import pandas as pd

from utils_dataloader import parse_fasta_file_greengenes2
from utils_bio import get_region

from pathlib import Path

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')



def split_and_explode_to_columns(df: pd.DataFrame, column: str, delimiter: str = "; ") -> pd.DataFrame:
    """
    Splits a specified column in the DataFrame by a delimiter and creates multiple new columns.

    Args:
        df (pd.DataFrame): The input DataFrame.
        column (str): The name of the column to be split and exploded.
        delimiter (str): The delimiter to use for splitting the column. Defaults to "; ".

    Returns:
        pd.DataFrame: The DataFrame with the split column expanded into multiple new columns.
    """
    # Split the column into multiple columns using the delimiter
    split_cols = df[column].str.split(delimiter, expand=True)

    # Rename the new columns based on the original column name
    split_cols.columns = ["Domain", "Phylum", "Class", "Order", "Family", "Genus", "Species"]

    # Combine the original DataFrame with the new split columns
    df = pd.concat([df, split_cols], axis=1)
    
    return df


def print_stats(df):
    levels = ['Phylum', 'Class', 'Order', 'Family', 'Genus', 'Species']

    for level in levels:
        print('   ', level, len(df[level].unique()))

    print('    Total sequences', df.shape[0])



def preprocess_full16s(DATA_FILE: Path, TAXONOMY_FILE: Path, MIN_LENGTH: int, MAX_LENGTH: int) -> pd.DataFrame:

    #
    # Read the sequences and filter on relevant length
    #
    df = parse_fasta_file_greengenes2(
        file=DATA_FILE,
        min_length=MIN_LENGTH,
        max_length=MAX_LENGTH
    )

    #
    # Load taxonomy file
    #
    taxa_file = pd.read_csv(filepath_or_buffer=TAXONOMY_FILE, sep='\t')

    #
    # Merge taxaonomy into main dataframe
    #
    df = pd.merge(df, taxa_file, how='inner', left_on='ID', right_on='Feature ID')
    columns = ['ID', 'Taxon', 'SeqLen', 'Seq']
    df = df[columns]

    #
    # Calculate counts for ducplicates
    #
    df_counts = df.groupby(['Taxon', 'Seq'], as_index=False).agg({'ID': list})
    df_counts['Count'] = df_counts['ID'].apply(lambda x: len(x))
    df_counts = df_counts.rename(columns={'ID': 'ID_List'})

    df.reset_index(inplace=True)

    #
    # Remove duplicates
    #
    df = df.drop_duplicates( 
    subset = ['Taxon', 'Seq'], 
    keep = 'first')

    #
    # Merge dedup data with counts

    df = pd.merge(df, df_counts,  how='left', left_on=['Taxon', 'Seq'], right_on = ['Taxon', 'Seq'])
    df.drop(columns=['index'], inplace=True)

    #
    # Explode taxonomy string in separate columns
    #
    df = split_and_explode_to_columns(df, column='Taxon', delimiter='; ')
    columns = ["ID", "Taxon", "Domain", "Phylum", "Class", "Order", "Family", "Genus", "Species", "SeqLen", "Seq", "Count", "ID_List"]
    df = df[columns]

    return df



def preprocess_v3v4(DATA_FILE: Path, TAXONOMY_FILE: Path) -> pd.DataFrame:

    #
    # Read all sequences
    #
    df = parse_fasta_file_greengenes2(
        file=DATA_FILE,
    )

    #
    # Load taxonomy file
    #
    taxa_file = pd.read_csv(filepath_or_buffer=TAXONOMY_FILE, sep='\t')

    #
    # Merge taxaonomy into main dataframe
    #
    df = pd.merge(df, taxa_file, how='inner', left_on='ID', right_on='Feature ID')
    columns = ['ID', 'Taxon', 'SeqLen', 'Seq']
    df = df[columns]

    #
    # Extract V3V4 region
    #
    region = 'V3V4'
    df[region] = df['Seq'].apply(lambda x: get_region(region=region, seq=x))
    df = df[df[region] != 'ACGT']
    df['V3V4Len'] = df[region].apply(lambda x: len(x))

    #
    # Calculate counts for ducplicates
    #
    df_counts = df.groupby(['Taxon', region], as_index=False).agg({'ID': list})
    df_counts['counts'] = df_counts['ID'].apply(lambda x: len(x))
    df_counts = df_counts.rename(columns={'ID': 'ID_List'})

    df.reset_index(inplace=True)

    #
    # Remove duplicates
    #
    df = df.drop_duplicates( 
    subset = ['Taxon', region], 
    keep = 'first')

    #
    # Merge dedup data with counts
    #
    df = pd.merge(df, df_counts,  how='left', left_on=['Taxon', region], right_on = ['Taxon', region])
    df.drop(columns=['index'], inplace=True)

    #
    # Explode taxonomy string in separate columns
    #
    df = split_and_explode_to_columns(df, column='Taxon', delimiter='; ')
    columns = ["ID", "Taxon", "Domain", "Phylum", "Class", "Order", "Family", "Genus", "Species", "V3V4Len", "V3V4", "counts", "ID_List"]
    df = df[columns]

    return df



def main():
    DATA_FILE = Path('../data/raw/5b42d9b6-2f24-4f01-b989-9b4dafca7d5e/data/dna-sequences.fasta')
    TAXONOMY_FILE = Path('../data/raw/b7c3e691-ea51-4547-94dd-f79f49e41a36/data/taxonomy.tsv')

    # Preprocess Full 16S data
    logging.info(f'Preprocessing and deduplicating - Full16S')
    MIN_LENGTH = 1400
    MAX_LENGTH = 1600
    df = preprocess_full16s(DATA_FILE=DATA_FILE, TAXONOMY_FILE=TAXONOMY_FILE, MIN_LENGTH=MIN_LENGTH, MAX_LENGTH=MAX_LENGTH)
    print_stats(df)
    logging.info(f'Saving dataframe as gg2.2024.09.backbone.full-length.csv.gz')
    df.to_csv(Path('../data/gg2.2024.09.backbone.full-length.csv.gz'), sep='\t', index=False, compression='gzip')
    logging.info(f'Preprocessing and deduplicating - Full16S - Completed')


    # Preprocess V3V4 data
    logging.info(f'Preprocessing and deduplicating - V3V4')
    df = preprocess_v3v4(DATA_FILE=DATA_FILE, TAXONOMY_FILE=TAXONOMY_FILE)
    print_stats(df)
    logging.info(f'Saving dataframe as gg2.2024.09.backbone.full-length.V3V4.csv.gz')
    df.to_csv(Path('../data/gg2.2024.09.backbone.full-length.V3V4.csv.gz'), sep='\t', index=False, compression='gzip')
    logging.info(f'Preprocessing and deduplicating - V3V4 - Completed')


if __name__ == "__main__":
    main()