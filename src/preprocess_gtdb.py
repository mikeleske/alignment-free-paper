import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import pandas as pd

from utils_dataloader import parse_fasta_file_gtdb_gzip
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


def preprocess_full16s(DATA_FILE: Path, MIN_LENGTH: int, MAX_LENGTH: int) -> pd.DataFrame:

    #
    # Read the sequences and filter on relevant length
    #
    df = parse_fasta_file_gtdb_gzip(
        file=DATA_FILE, 
        domain='d__Bacteria'
    )

    #
    # Only keep certain bp lengths
    #
    df = df[(df['SeqLen'] >= MIN_LENGTH) & (df['SeqLen'] <= MAX_LENGTH)]

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

    return df



def preprocess_v3v4(DATA_FILE: Path) -> pd.DataFrame:

    #
    # Read the sequences and filter on relevant length
    #
    df = parse_fasta_file_gtdb_gzip(
        file=DATA_FILE, 
        domain='d__Bacteria'
    )

    #
    # Extract V3V4 region
    #
    region = 'V3V4'
    df[region] = df['Seq'].apply(lambda x: get_region(region=region, seq=x))
    df = df[df[region] != 'ACGT']
    df['V3V4Len'] = df[region].apply(lambda x: len(x))

    #
    # Only keep certain bp lengths
    #
    df = df[(df['V3V4Len'] >= 390) & (df['V3V4Len'] <= 440)]

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

    df = df.reset_index(drop=True)

    return df



def main():
    DATA_FILE = Path(Path('../data/raw/ssu_all_r220.fna.gz'))

    # Preprocess Full 16S data
    logging.info(f'Preprocessing and deduplicating - Full16S')
    MIN_LENGTH = 1400
    MAX_LENGTH = 1600
    df = preprocess_full16s(DATA_FILE=DATA_FILE, MIN_LENGTH=MIN_LENGTH, MAX_LENGTH=MAX_LENGTH)
    print_stats(df)
    logging.info(f'Saving dataframe as gtdb.r220.full-length.csv.gz')
    df.to_csv(Path('../data/gtdb.r220.full-length.csv.gz'), sep=';', index=False, compression='gzip')
    logging.info(f'Preprocessing and deduplicating - Full16S - Completed')


    # Preprocess V3V4 data
    logging.info(f'Preprocessing and deduplicating - V3V4')
    df = preprocess_v3v4(DATA_FILE=DATA_FILE)
    print_stats(df)
    logging.info(f'Saving dataframe as gtdb.r220.full-length.V3V4.csv.gz')
    df.to_csv(Path('../data/gtdb.r220.full-length.V3V4.csv.gz'), sep=';', index=False, compression='gzip')
    logging.info(f'Preprocessing and deduplicating - V3V4 - Completed')


if __name__ == "__main__":
    main()