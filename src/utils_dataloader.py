import gzip

import pandas as pd
from Bio import SeqIO


def parse_fasta_file_greengenes2(
    file: str, min_length: int = 0, max_length: int = 10_000_000
) -> pd.DataFrame:
    """
    Read Greengenes2 gzip fasta file
    """

    rows_list = []
    columns = ["ID", "SeqLen", "Seq"]

    with open(file, "rt") as handle:
        for rec in SeqIO.parse(handle, "fasta"):
            seqlen = len(str(rec.seq))
            if seqlen > min_length and seqlen < max_length:
                rows_list.append(
                    {
                        "ID": str(rec.description).split()[0],
                        "SeqLen": len(rec.seq),
                        "Seq": str(rec.seq),
                    }
                )

    df = pd.DataFrame(rows_list, columns=columns)
    return df


def parse_fasta_file_gtdb_gzip(
    file: str, domain: str, min_seq_length: int = None, max_seq_length: int = 10_000_000
) -> pd.DataFrame:
    """
    Read GTDB gzip fasta file
    """

    rows_list = []
    columns = [
        "ID",
        "Taxon",
        "Kingdom",
        "Phylum",
        "Class",
        "Order",
        "Family",
        "Genus",
        "Species",
        "SeqLen",
        "Seq",
    ]

    with gzip.open(file, "rt") as handle:
        for rec in SeqIO.parse(handle, "fasta"):
            if str(rec.description).split()[1].startswith(domain):
                description = str(rec.description)
                id = description.split()[0]
                taxon = description.rsplit(id)[1].split(" [")[0].strip()
                taxon_split = taxon.split(";")

                if len(taxon_split) == 7:
                    kingdom, phylum, _class, order, family, genus, species = taxon_split
                    seq = str(rec.seq)
                    seq_len = len(seq)

                    if max_seq_length:
                        seq = seq[:max_seq_length]

                    dict1 = dict(
                        (col, val)
                        for (col, val) in zip(
                            columns,
                            [
                                id,
                                taxon,
                                kingdom,
                                phylum,
                                _class,
                                order,
                                family,
                                genus,
                                species,
                                seq_len,
                                seq,
                            ],
                        )
                    )
                    rows_list.append(dict1)
                else:
                    print(taxon)

    df = pd.DataFrame(rows_list, columns=columns)

    if min_seq_length:
        df = df[df["SeqLen"] >= min_seq_length].reset_index(drop=True)

    return df
