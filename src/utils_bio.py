from Bio.Seq import Seq
import re

primers = {
    '27F': 'AGAGTTTGATYMTGGCTCAG',
    '68F': 'TNANACATGCAAGTCGRRCG',
    '341F': 'CCTACGGGNGGCWGCAG',
    '515F': 'GTYCAGCMGCCGCGGTAA',
    '799F': 'AACMGGATTAGATACCCKG',
    '939F': 'GAATTGACGGGGGCCCGCACAAG',
    '967F': 'CAACGCGAAGAACCTTACC',
    '1115F': 'CAACGAGCGCAACCCT',
    '338R': 'GCTGCCTCCCGTAGGAGT',
    '518R': 'WTTACCGCGGCTGCTG',
    '534R': 'ATTACCGCGGCTGCTGG',
    '785R': 'GACTACHVGGGTATCTAATCC',
    '806R': 'GGACTACHVGGGTWTCTAAT',
    '926R': 'CCGYCAATTYMTTTRAGTTT',
    '944R': 'GAATTAAACCACATGCTC',
    '1193R': 'ACGTCATCCCCACCTTCC',
    '1378R': 'CGGTGTGTACAAGGCCCGGGAACG',
    '1391R': 'GACGGGCGGTGWGTRCA',
    '1492R': 'TACGGYTACCTTGTTACGACTT'
}

primers_regex = {
    '27F': 'AGAGTTTGAT[CT][AC]TGGCTCAG',
    '68F': 'T[ACGT]A[ACGT]ACATGCAAGTCG[AG][AG]CG',
    '341F': 'CCTACGGG[ACGT]GGC[AT]GCAG',
    '515F': 'GTG[CT]CAGC[AC]GCCGCGGTAA',
    '799F': 'AAC[AC]GGATTAGATACCC[GT]G',
    '939F': 'GAATTGACGGGGGCCCGCACAAG',
    '967F': 'CAACGCGAAGAACCTTACC',
    '1115F': 'CAACGAGCGCAACCCT',
    '338R': 'GCTGCCTCCCGTAGGAGT',
    '518R': '[AT]TTACCGCGGCTGCTG',
    '534R': 'ATTACCGCGGCTGCTGG',
    '785R': 'GACTAC[ACT][GCA]GGGTATCTAATCC',
    '806R': 'GGACTAC[ACT][GCA]GGGT[AT]TCTAAT',
    '926R': 'CCG[CT]CAATT[CT][AC]TTT[AG]AGTTT',
    '944R': 'GAATTAAACCACATGCTC',
    '1193R': 'ACGTCATCCCCACCTTCC',
    '1378R': 'CGGTGTGTACAAGGCCCGGGAACG',
    '1391R': 'GACGGGCGGTG[AT]GT[AG]CA',
    '1492R': 'TACGG[CT]TACCTTGTTACGACTT'
}

def get_primers(start:str = None, end:str = None, seq = None):
    f_primer = None
    r_primer = None
    rev_seq = str(Seq(seq).reverse_complement())
    try:
        f_primer = re.findall(primers_regex[start], seq)[0]
    except:
        pass
    try:
        r_primer = re.findall(primers_regex[end], rev_seq)[0]
    except:
        pass
    return (f_primer, r_primer)

def get_region(region:str = None, seq:str = None):
    if region == 'V1V2':
        f_primer, r_primer = get_primers(start='27F', end='338R', seq=seq)
    if region == 'V1V3':
        f_primer, r_primer = get_primers(start='27F', end='534R', seq=seq)
    elif region == 'V3V4':
        f_primer, r_primer = get_primers(start='341F', end='785R', seq=seq)
    elif region == 'V4':
        f_primer, r_primer = get_primers(start='515F', end='806R', seq=seq)
    elif region == 'V4V5':
        f_primer, r_primer = get_primers(start='515F', end='944R', seq=seq)
    elif region == 'V6V8':
        f_primer, r_primer = get_primers(start='939F', end='1378R', seq=seq)
    elif region == 'V7V9':
        f_primer, r_primer = get_primers(start='1115F', end='1492R', seq=seq)
    elif region == 'V1V8':
        f_primer, r_primer = get_primers(start='27F', end='1378R', seq=seq)
    elif region == 'V1V9':
        f_primer, r_primer = get_primers(start='27F', end='1492R', seq=seq)

    try:
        return str(Seq(str(Seq(seq.split(f_primer)[1]).reverse_complement()).split(r_primer)[1]).reverse_complement())
    except:
        return str('ACGT')