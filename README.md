# Alignment-free Bacterial Taxonomy Classification with Genomic Language Models

**\[Alignment-free Bacterial Taxonomy Classification with Genomic Language Models]**
Author(s): Mike Leske, Jamie A. FitzGerald, Keith Coughlan, Francesca Bottacini, Haithem Afli, Bruno Gabriel Nascimento Andrade
📄 [Link to Paper](https://www.biorxiv.org/content/10.1101/2025.06.27.662019v1)

---

## 📚 Overview

This repository accompanies our paper on leveraging **Genomic Language Models (gLM)** for **bacterial classification using 16S rRNA sequences**. We explore the feasibility of applying gLM embeddings to bioinformatics tasks traditionally dominated by alignment-based methods.

Our study demonstrates that:
~~~~
* gLM embeddings can classify bacterial sequences at the **species level** with accuracy **matching or exceeding** established tools like **BLAST+** and **VSEARCH**.
* The **cosine similarity** metric allows the compute-efficient sequence classification **orders of magnitude faster** than BLAST+ and VSEARCH, while still preserving **biological relevance**.
* Embeddings allow for the **detection of mislabeled sequences**, adding a layer of quality control.

---

## 🛠️ Installation

Instructions for setting up the environment (e.g., using `conda`, `pip`, or `Docker`):

```bash
git clone https://github.com/mikeleske/alignment-free-paper.git
cd alignment-free-paper
pip install -r requirements.txt
```

---

## 📂 Project Structure

```
alignment-free-paper/
├── data/               # To store raw data, preprocessed data, and embeddings
├── notebooks/          # Notebooks for paper figues
├── src/                # Source code
├── requirements.txt
└── README.md
```

---

## 🚀 Usage

1. Acquire data:

```bash
cd src
python download_sequence_files.py
python preprocess_gg2.py
python preprocess_gtdb.py
```

2. Generate embeddings (GPU required)

```bash
python embed.py
```

3. Perform classification runs

```bash
python classify.py
```

---

## 📎 Related Work / Citations

If you use this code, please cite:

```bibtex
@article {Leske2025.06.27.662019,
	author = {Leske, Mike and FitzGerald, Jamie A. and Coughlan, Keith and Bottacini, Francesca and Afli, Haithem and Andrade, Bruno Gabriel Nascimento},
	title = {Alignment-free Bacterial Taxonomy Classification with Genomic Language Models},
	elocation-id = {2025.06.27.662019},
	year = {2025},
	doi = {10.1101/2025.06.27.662019},
	publisher = {Cold Spring Harbor Laboratory},
	abstract = {Advances in natural language processing, including the ability to process long sequences, have paved the way for the development of Genomic Language Models (gLM). This study evaluates the feasibility of four models for bacterial classification using 16S rRNA sequences and demonstrates that gLM embeddings can be applied to effectively classify sequences at the species level, matching or outperforming the accuracy of established bioinformatics tools like BLAST+ and VSEARCH. We adopt cosine similarity as a computationally efficient metric, enabling classification orders of magnitude faster than current methods, and show that it carries biologically relevant signals. In addition, we demonstrate how sequence embeddings can be used to identify mislabeled sequences. Our findings place gLM embeddings as a promising alternative to traditional alignment-based methods, especially in large-scale applications such as metataxonomic assignments. Despite its wide potential, key challenges remain, including the sensitivity of embeddings to sequences of different lengths.Competing Interest StatementThe authors have declared no competing interest.European Commission, https://ror.org/00k4n6c32, 101182801},
	URL = {https://www.biorxiv.org/content/early/2025/06/28/2025.06.27.662019},
	eprint = {https://www.biorxiv.org/content/early/2025/06/28/2025.06.27.662019.full.pdf},
	journal = {bioRxiv}
}

```

