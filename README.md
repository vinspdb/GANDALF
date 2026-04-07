# GANDALF: A LLM-based approach to map bark beetle outbreaks in semantic stories of Sentinel-2 images

**The repository contains code referred to the work:**

*Vincenzo Pasquadibisceglie, Vito Recchia, Annalisa Appice, Donato Malerba and Giuseppe Fiameni*


[*GANDALF: A LLM-based approach to map bark beetle outbreaks in semantic stories of Sentinel-2 images*](https://dl.acm.org/doi/10.1145/3672608.3707751)

Please cite our work if you find it useful for your research and work.

```
@inproceedings{10.1145/3672608.3707751,
author = {Pasquadibisceglie, Vincenzo and Recchia, Vito and Appice, Annalisa and Malerba, Donato and Fiameni, Giuseppe},
title = {GANDALF: A LLM-based approach to map bark beetle outbreaks in semantic stories of Sentinel-2 images},
year = {2025},
isbn = {9798400706295},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3672608.3707751},
doi = {10.1145/3672608.3707751},
booktitle = {Proceedings of the 40th ACM/SIGAPP Symposium on Applied Computing},
pages = {1074–1081},
numpages = {8},
location = {Catania International Airport, Catania, Italy},
series = {SAC '25}
}
```

# How to use:

Dataset Generation – please download the raw files from the link below, and then run the following command:

```
python -m csv_to_text
```
Domain Adaptation stage (training folder):

```
python -m mlm_mbert
```
Fine-Tuning stage (training folder):

```
python -m train_mbert
```

Link to the datasets:
https://mega.nz/folder/RqtnXJhR#AnBRLoGuiybszQUU4TqkZg
