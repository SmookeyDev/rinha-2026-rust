# tools/

Scripts Python pra regenerar o `ivf_int16.bin` que vai embutido na imagem
Docker. Não são executados em runtime, só durante o build do índice.

## Pré-requisitos

```bash
pip install numpy faiss-cpu orjson
```

## Pipeline

A partir do `references.json.gz` do repo oficial:

```bash
gunzip -k references.json.gz                      # produz references.json (~284 MB)
python preprocess.py references.json              # references_vec.npy + references_lbl.npy
python build_index.py references_vec.npy references_lbl.npy -o ivf_int16.bin
```

## Validação

```bash
python normalize.py                               # checa as 14 dimensões contra os exemplos do doc
python verify_knn.py references_vec.npy references_lbl.npy /path/to/test-data.json
```

`verify_knn.py` precisa bater 100.0000% (54100/54100). Caso contrário, a
normalização divergiu do gerador C oficial.
