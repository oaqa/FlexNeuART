### Data utilities for NDRM models

These are utilities for NDRM models including utilities to generate embeddings, and IDFs for
the NDRM models:

[Mitra, B., Hofstatter, S., Zamani, H., & Craswell, N. (2020). Conformer-kernel with query term independence for document retrieval. 
arXiv preprint arXiv:2007.10434.](https://github.com/bmitra-msft/TREC-Deep-Learning-Quick-Start).

They require additional components, see [requirements_add.txt](requirements_add.txt).

### Caveats

1. It appears to be hard to train NDRM models using this framework. For this reason, one option is
to use [the original software]((https://github.com/bmitra-msft/TREC-Deep-Learning-Quick-Start)) and load the
resulting model.
2. To do so, we provide a [conversion script](convert_model.py)
3. Model needs to operate on the text processed by the Krovetz stemmer, you can use, e.g., the 
[following script](/scripts/data_convert/add_stemmed_field.py) to create a respective stemmed text field.
4. There are differences in the implementation so the resulting model will slightly underperform (by about 3.6%)
compared to the original one.