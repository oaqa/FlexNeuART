## Configurable pipelines for IR-datasets

We support (somewhat experimentally) simple conversion pipelines for [ir-datasets](https://ir-datasets.com/).
They are processed using the script [configurable_convert.py](configurable_convert.py),
which is guided by a JSON configuration file. 
The output of this file is one or more JSONL files, 
which can be processed using standard FlexNeuART scripts.

**Note**: A small number of documents in Gov2 and ClueWeb12B are currently being lost, but the exact reason is not known (it is possible that there are parsing errors). The issue might be related to multiprocessing. So, for small collections like `Robust04` and `Cranfield` we recommend to do conversion using a single process.

We have (sample) configuration files for the following collections:

1. [Cranfield](sample_configs/cranfield.json)
2. [Robust04](sample_configs/trec_robust04.json)
3. [Gov2](sample_configs/gov2.json)
4. [ClueWeb12B](sample_configs/clueweb12-b13.json)
 
Although some datasets will be automatically downloaded by `ir-datasets`, 
many datasets are licensed and need to be `installed` manually.

The JSON configuration file contains an array of descriptions, each of which is supposed
to be applicable to a specific data set part or a split. 
The split is identified by the IR-dataset name (attribute `dataset_name`).
It has two crucial parameters:
1. a destination sub-folder in the output catalog (attribute `part_name`);
2. a flag specifyng whether it is a query set or not.
3. a list of input attributes (except query and document IDs), which we also call fields (parameter `src_attributes`).

Each pipeline stage accepts a number of attributes/fields and converts/processes them. 
It is achieved with a help of components: There is an array of component descriptions.
Components are applied to a set of input attributes one by one and they produce
output attributes for the next pipeline stage.

In the simplest case,
no data processing is happening and the attribute/field is renamed.
Another simple operation is concatenation of fields.

Crucially, there are no "fall through" attributes, which pass a pipeline stage without explicitly defined operation. 
If we need to pass some document attribute "as is", we need to use either a `copy` or `rename`
processing component.

