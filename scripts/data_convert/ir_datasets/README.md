## Configurable pipelines for IR-datasets

We support (somewhat experimentally) simple conversion pipelines for [ir-datasets](https://ir-datasets.com/).
They are processed using the script [configurable_convert.py](configurable_convert.py)
using a JSON configuration file. There are examples
for [Cranfield](sample_configs/cranfield.json) and [ClueWeb12B](sample_configs/clueweb12-b13.json).

The JSON configuration file contains an array of descriptions, each of which is supposed
to be applicable to a specific data set part or a split. The split is identified
by the IR-dataset name (attribute `dataset_name`)
and it has a destination sub-folder in the output catalog (attribute `part_name`).
You also have to specify whether it is a query set or not.

A pipeline can use one more attributes from an IR-dataset object. 
One has to explicitly specify all input attributes (except query and document IDs)
using `src_attributes`. 

Each pipeline stage accepts a number of attributes/fields and converts them. 
It is achieved with a help of components: There is an array of component descriptions.
Components are applied to a set of input attributes one by one and they produce
output attributes for the next pipeline stage.


In the simplest case,
no data processing is happening and the attribute/field is renamed.
Another simple operation is concatenation of fields.
Crucially, there are no "fall through" attributes. 
If we need to pass some document attribute "as is", we need to use either a `copy` or `rename`
processing component.

