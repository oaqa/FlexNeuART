from flexneuart.data_augmentation.rule_based.data_augment import DataAugment
import random
import json
from flexneuart.data_augmentation import register_augmentation


@register_augmentation("query_cache")
class QueryCache(DataAugment):
    """
    Return a pre-computed similar query for the original query
    ...

    Attributes
    ----------
    cache_path : string 
        Path to the json file with the query cache

    Methods
    -------
    augment(text)
        returns the augmented text
    """
    def __init__(self, name, conf):
        super().__init__(name)
        try:
            cache_path = conf[self.augmentation_name]["cache_path"]
            self.types = "query"
        except:
            expected_config = {self.augmentation_name :
                               {"cache_path": "/path/to/query/cache"}}
            raise Exception("ERROR: Config file is missing parameters. Please ensure the following parameters exists - \n%s"
                            % json.dumps(expected_config, indent = 3))
        
        self.cache = self.__read_cache(cache_path)

    def __read_cache(self, cache_path):
        f = open(cache_path)
        cache = json.load(f)
        f.close()
        return cache

    def augment(self, text):
        if text not in self.cache:
            print("Not Found")
            print(text)
            return text
        
        if len(self.cache[text])==0:
            print("No Samples found")
        
        try:
            new_query = random.choice(self.cache[text])
        except:
            new_query = text

        print(text, new_query)

        return new_query

