#
#  Copyright 2014+ Carnegie Mellon University
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
import torch

class BatchObject:
    def __init__(self, query_ids, doc_ids, labels, cand_scores, features):
        """A simple wrapper to keep the batch data, which can be used to
           move data to a specific computation device.
        """
        qty = len(query_ids)
        self.query_ids = query_ids
        self.doc_ids = doc_ids
        self.features = features
        self.labels = labels
        self.cand_scores = cand_scores

        # Features potentially can be more complicated so we won't make a similar check here
        assert qty == len(self.doc_ids)
        assert qty == len(self.labels)
        assert qty == len(self.cand_scores)

    def __len__(self):
        return len(self.query_ids)

    def to(self, device_name):
        """Move featurs and labels to a given device."""
        self.labels = self.labels.to(device_name)
        self.cand_scores = self.cand_scores.to(device_name)
        self.features = self.move_features_to_device(self.features, device_name)

    def move_features_to_device(self, features, device_name):
        #
        # Based on the function from
        # https://github.com/bmitra-msft/TREC-Deep-Learning-Quick-Start,
        # which has an Apache2 compatible MIT license
        #
        if features is None:
            return None
        if isinstance(features, torch.Tensor):
            return features.to(device_name)
        if isinstance(features, tuple):
            return tuple(self.move_features_to_device(one_feat, device_name) for one_feat in features)
        if isinstance(features, list):
            return [self.move_features_to_device(one_feat, device_name) for one_feat in features]
        if isinstance(features, dict):
            return {k : self.move_features_to_device(one_feat, device_name) for k, one_feat in features.items()}

        raise Exception(f'Unsupported feature type: {type(features)}')

