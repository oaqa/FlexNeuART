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
def create_featextr_resource_manager(resource_root_dir,
                                     fwd_index_dir=None, model1_root_dir=None, embed_root_dir=None):
    """Create a resource manager use for feature extraction and re-ranking. Note that
       all paths are relative to the collection root.

    :param resource_root_dir:   a collection/resource root directory
    :param fwd_index_dir:       a forward index root  (optional)
    :param model1_root_dir:     a Model 1 (giza) translation data root (optional)
    :param embed_root_dir:      a root word embedding directory (optional)

    :return: a references to the FeatExtrResourceManager class
    """
    # Do not import this at the top of the module, because it starts a JVM
    # and JVM can be started only once and it should be started before the
    # configure_classpath call
    from jnius import autoclass

    JResourceManager = autoclass('edu.cmu.lti.oaqa.flexneuart.resources.ResourceManager')

    return JResourceManager(resource_root_dir, fwd_index_dir, model1_root_dir, embed_root_dir)

