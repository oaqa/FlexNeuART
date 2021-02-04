"""
FlexNeuART basic setup
"""

import glob
import os

def configure_classpath(source_root):
    """Add the latest FlexNeuART jar to the path.

       This function is based on Pyserini code https://github.com/castorini/pyserini

    :param source_root: source root
    """

    from jnius_config import set_classpath

    paths = glob.glob(os.path.join(source_root, 'FlexNeuART-*-fatjar.jar'))
    if not paths:
        raise Exception('No matching jar file found in {}'.format(os.path.abspath(source_root)))

    latest = max(paths, key=os.path.getctime)
    set_classpath(latest)


def create_featextr_resource_manager(fwd_index_dir=None, model1_root_dir=None, root_embed_dir=None):
    """Create a resource manager use for feature extraction and re-ranking.

    :param fwd_index_dir:     a forward index root (optional)
    :param model1_root_dir:   a Model 1 translation data root (optional)
    :param root_embed_dir:    a root word embedding directory (optional)

    :return: a references to the FeatExtrResourceManager class
    """
    # Do not import this at the top of the module, because it starts a JVM
    # and JVM can be started only once and it should be started before the
    # configure_classpath call
    from jnius import autoclass

    JFeatExtrResourceManager = autoclass('edu.cmu.lti.oaqa.flexneuart.letor.FeatExtrResourceManager')

    return JFeatExtrResourceManager(fwd_index_dir, model1_root_dir, root_embed_dir)