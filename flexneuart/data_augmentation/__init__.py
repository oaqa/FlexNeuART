import os
import importlib

from flexneuart.data_augmentation.utils.data_augment import DataAugmentTest

# decorator implementation inspired from Fairseq

AUGMENTATION_REGISTRY = {}
AUGMENTATION_CLASS_NAMES = set()

def register_augmentation(name):
    def register_task_cls(cls):
        if name in AUGMENTATION_REGISTRY:
            raise ValueError("Cannot register duplicate task ({})".format(name))
        if not issubclass(cls, DataAugmentTest):
            raise ValueError(
                "Augmentation Method ({}: {}) must extend DataAugment".format(name, cls.__name__)
            )
        if cls.__name__ in AUGMENTATION_CLASS_NAMES:
            raise ValueError(
                "Cannot register task with duplicate class name ({})".format(
                    cls.__name__
                )
            )
        AUGMENTATION_REGISTRY[name] = cls
        AUGMENTATION_CLASS_NAMES.add(cls.__name__)

        return cls

    return register_task_cls

def get_augmentation_method(name):
    return AUGMENTATION_REGISTRY[name]

def import_augment_methods(mod_dir, namespace):
    for file in os.listdir(mod_dir):
        path = os.path.join(mod_dir, file)
        if (
            not file.startswith("_")
            and not file.startswith(".")
            and (file.endswith(".py") or os.path.isdir(path))
        ):
            aug_name = file[: file.find(".py")] if file.endswith(".py") else file
            if "_test" in aug_name:
                continue
            importlib.import_module(namespace + "." + aug_name)

mod_dir = os.path.dirname(__file__)
import_augment_methods(os.path.join(mod_dir, "utils"), "flexneuart.data_augmentation.utils")