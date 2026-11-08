"""ZIP-backed native detection/SGG augmentation for a future clean release.

Uses pinned source definitions without importing Lightning/data extraction code.
The detection recipe includes random cropping; the SGG recipe does not.
"""
import ast
import importlib
from pathlib import Path
import sys
import types

import torch


def native_components(root, task, split):
    assert task in ('detection', 'sgg')
    vendor = Path(root) / 'vendor/salience_sgg'
    sys.path.insert(0, str(vendor))
    if 'datasets' not in sys.modules:
        package = types.ModuleType('datasets')
        package.__path__ = [str(vendor / 'datasets')]
        sys.modules['datasets'] = package
    transforms = importlib.import_module('datasets.transforms')
    coco_path = vendor / 'datasets/coco.py'
    tree = ast.parse(coco_path.read_text())
    converter = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                     and node.name == 'ConvertCocoPolysToMask')
    namespace = {'torch': torch, 'T': transforms}
    exec(compile(ast.Module(body=[converter], type_ignores=[]), str(coco_path), 'exec'), namespace)
    prepare = namespace['ConvertCocoPolysToMask'](False)
    recipe_path = coco_path if task == 'detection' else vendor / 'datasets/visual_genome.py'
    recipe_tree = ast.parse(recipe_path.read_text())
    class_name = 'CocoDetectionDataset' if task == 'detection' else 'VGDataset'
    recipe_class = next(node for node in recipe_tree.body if isinstance(node, ast.ClassDef)
                        and node.name == class_name)
    method = next(node for node in recipe_class.body if isinstance(node, ast.FunctionDef)
                  and node.name == 'make_transform')
    # Compile the unmodified static method inside a minimal class, not its loader.
    isolated = ast.ClassDef(name='PinnedRecipe', bases=[], keywords=[], body=[method], decorator_list=[])
    isolated = ast.fix_missing_locations(isolated)
    exec(compile(ast.Module(body=[isolated], type_ignores=[]), str(recipe_path), 'exec'), namespace)
    return prepare, namespace['PinnedRecipe'].make_transform(split)


class VGCanonicalTraining:
    def __init__(self, canonical_dataset, root, task='detection', queries=200):
        assert task in ('detection', 'sgg')
        self.canonical = canonical_dataset
        self.root = str(root)
        self.task = task
        self.queries = queries
        self._runtime = None

    def __len__(self):
        return len(self.canonical)

    def __getitem__(self, index):
        if self._runtime is None:
            self._runtime = native_components(self.root, self.task, self.canonical.split)
        prepare, transform = self._runtime
        sample = self.canonical[index]  # asserts graph object order survives COCO preparation
        image = self.canonical.image(index)
        image, target = prepare(image, {'image_id': sample.image_id,
                                       'annotations': self.canonical.annotations[sample.image_id]})
        if self.task == 'sgg':
            target['labels'] -= 1
            count = len(target['labels'])
            if count > self.queries:
                raise ValueError('sgg_query_capacity_requires_registered_policy')
            relation = torch.zeros(count, count, 50)
            indices = torch.from_numpy(sample.relations)
            relation[indices[:, 0], indices[:, 1], indices[:, 2] - 1] = 1
            target['rel'] = relation
        image, target = transform(image, target)
        if self.task == 'detection':
            target['labels'] -= 1  # same ordering as native VGDetection
        else:
            count = len(target['labels'])
            assert target['rel'].shape == (count, count, 50)
            padded = torch.zeros(self.queries, self.queries, 50)
            padded[:count, :count] = target['rel']
            target['rel'] = padded
        return image, target

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_runtime'] = None
        return state
