# Copyright (c) OpenMMLab. All rights reserved.
from .ade import ADE20KDataset
from .builder import DATASETS, PIPELINES, build_dataloader, build_dataset
from .chase_db1 import ChaseDB1Dataset
from .cityscapes import CityscapesDataset
from .coco_stuff import COCOStuffDataset
from .custom import CustomDataset
from .dark_zurich import DarkZurichDataset
from .dataset_wrappers import (ConcatDataset, MultiImageMixDataset,
                               RepeatDataset)
from .drive import DRIVEDataset
from .hrf import HRFDataset
from .isaid import iSAIDDataset
from .isprs import ISPRSDataset
from .loveda import LoveDADataset
from .night_driving import NightDrivingDataset
from .pascal_context import PascalContextDataset, PascalContextDataset59
from .potsdam import PotsdamDataset
from .stare import STAREDataset
from .voc import PascalVOCDataset
from .cityscapes_night import Cityscapes_NightDataset
from .cityscapes_rainy import Cityscapes_RainyDataset
from .cityscapes_snowy import Cityscapes_SnowyDataset
from .ade_night import ADE20K_NightDataset
from .cityscapes_foggy import Cityscapes_FoggyDataset
from .bdd100k import BDD100KDataset
from .bdd100k_night import BDD100K_NightDataset
from .bdd100k_snowy import BDD100K_SnowyDataset
from .bdd100k_foggy import BDD100K_FoggyDataset
from .bdd100k_rainy import BDD100K_RainyDataset
from .bdd100k_aug import BDD100K_AugDataset

__all__ = [
    'CustomDataset', 'build_dataloader', 'ConcatDataset', 'RepeatDataset',
    'DATASETS', 'build_dataset', 'PIPELINES', 'CityscapesDataset',
    'PascalVOCDataset', 'ADE20KDataset', 'PascalContextDataset',
    'PascalContextDataset59', 'ChaseDB1Dataset', 'DRIVEDataset', 'HRFDataset',
    'STAREDataset', 'DarkZurichDataset', 'NightDrivingDataset',
    'COCOStuffDataset', 'LoveDADataset', 'MultiImageMixDataset',
    'iSAIDDataset', 'ISPRSDataset', 'PotsdamDataset', 'Cityscapes_NightDataset',
    'Cityscapes_RainyDataset', 'Cityscapes_SnowyDataset', 'ADE20K_NightDataset',
    'Cityscapes_FoggyDataset', 'BDD100KDataset', 'BDD100K_NightDataset',
    'BDD100K_SnowyDataset', 'BDD100K_FoggyDataset', 'BDD100K_RainyDataset',
    'BDD100K_AugDataset'
]
