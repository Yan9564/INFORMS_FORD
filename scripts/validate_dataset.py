#!/usr/bin/env python
from __future__ import annotations
import argparse, yaml
from benchmark.datasets.ford import FordDatasetConfig, FordDatasetLoader
p=argparse.ArgumentParser(); p.add_argument('--config', required=True); p.add_argument('--accepted-dir'); p.add_argument('--rejected-dir')
a=p.parse_args(); cfg=yaml.safe_load(open(a.config))['dataset'];
if a.accepted_dir: cfg['accepted_dir']=a.accepted_dir
if a.rejected_dir: cfg['rejected_dir']=a.rejected_dir
loader=FordDatasetLoader(FordDatasetConfig.from_dict(cfg)); b=loader.load()
print('accepted files:', [p.name for _,p in loader.discover_files('accept')]); print('rejected files:', [p.name for _,p in loader.discover_files('reject')])
print('features:', b.feature_names); print('shapes:', b.X_train.shape, b.X_validation.shape, b.X_test.shape)
print('validation labels:', dict(zip(*__import__('numpy').unique(b.y_validation, return_counts=True)))); print('test labels:', dict(zip(*__import__('numpy').unique(b.y_test, return_counts=True))))
print('warnings:', b.warnings)
