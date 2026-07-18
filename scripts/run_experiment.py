#!/usr/bin/env python
from __future__ import annotations
import argparse
from benchmark.runners.experiment import run_experiment
p=argparse.ArgumentParser(); p.add_argument('--config', required=True); p.add_argument('--accepted-dir'); p.add_argument('--rejected-dir'); p.add_argument('--output-dir', default='results/raw')
a=p.parse_args(); r=run_experiment(a.config, a.output_dir, {'accepted_dir': a.accepted_dir, 'rejected_dir': a.rejected_dir}); print(r['run_id'])
