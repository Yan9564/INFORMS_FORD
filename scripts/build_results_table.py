#!/usr/bin/env python
from __future__ import annotations
import argparse
from benchmark.reporting.build_tables import build_tables
p=argparse.ArgumentParser(); p.add_argument('--results-dir', default='results/raw'); p.add_argument('--output-dir', default='results/aggregated')
a=p.parse_args(); print(build_tables(a.results_dir, a.output_dir))
