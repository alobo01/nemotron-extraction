"""Load callback definitions from a nemotron-sae checkout without its training imports."""
from __future__ import annotations

import ast
from collections.abc import Iterable
from concurrent.futures import Future, ThreadPoolExecutor
from functools import partial
import json
from pathlib import Path
import time

import torch
from torch import nn
from safetensors.torch import save_file


def sibling_namespace(directory):
    """Execute attributed callback definitions without training/queue imports."""
    scope = dict(torch=torch, nn=nn, partial=partial, Path=Path, json=json, time=time,
                 ThreadPoolExecutor=ThreadPoolExecutor, Future=Future, Iterable=Iterable, save_file=save_file)
    for filename in ('attention.py', 'worker.py'):
        path = Path(directory)/filename
        tree = ast.parse(path.read_text())
        nodes = [ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)]
        nodes += [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) or
                  isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in
                  ('MIXER_KINDS','RESIDUAL_KINDS') for t in n.targets)]
        exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), str(path), 'exec'), scope)
    return scope
