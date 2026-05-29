#!/usr/bin/env python3
"""Subprocess gauntlet: core imports must exit 0 with clean stderr."""
from __future__ import annotations

import subprocess
import sys

CHILD = r"""
import importlib.metadata as md
import json

import bytewax.dataflow as bytewax_dataflow
from gliner import GLiNERConfig
from river import linear_model
from treelite import model_builder as treelite_builder

gliner_config = GLiNERConfig(model_name="distilbert-base-uncased")
river_model = linear_model.LogisticRegression()
bytewax_flow = bytewax_dataflow.Dataflow("lucidota-core-import-gauntlet")
treelite_meta = treelite_builder.Metadata(
    num_feature=3,
    task_type="kRegressor",
    average_tree_output=False,
    num_target=1,
    num_class=[1],
    leaf_vector_shape=(1, 1),
)
treelite_ann = treelite_builder.TreeAnnotation(num_tree=1, target_id=[0], class_id=[0])
treelite_post = treelite_builder.PostProcessorFunc(name="identity")
treelite_model_builder = treelite_builder.ModelBuilder(
    threshold_type="float32",
    leaf_output_type="float32",
    metadata=treelite_meta,
    tree_annotation=treelite_ann,
    postprocessor=treelite_post,
    base_scores=[0.0],
)
print(json.dumps({
    "ok": True,
    "schema": "lucidota.core_import_gauntlet.v1",
    "imports": {
        "gliner": {"version": md.version("gliner"), "class": type(gliner_config).__name__},
        "river": {"version": md.version("river"), "class": type(river_model).__name__},
        "bytewax": {"version": md.version("bytewax"), "class": type(bytewax_flow).__name__},
        "treelite": {"version": md.version("treelite"), "class": type(treelite_model_builder).__name__},
    },
}, sort_keys=True))
"""


def main() -> int:
    proc = subprocess.run([sys.executable, "-c", CHILD], text=True, capture_output=True, check=False)
    if proc.stderr:
        sys.stderr.write(proc.stderr)
        return 1
    sys.stdout.write(proc.stdout)
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
