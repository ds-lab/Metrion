import os

# Thread pools are sized when polars and numpy are first imported, so the limits must be set
# before either is loaded. This file runs first, in the agent and in every spawned child.
os.environ.setdefault("POLARS_MAX_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
