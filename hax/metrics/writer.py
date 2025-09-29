#!/usr/bin/env python


import os
import numpy as np
from datetime import datetime

from torch.utils.tensorboard import SummaryWriter

import jax
from jax import device_get
from jax.numpy import ndarray as JaxArray


class JaxSummaryWriter(SummaryWriter):
    def __init__(self, log_dir=None, **kwargs):
        super(JaxSummaryWriter, self).__init__(log_dir=os.path.join(log_dir, datetime.now().strftime("%Y%m%d-%H%M%S")), **kwargs)

    def _to_numpy(self, x):
        # If it’s a JAX array, pull it to host and convert
        if isinstance(x, JaxArray):
            return np.asarray(device_get(x))
        return x

    def _convert_tree(self, tree):
        # Recursively map over lists/tuples/dicts
        return jax.tree_util.tree_map(self._to_numpy, tree)

    def __getattribute__(self, name):
        # 1) Always let internal/private names through unwrapped:
        if name.startswith("_"):
            return object.__getattribute__(self, name)

        # 2) If the base class doesn’t define this attr, pass it straight through:
        if not hasattr(SummaryWriter, name):
            return object.__getattribute__(self, name)

        # 3) Grab the real method from *self* (so bound correctly):
        target = object.__getattribute__(self, name)

        # 4) If it isn’t callable, return it as-is:
        if not callable(target):
            return target

        # 5) Otherwise return our wrapper:
        def wrapped(*args, **kwargs):
            # cheap check: do we actually need to convert?
            leaves = jax.tree_util.tree_leaves((args, kwargs))
            if not any(isinstance(x, JaxArray) for x in leaves):
                return target(*args, **kwargs)

            # only pay conversion cost when we see a JAX array
            args_np = self._convert_tree(args)
            kwargs_np = self._convert_tree(kwargs)
            return target(*args_np, **kwargs_np)

        return wrapped




def main():
    import argparse
    import time
    from tensorboard import program
    from hax.utils import bcolors

    parser = argparse.ArgumentParser(
        description="Launch TensorBoard for a given JAX SummaryWriter log directory"
    )
    parser.add_argument(
        "--logdir", type=str, required=True,
        help="Path to the TensorBoard log directory"
    )
    args = parser.parse_args()

    # Launch TensorBoard programmatically
    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', args.logdir])
    url = tb.launch()
    print(f"{bcolors.WARNING}TensorBoard is running at {url}")
    print(f"Press {bcolors.UNDERLINE}{bcolors.BOLD}Ctrl+C{bcolors.ENDC} {bcolors.WARNING}to stop.{bcolors.ENDC}")

    # Block until Ctrl-C
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nReceived Ctrl+C—shutting down.")

if __name__ == "__main__":
    main()
