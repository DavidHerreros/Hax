import os
from etils import epath

from flax import nnx
import orbax.checkpoint as ocp
import cloudpickle

from hax.utils import bcolors


class NeuralNetworkCheckpointer:

    @classmethod
    def save(cls, model, checkpoint_path, mode="orbax"):
        os.makedirs(checkpoint_path, exist_ok=True)

        checkpoint_path = ocp.test_utils.erase_and_create_empty(os.path.abspath(os.path.join(checkpoint_path, 'checkpoints')))

        if mode == "orbax":
            _, weight_state, _  = nnx.split(model, nnx.Param, ...)
            checkpointer = ocp.PyTreeCheckpointer()
            checkpointer.save(checkpoint_path / 'state', weight_state)
        elif mode == "pickle":
            with open(checkpoint_path / "binary", "wb") as binary_file:
                cloudpickle.dump(model, binary_file)
        else:
            raise ValueError(f"{bcolors.FAIL}Saving mode not implemented")

    @classmethod
    def load(cls, blank_model, checkpoint_path, mode="auto"):
        checkpoint_path = epath.Path(os.path.abspath(os.path.join(checkpoint_path, 'checkpoints')))

        # Automatic loading mode
        if mode == "auto":
            folder = os.listdir(checkpoint_path)
            if folder:
                if folder[0] == "state":
                    mode = "orbax"
                elif folder[0] == "binary":
                    mode = "pickle"
                else:
                    mode = "NotFound"

        if mode == "orbax":
            checkpointer = ocp.PyTreeCheckpointer()
            _, weight_state, _  = nnx.split(blank_model, nnx.Param, ...)
            restored_state = checkpointer.restore(checkpoint_path / 'state', item=weight_state)
            nnx.update(blank_model, restored_state)
        elif mode == "pickle":
            with open(checkpoint_path / "binary", "rb") as binary_file:
                blank_model = cloudpickle.load(binary_file)
        else:
            raise ValueError(f"{bcolors.FAIL}Loading mode not implemented")
        blank_model.eval()
        return blank_model
