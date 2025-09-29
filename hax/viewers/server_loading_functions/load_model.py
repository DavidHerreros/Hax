def prepare_heterogeneity_program(**kwargs):
    from hax.checkpointer import NeuralNetworkCheckpointer

    # Load neural network (note it MUST be saved in pickle mode to make this script general)
    model = NeuralNetworkCheckpointer.load(None, kwargs.pop("pickled_nn"), mode="pickle")

    return model

def decode_state_from_latent(latent, model, path_template):
    from xmipp_metadata.image_handler import ImageHandler
    import numpy as np

    states = np.array(model.decode_states(latent))

    idx = 1
    for state in states:
        ImageHandler().write(state, filename=path_template.format(idx), overwrite=True)
        idx += 1
