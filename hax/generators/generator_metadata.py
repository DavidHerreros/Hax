import numpy as np
import random
from jax import numpy as jnp
from jax.tree_util import tree_map
from xmipp_metadata.metadata import XmippMetaData


class MetaDataGenerator:
    def __init__(self, file, mode=None):
        self.md = XmippMetaData(file)
        self.mode = mode

        # Generator mode
        if mode == "tomo" or self.md.isMetaDataLabel("subtomo_labels"):
            unique_labels = np.unique(self.md[:, "subtomo_labels"]).astype(int)
            self.sinusoid_tabled = get_sinusoid_encoding_table(np.amax(unique_labels), 100)
        else:
            self.sinusoid_tabled = np.zeros(len(self.md))

    def __len__(self):
        return len(self.md)

    def __getitem__(self, idx):
        if self.mode == "tomo":
            return self.md.getMetaDataImage(idx)[..., None], self.sinusoid_tabled[idx], idx
        else:
            return self.md.getMetaDataImage(idx)[..., None], idx

    def return_tf_dataset(self, preShuffle=False, shuffle=True, prefetch=5, batch_size=8):
        import tensorflow_datasets as tfds
        import tensorflow as tf
        tf.config.set_visible_devices([], device_type='GPU')
        with tf.device("/CPU:0"):
            file_idx = np.arange(len(self.md))
            if preShuffle:
                np.random.shuffle(file_idx)
            images = self.md.getMetaDataImage(file_idx)[..., None]
            if self.mode == "tomo":
                subtomo_labels = self.sinusoid_table[self.md[file_idx, "subtomo_labels"].astype(int) - 1]
                dataset = tf.data.Dataset.from_tensor_slices(((images, subtomo_labels), (file_idx, file_idx)))
            else:
                dataset = tf.data.Dataset.from_tensor_slices((images, file_idx))
            if shuffle:
                dataset = dataset.shuffle(len(file_idx))
            # dataset = dataset.map(lambda image, label: (self.data_augmentation(image), label))
            return tfds.as_numpy(dataset.batch(batch_size).prefetch(prefetch))

    def return_torch_dataset(self, shuffle=True, batch_size=8):
        from torch.utils.data import default_collate, DataLoader

        def numpy_collate(batch):
            return tree_map(np.asarray, default_collate(batch))

        return DataLoader(self, batch_size=batch_size, shuffle=shuffle, num_workers=0, collate_fn=numpy_collate)

    def return_grain_dataset(self, preShuffle=False, shuffle=True, batch_size=8):
        import grain

        class CustomSource(grain.sources.RandomAccessDataSource):
            def __init__(self, data, labels, subtomo_labels=None):
                self.data = data
                self.labels = labels
                self.subtomo_labels = subtomo_labels

            def __getitem__(self, idx):
                if self.subtomo_labels is None:
                    return self.data[idx], self.labels[idx]
                else:
                    return (self.data[idx], self.subtomo_labels[idx]), self.labels[idx]

            def __len__(self):
                return len(self.data)

        file_idx = np.arange(len(self.md))
        if preShuffle:
            np.random.shuffle(file_idx)
        images = self.md.getMetaDataImage(file_idx)[..., None]
        if self.mode == "tomo":
            subtomo_labels = self.sinusoid_table[self.md[file_idx, "subtomo_labels"].astype(int) - 1]
            dataset = grain.MapDataset.source(CustomSource(images, file_idx, subtomo_labels))
        else:
            dataset = grain.MapDataset.source(CustomSource(images, file_idx))
        if shuffle:
            seed = random.randint(0, 2 ** 32 - 1)
            dataset = dataset.shuffle(seed=seed)
        return dataset.to_iter_dataset().batch(batch_size)

def extract_columns(md, hasCTF=None):
    hasCTF = md.isMetaDataLabel("ctfDefocusU") if hasCTF is None else hasCTF

    columns = {}
    columns["euler_angles"] = jnp.array(md.getMetaDataColumns(["angleRot", "angleTilt", "anglePsi"]).astype(jnp.float32))
    columns["shifts"] = jnp.array(md.getMetaDataColumns(["shiftX", "shiftY"]).astype(jnp.float32))
    if hasCTF:
        columns["ctfDefocusU"] = jnp.array(md.getMetaDataColumns("ctfDefocusU").astype(jnp.float32))
        columns["ctfDefocusV"] = jnp.array(md.getMetaDataColumns("ctfDefocusV").astype(jnp.float32))
        columns["ctfDefocusAngle"] = jnp.array(md.getMetaDataColumns("ctfDefocusAngle").astype(jnp.float32))
        columns["ctfSphericalAberration"] = jnp.array(md.getMetaDataColumns("ctfSphericalAberration").astype(jnp.float32))
        columns["ctfVoltage"] = jnp.array(md.getMetaDataColumns("ctfVoltage").astype(jnp.float32))
    return columns


def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    """
    numpy sinusoid position encoding of Transformer model.
    params:
        n_position(n):number of positions
        d_hid(m): dimension of embedding vector
        padding_idx:set 0 dimension
    return:
        sinusoid_table(n*m):numpy array
    """

    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.

    return sinusoid_table
