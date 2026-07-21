# data.py

import os

import h5py
import numpy as np
import torch

from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm


class SegmentationDataset(Dataset):
    """Dataset suitable for segmentation tasks, backed by per-class HDF5 files.

        Each class contributes one HDF5 file (written by preprocess.process_file) holding
        two chunked, resizable datasets, 'hits' and 'truth', chunked at one sample per
        chunk. Because HDF5 chunks are read (and decompressed) independently, a single
        __getitem__ call only touches the one chunk it needs - unlike the previous
        .npz-shard format, there's no need to decompress a whole shard up front, and
        therefore no LRU shard cache or shard-grouped sampler required to make random
        access affordable.

        h5py.File handles are NOT opened in __init__: they aren't safe to share across a
        fork, so with num_workers>0 each DataLoader worker process must open its own
        independent handle. Handles are instead opened lazily on first access and cached
        per-process in self._files. __getstate__ strips any open handles before the
        Dataset is copied into a worker (relevant on the 'spawn' start method, e.g.
        Windows/macOS, where the Dataset is pickled rather than fork-copied).

        By default tensors are built on CPU, meant to be moved to GPU in the training loop
        with `.to(device, non_blocking=True)` (works together with num_workers>0 +
        pin_memory for overlapped I/O). If `device` is set to a CUDA device here instead,
        samples are placed on the GPU inside __getitem__ directly. CUDA tensors cannot
        cross process boundaries, so this only works with num_workers=0 (single-process,
        synchronous loading) - SegmentationBunch enforces this automatically when device
        is set.
    """

    def __init__(self, h5_paths, sample_index, transform=False, device=None):
        """Constructor.

            Args:
                h5_paths: Array of HDF5 file paths, one per class, indexed by file_id.
                sample_index: Array of shape (n_samples, 2) with (file_id, local_idx)
                    pairs, one row per dataset sample.
                transform: Whether or not to apply random flip/transpose augmentation
                    (default: False).
                device: If given, samples are created directly on this device (e.g.
                    torch.device('cuda:0')) instead of staying on CPU. Requires
                    num_workers=0 on the DataLoader (default: None -> CPU tensors).
        """
        self.h5_paths = h5_paths
        self.sample_index = sample_index
        self.transform = transform
        # Always resolve to an explicit torch.device, never a bare None - passing
        # device=None into tensor constructors silently falls back to whatever
        # torch.set_default_device() was last set to elsewhere in the notebook, rather
        # than reliably meaning "CPU".
        self.device = device if device is not None else torch.device('cpu')
        self._files = {}  # populated lazily per-worker process: file_id -> h5py.File

    def __len__(self):
        """Retrieve the number of samples in the dataset.

            Returns:
                The number of samples in the dataset
        """
        return len(self.sample_index)

    def _get_file(self, file_id):
        """Lazily open (and cache, per worker process) the HDF5 file for file_id.

            Args:
                file_id: Index into self.h5_paths

            Returns:
                An open h5py.File for that path
        """
        hf = self._files.get(file_id)
        if hf is None:
            hf = h5py.File(self.h5_paths[file_id], 'r')
            self._files[file_id] = hf
        return hf

    def __getitem__(self, idx):
        """Retrieve a sample from the dataset.

            Args:
                idx: The index of the sample to be retrieved

            Returns:
                A (image, mask) tuple of tensors, on self.device (always CPU unless a
                device was explicitly given to the constructor)
        """
        file_id, local_idx = self.sample_index[idx]
        hf = self._get_file(int(file_id))
        # Each of these reads exactly one HDF5 chunk (one event) off disk/decompresses it -
        # not the surrounding shard/file.
        image = hf['hits'][int(local_idx)]
        mask = hf['truth'][int(local_idx)]

        image = torch.as_tensor(np.expand_dims(image, axis=0), device=self.device, dtype=torch.float)
        mask = torch.as_tensor(mask, device=self.device, dtype=torch.long)

        if self.transform:
            should_hflip = torch.rand(1).item() > 0.5
            should_vflip = torch.rand(1).item() > 0.5
            should_transpose = torch.rand(1).item() > 0.5
            if should_hflip:
                image = torch.flip(image, dims=[-1])
                mask = torch.flip(mask, dims=[-1])
            if should_vflip:
                image = torch.flip(image, dims=[-2])
                mask = torch.flip(mask, dims=[-2])
            if should_transpose:
                image = image.transpose(-2, -1)
                mask = mask.transpose(-2, -1)

        return (image, mask)

    def __getstate__(self):
        """Strip open h5py.File handles before this Dataset is copied into a DataLoader
            worker process. File handles aren't picklable (and aren't safe to share across
            a fork even when pickling is skipped), so each worker must lazily reopen its
            own handles via _get_file on first access instead.

            Returns:
                A dict suitable for pickling/copying, with self._files reset to empty
        """
        state = self.__dict__.copy()
        state['_files'] = {}
        return state


def build_sample_index(h5_paths, counts):
    """Build a (file_id, local_idx) index covering every sample across a set of HDF5 files.

        Args:
            h5_paths: List/array of HDF5 file paths, in file_id order.
            counts: Array of per-file sample counts (h5_paths[i] has counts[i] samples).

        Returns:
            A numpy array of shape (sum(counts), 2) with (file_id, local_idx) pairs.
    """
    total = int(np.sum(counts))
    sample_index = np.empty((total, 2), dtype=np.int64)
    pos = 0
    for file_id, n in enumerate(counts):
        n = int(n)
        sample_index[pos:pos + n, 0] = file_id
        sample_index[pos:pos + n, 1] = np.arange(n)
        pos += n
    return sample_index


def _h5_length(path):
    """Get a HDF5 file's sample count from its dataset shape.

        Args:
            path: Path to a data.h5 file written by preprocess.process_file

        Returns:
            The number of samples ('hits' dataset length) in the file

        Note:
            This just reads a dataset's .shape attribute - no decompression needed, unlike
            the old .npz format where getting a shard's length required decompressing the
            entire array. Cheap enough to call once per file at construction time.
    """
    with h5py.File(path, 'r') as hf:
        return hf['hits'].shape[0]


class SegmentationBunch():
    """Associates batches of training and validation datasets suitable for segmentation
        tasks, reading from the per-class HDF5 files written by preprocess.process_file
        (root_dir/<class_dir>/data.h5 - one file per class, not many shard files).
    """

    def __init__(self, root_dir, balance_map, batch_size, valid_pct=0.1, test_pct=0.0,
                 transform=False, shuffle_training=True, num_workers=8, pin_memory=True, device=None):
        """Constructor.

            Args:
                root_dir: The top-level directory containing per-class subdirectories
                balance_map: Dict mapping the different classes (NuMI/numu, NuMI/nue, etc...)
                    to their directory (relative to root_dir, containing a data.h5 written
                    by preprocess.process_file) and target fraction of the total, e.g.:
                        {
                            'class_1': {'dir': 'NuMI/numu', 'fraction': 0.25},
                            'class_2': {'dir': 'BNB/numu', 'fraction': 0.5},
                            'class_3': {'dir': 'BNB/nue', 'fraction': 0.25}
                        }
                    Fractions must sum to 1.0. Sizes are normalized to the bottleneck class.
                batch_size: The batch size
                valid_pct: The fraction of samples to be used for validation (default: 0.1)
                test_pct: The fraction of samples reserved for testing, currently unused for
                    splitting but kept to preserve the (valid_pct + test_pct) < 1 sanity check
                    (default: 0.0)
                transform: Whether or not to apply augmentation to the training set (default: False)
                num_workers: DataLoader worker processes for parallel I/O (default: 8).
                    Ignored (forced to 0) if device is set - see device below. Each worker
                    opens its own HDF5 file handles lazily (see SegmentationDataset), which
                    is cheap compared to decompressing a whole .npz shard, so parallel
                    workers are the normal, expected configuration here.
                pin_memory: Whether to use pinned host memory so H2D copies can be async
                    (default: True). Ignored (forced to False) if device is set, since pinned
                    memory only matters for CPU tensors being copied to GPU.
                device: If given (e.g. torch.device('cuda:0')), samples are created directly
                    on this device inside the Dataset instead of staying on CPU. CUDA tensors
                    cannot cross process boundaries, so this forces num_workers=0 and
                    pin_memory=False - loading becomes single-process and synchronous, and
                    each __getitem__ call blocks the GPU until the sample is ready. Default:
                    None -> CPU tensors, parallel loading via num_workers.
        """
        assert (valid_pct + test_pct) < 1.
        total_fraction = sum(cfg['fraction'] for cfg in balance_map.values())
        assert np.isclose(total_fraction, 1.0), "Fractions in balance_map must sum to 1.0"

        if device is not None:
            if num_workers > 0:
                print(f"device={device} was set: forcing num_workers=0, shuffle_training=False and pin_memory=False "
                      f"(was {num_workers}) since CUDA tensors can't cross process boundaries")
            num_workers = 0
            shuffle_training = False
            pin_memory = False

        class_names = list(balance_map.keys())
        h5_paths = np.array([
            os.path.join(root_dir, balance_map[c]['dir'], 'data.h5') for c in class_names
        ])
        
        # Cheap: just reads each file's dataset shape, no decompression.
        counts = np.array([_h5_length(p) for p in tqdm(h5_paths, desc='Reading class sizes')])
        per_class_count = dict(zip(class_names, counts))

        # Find the total dataset capacity dictated by the bottleneck class
        max_total = min(per_class_count[c] / balance_map[c]['fraction'] for c in balance_map)

        train_rows, valid_rows = [], []

        for file_id, class_name in enumerate(tqdm(class_names, desc='Sampling classes')):
            config = balance_map[class_name]
            n_to_sample = int(max_total * config['fraction'])
            n_total = int(counts[file_id])

            chosen_local = np.random.permutation(n_total)[:n_to_sample]
            chosen = np.empty((n_to_sample, 2), dtype=np.int64)
            chosen[:, 0] = file_id
            chosen[:, 1] = chosen_local

            n_valid = int(len(chosen) * valid_pct)
            perm = np.random.permutation(len(chosen))
            valid_rows.append(chosen[perm[:n_valid]])
            train_rows.append(chosen[perm[n_valid:]])

        train_index = np.concatenate(train_rows)
        valid_index = np.concatenate(valid_rows)
        # No manual shuffle needed here (and no shard-grouped sampler, unlike the old
        
        # .npz-shard version): DataLoader(shuffle=True) below reshuffles every epoch, and
        # because each HDF5 read only costs one chunk regardless of access order, plain
        # random per-sample order no longer thrashes a cache the way it did with shards.

        train_ds = SegmentationDataset(h5_paths, train_index, transform=transform, device=device)
        valid_ds = SegmentationDataset(h5_paths, valid_index, transform=False, device=device)

        self.train_dl = DataLoader(
            train_ds, batch_size=batch_size, shuffle=shuffle_training, drop_last=True,
            num_workers=num_workers, pin_memory=pin_memory,
            persistent_workers=(num_workers > 0), prefetch_factor=4 if num_workers > 0 else None
        )
        self.valid_dl = DataLoader(
            valid_ds, batch_size=batch_size, shuffle=False, drop_last=True,
            num_workers=num_workers, pin_memory=pin_memory,
            persistent_workers=(num_workers > 0), prefetch_factor=4 if num_workers > 0 else None
        )

    def count_classes(self, num_classes, device=None, read_batch_size=20000):
        """Count the number of instances of each class in the training set.
 
            Args:
                num_classes: The number of classes in the training set
                device: Device to accumulate the running count on (default: None -> CPU).
                    Batches arrive on CPU from the DataLoader; pass a CUDA device here only
                    if you want the accumulation itself done on GPU.
                read_batch_size: Max samples to fancy-index out of a file in one read
                    (default: 2000). A single file can contribute hundreds of thousands of
                    samples to the training split - fancy-indexing them all in one h5py call
                    would materialize that entire subset in memory at once (and upcasting to
                    torch.long before bincount made that an extra 8x on top, since truth is
                    stored as uint8). Reading in bounded batches instead keeps peak memory to
                    roughly read_batch_size * image_height * image_width regardless of how
                    large any one file's split is.
 
            Returns:
                A numpy array of the number of instances of each class
        """
        resolved_device = device if device is not None else torch.device('cpu')
        ds = self.train_dl.dataset
 
        # Group this Dataset's training sample indices by which HDF5 file they live in, so
        # each file is opened exactly once here.
        file_to_local = {}
        for file_id, local_idx in ds.sample_index:
            file_to_local.setdefault(int(file_id), []).append(int(local_idx))
 
        count = torch.zeros(num_classes, dtype=torch.long, device=resolved_device)
        for file_id, local_indices in tqdm(file_to_local.items(), desc='Counting classes'):
            # h5py requires fancy-index lists to be strictly increasing.
            local_indices = sorted(local_indices)
            with h5py.File(ds.h5_paths[file_id], 'r') as hf:
                truth_ds = hf['truth']
                for start in tqdm(range(0, len(local_indices), read_batch_size), desc=f'Counting for batch'):
                    batch_idx = local_indices[start:start + read_batch_size]
                    truth = torch.from_numpy(truth_ds[batch_idx])  # stays uint8 here
                    truth = truth.to(resolved_device, non_blocking=True)
                    # .long() only on this bounded batch, not the whole file's selection
                    count += torch.bincount(truth.flatten().long(), minlength=num_classes)
        return count.cpu().numpy()