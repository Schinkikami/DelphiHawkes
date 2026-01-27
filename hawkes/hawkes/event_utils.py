from tensordict import TensorDict
from torch import Tensor
import torch

import numpy as np
from typing import Optional


class MVEventData(TensorDict):
    def __init__(self, time_points: Tensor, event_types: Tensor, sort: bool = False):
        assert time_points.dim() == 1
        assert event_types.dim() == 1
        assert time_points.shape == event_types.shape

        # Assert time_points have float type
        assert time_points.dtype.is_floating_point
        # Assert event_types have integer type
        assert not event_types.dtype.is_floating_point

        if sort:
            sorted_indices = torch.argsort(time_points)
            time_points = time_points[sorted_indices]
            event_types = event_types[sorted_indices]

        super().__init__({"time_points": time_points, "event_types": event_types}, batch_size=len(time_points))

    @property
    def time_points(self):
        return self["time_points"]

    @property
    def event_types(self):
        return self["event_types"]

    def __getitem__(self, idx):
        # If idx is an int/scalar, yield tuple of Python scalars
        if isinstance(idx, int):
            return (self["time_points"][idx], self["event_types"][idx])
        # Otherwise, behave like slicing (keep as MVEventData)
        else:
            return super().__getitem__(idx)

    def __iter__(self):
        # Iteration yields a tuple for each event
        for i in range(len(self)):
            yield self.time_points[i], self.event_types[i]

    def __repr__(self):
        def abbrev(tensor, max_len=8, float_fmt="{:.2f}"):
            l = tensor.tolist()
            if tensor.dtype.is_floating_point:
                l = [float_fmt.format(v) for v in l]
            else:
                l = [str(v) for v in l]
            if len(l) > max_len:
                return "[" + ", ".join(l[:4]) + ", ..., " + ", ".join(l[-3:]) + "]"
            return "[" + ", ".join(l) + "]"

        if self.time_points.dim() == 1:
            length = self.shape[0]
        else:
            length = 0
        tps = abbrev(self.time_points)
        ets = abbrev(self.event_types)
        return (
            f"MVEventData(len={length}, [{self.time_points.dtype}, {self.event_types.dtype}])\n"
            f"  time_points: {tps}\n"
            f"  event_types: {ets}"
        )


# %%
class BatchedMVEventData(TensorDict):
    MAX_TIME_OFFSET = 1.0

    def __init__(
        self,
        time_points: Optional[list[Tensor]] = None,
        event_types: Optional[list[Tensor]] = None,
        mv_events: Optional[list[MVEventData]] = None,
        sort: bool = False,
    ):
        assert (time_points is not None and event_types is not None) or (mv_events is not None)

        if mv_events is not None:
            assert isinstance(mv_events, list)
            sort = False
            time_points = [ts.time_points for ts in mv_events]
            event_types = [ts.event_types for ts in mv_events]
            batch_size = len(mv_events)

        else:
            assert isinstance(time_points, list)
            assert isinstance(event_types, list)
            assert len(time_points) == len(event_types)

            batch_size = len(time_points)

            for t, e in zip(time_points, event_types):
                assert t.dim() == 1
                assert e.dim() == 1
                assert t.shape == e.shape

            # Assert time_points have float type and event_types have integer type (if not empty)
            if batch_size > 0:
                assert time_points[0].dtype.is_floating_point
                assert not event_types[0].dtype.is_floating_point

        if sort:
            for i in range(len(time_points)):
                sorted_indices = torch.argsort(time_points[i])
                time_points[i] = time_points[i][sorted_indices]
                event_types[i] = event_types[i][sorted_indices]

        lengths = [len(b) for b in time_points]
        length = max(lengths) if lengths else 0

        # Determine device and dtypes
        if batch_size > 0:
            device = time_points[0].device
            dtype_time = time_points[0].dtype
            dtype_event = event_types[0].dtype
        else:
            # For empty batch, use default dtypes and CPU device
            device = torch.device("cpu")
            dtype_time = torch.float32
            dtype_event = torch.long

        # Handle completely empty batch
        if batch_size == 0:
            # Create empty tensors with determined dtypes and device
            padded_time_points = torch.empty((0, 0), dtype=dtype_time, device=device)
            padded_event_types = torch.empty((0, 0), dtype=dtype_event, device=device)
        else:
            # Calculate max_time, handling empty sequences
            # Find the maximum time across all non-empty sequences to use as padding value
            non_empty_last_times = [tp[-1] for tp in time_points if len(tp) > 0]

            if non_empty_last_times:
                max_time = torch.tensor(non_empty_last_times).max() + self.MAX_TIME_OFFSET
            else:
                # All sequences are empty - use just the offset
                max_time = torch.tensor(0, dtype=dtype_time, device=device)

            # Initialize padded tensors with padding values
            # max_time is used for padding time_points (so empty sequences get this value)
            # -1 is used for padding event_types
            padded_time_points = torch.full((batch_size, length), max_time, dtype=dtype_time, device=device)
            padded_event_types = torch.full((batch_size, length), -1, dtype=dtype_event, device=device)

            for i, (t, e) in enumerate(zip(time_points, event_types)):
                n = len(t)
                padded_time_points[i, :n] = t
                padded_event_types[i, :n] = e

        self.seq_lengths = lengths

        super().__init__(
            {"time_points": padded_time_points, "event_types": padded_event_types}, batch_size=(batch_size, length)
        )

    @property
    def time_points(self):
        return self["time_points"]

    @property
    def event_types(self):
        return self["event_types"]

    @property
    def max_time(self):
        """Returns the time of the last event per sequence.

        For empty sequences (length 0), returns 0.0.
        For empty batch (0 sequences), returns empty tensor of shape (0,).
        """
        if len(self.seq_lengths) == 0:
            return torch.empty(0, dtype=self.time_points.dtype, device=self.time_points.device)

        seq_lengths_tensor = torch.tensor(self.seq_lengths, device=self.time_points.device)

        # If all sequences are empty, time_points has shape (B, 0) and we can't index it
        # Just return zeros for all sequences
        if self.time_points.shape[1] == 0:
            return torch.zeros(len(self.seq_lengths), dtype=self.time_points.dtype, device=self.time_points.device)

        batch_indices = torch.arange(len(self.seq_lengths), device=self.time_points.device)

        # For sequences with length > 0, get the last time_point; for length 0, index 0 (will be overwritten)
        last_indices = torch.where(seq_lengths_tensor > 0, seq_lengths_tensor - 1, 0)
        max_times = self.time_points[batch_indices, last_indices]

        # For empty sequences, set max_time to 0
        empty_mask = seq_lengths_tensor == 0
        max_times = torch.where(empty_mask, torch.zeros_like(max_times), max_times)

        return max_times

    def to(self, device):
        """Override to() to preserve seq_lengths attribute when moving to device."""
        result = super().to(device)
        result.seq_lengths = self.seq_lengths
        return result

    def cuda(self, device=None):
        """Override cuda() to preserve seq_lengths attribute."""
        result = super().cuda(device)
        result.seq_lengths = self.seq_lengths
        return result

    def cpu(self):
        """Override cpu() to preserve seq_lengths attribute."""
        result = super().cpu()
        result.seq_lengths = self.seq_lengths
        return result

    def get_unpadded_sequences(self) -> list[MVEventData]:
        """
        Returns a list of unpadded MVEventData sequences.

        Returns:
            List of MVEventData objects containing only the actual events (no padding).
        """
        unpadded = []
        for i in range(len(self.seq_lengths)):
            seq_len = self.seq_lengths[i]
            unpadded.append(
                MVEventData(
                    time_points=self.time_points[i, :seq_len].clone(), event_types=self.event_types[i, :seq_len].clone()
                )
            )
        return unpadded

    def __iter__(self):
        # Iteration yields a batch entry for each event
        for i in range(len(self)):
            yield MVEventData(self.time_points[i, :], self.event_types[i, :])


def generate_data(D, max_len, num_seq, max_T):
    sequences = []

    for _ in range(num_seq):
        # Uniformly draw the lenght of the sequence.
        l = np.random.randint(0, max_len)
        T = np.random.rand() * max_T

        time_points = np.random.rand(l) * T
        time_points = np.sort(time_points)
        T += 1

        event_types = np.random.randint(0, D, l)

        sequences.append((torch.tensor(time_points), torch.tensor(event_types, dtype=torch.long)))

    return sequences


def prepare_data(sequences, batch_size):
    mv_events = [MVEventData(t, e) for t, e in sequences]

    batched_mv_events = []
    for start in range(0, len(sequences), batch_size):
        end = min(start + batch_size, len(sequences))
        subset = sequences[start:end]
        ts = [s[0] for s in subset]
        es = [s[1] for s in subset]
        batch = BatchedMVEventData(ts, es)
        batched_mv_events.append(batch)

    return mv_events, batched_mv_events
