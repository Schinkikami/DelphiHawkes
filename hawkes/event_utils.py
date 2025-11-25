#%%
from tensordict import TensorDict
from torch import Tensor
import torch


def inverse_softplus(x:Tensor):
    # Computes the inverse of the softplus function,
    # using the numerically stable log(expm1(x)) implementation
    # (sadly torch does not provide logexpm1)
    return torch.log(torch.expm1(x))

#%%

class MVEventData(TensorDict):
    def __init__(self, time_points:Tensor, event_types:Tensor, sort:bool=False):

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


        super().__init__({
            'time_points': time_points,
            'event_types': event_types
        }, batch_size=len(time_points))



    @property
    def time_points(self):
        return self['time_points']

    @property
    def event_types(self):
        return self['event_types']

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
        return (f"MVEventData(len={length}, [{self.time_points.dtype}, {self.event_types.dtype}])\n"
                f"  time_points: {tps}\n"
                f"  event_types: {ets}")

