import numpy as np
from pathlib import Path
import torch
from event_utils import MVEventData


def load_ukb_sequences(
    data_path: str | Path,
    limit_size,
):
    data_path = Path(data_path)
    assert data_path.exists(), "Data file does not exist."
    np_data = np.memmap(str(data_path), mode="r", dtype=np.uint32).reshape(-1, 3)

    batch_ids = np_data[:, 0].astype(int)
    time_points = np_data[:, 1].astype(float)
    event_types = np_data[:, 2].astype(int)

    male_token = 2
    female_token = 3
    death_token = 1269

    # Remap tokens, so that we start at 0. (0 and 1 are reserved for Padding and NoEvent).
    event_types -= 2
    male_token -= 2
    female_token -= 2
    death_token -= 2

    num_events = death_token + 1

    time_points = time_points / 365.0  # Convert to years
    time_points /= 80  # Encode in 80 years to have be better numerical stability.
    # Split data by batch id
    sequences = []

    batch_ids_np = np.asarray(batch_ids)
    change_idcs = np.flatnonzero(batch_ids_np[1:] != batch_ids_np[:-1]) + 1
    change_idcs = np.concatenate([np.array([0]), change_idcs, np.array([len(batch_ids_np)])])

    if limit_size is not None:
        change_idcs = change_idcs[: limit_size + 1]
    for start, stop in zip(change_idcs[:-1], change_idcs[1:]):
        sequence_time = time_points[start:stop]
        sequence_event = event_types[start:stop]

        ev_data = MVEventData(
            time_points=torch.tensor(sequence_time, dtype=torch.float32),
            event_types=torch.tensor(sequence_event, dtype=torch.long),
            sort=True,
        )
        sequences.append(ev_data)

    # Sequences always start with male/female. Filter illegal sequences.
    first_tokens = np.array([ts.event_types[0] for ts in sequences])

    legal_sequence_icds = np.flatnonzero(np.logical_or(first_tokens == male_token, first_tokens == female_token))
    sequences = [sequences[i] for i in legal_sequence_icds]

    # Diseases can't occure twice (first occurence data).
    # Result: We don't need to filter. Data is correct.

    # Remove conditioning covariates (for now only sex) from the sequence beginning.
    sexes = []

    for i, seq in enumerate(sequences):
        sex = seq.event_types[0]

        new_seq = MVEventData(seq.time_points[1:], seq.event_types[1:])

        sexes.append(sex)
        sequences[i] = new_seq

    # Filter out empty sequences.
    sequences = [seq for seq in sequences if len(seq) > 0]

    return sequences, sexes, num_events
