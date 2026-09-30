"""Compact first-turn InstructS2S view: real question audio and full answer text."""

import json
from pathlib import Path

import numpy as np
import soundfile as sf
from torch.utils.data import Dataset

from .streaming import pad_audio_to_chunk_boundary
from .turn_packed import PackedConversation


DATASET_VIEW = 'InstructS2SFirstTurn'
SOURCE_ID = 'yuekai/InstructS2S-200K'
SOURCE_REVISION = '2627cebf9734d959c64dcb86ca1a13e969e5a284'


class InstructS2SFirstTurnDataset(Dataset):
    """Use pre-resampled FLACs without duplicating their bytes in Arrow caches."""

    def __init__(self, root: str | Path, *, split: str):
        self.root = Path(root)
        if split not in ('train', 'dev'):
            raise ValueError('InstructS2S split must be train or dev.')
        self.records = [json.loads(line) for line in (self.root / f'{split}.jsonl').read_text().splitlines()]
        self.conversation_ids = [row['id'] for row in self.records]
        self.frame_lengths = [row['frames'] for row in self.records]
        if len(set(self.conversation_ids)) != len(self.records):
            raise ValueError('Duplicate InstructS2S conversation IDs.')

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        row = self.records[index]
        waveform, rate = sf.read(self.root / row['audio'], dtype='float32')
        if rate != 16000 or waveform.ndim != 1 or len(waveform) != row['user_samples']:
            raise ValueError(f"{row['id']}: prepared question audio differs from its manifest.")
        waveform = pad_audio_to_chunk_boundary(np.asarray(waveform, dtype=np.float32))
        return PackedConversation(conversation_id=row['id'], user_waveform=waveform,
            user_valid_samples=row['user_samples'], assistant_samples=row['assistant_samples'],
            assistant_text=row['answer'])
