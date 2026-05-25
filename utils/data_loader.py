import pickle
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchaudio
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, Wav2Vec2FeatureExtractor


warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message="To copy construct from a tensor*",
)


INVALID_MOSEI_FILES = {
    ("3aIQUQgawaI", 12),
    ("94ULum9MYX0", 2),
    ("mRnEJOLkhp8", 24),
    ("aE-X_QdDaqQ", 3),
    ("94ULum9MYX0", 11),
    ("mRnEJOLkhp8", 26),
}


class MSADataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_root,
        mode,
        text_context_length=2,
        audio_context_length=1,
    ):
        self.dataset_root = Path(dataset_root)
        self.mode = mode
        self.text_context_length = text_context_length
        self.audio_context_length = audio_context_length

        label_path = self.dataset_root / "label.csv"
        audio_root = self.dataset_root / "wav"
        vision_path = self.dataset_root / "Processed" / "unaligned_50.pkl"

        if not label_path.exists():
            raise FileNotFoundError(f"Missing label file: {label_path}")
        if not audio_root.exists():
            raise FileNotFoundError(f"Missing audio directory: {audio_root}")
        if not vision_path.exists():
            raise FileNotFoundError(f"Missing video feature file: {vision_path}")

        df = pd.read_csv(label_path)
        if self.dataset_root.name.upper() == "MOSEI":
            for video_id, clip_id in INVALID_MOSEI_FILES:
                df = df[~((df["video_id"] == video_id) & (df["clip_id"] == clip_id))]
        df = df[df["mode"] == mode].sort_values(by=["video_id", "clip_id"]).reset_index()

        self.targets = df["label"].astype(float).tolist()
        self.texts = df["text"].fillna("").astype(str).tolist()
        self.video_ids = df["video_id"].astype(str).tolist()
        self.clip_ids = df["clip_id"].astype(int).tolist()
        self.audio_paths = [
            audio_root / video_id / f"{clip_id}.wav"
            for video_id, clip_id in zip(self.video_ids, self.clip_ids)
        ]

        self.tokenizer = AutoTokenizer.from_pretrained("roberta-large")
        self.feature_extractor = Wav2Vec2FeatureExtractor(
            feature_size=1,
            sampling_rate=16000,
            padding_value=0.0,
            do_normalize=True,
            return_attention_mask=True,
        )

        self.vision_dict, self.vision_shape = self._load_vision_features(vision_path)
        self.vision_batch = torch.stack(
            [
                self._vision_tensor(video_id, clip_id)
                for video_id, clip_id in zip(self.video_ids, self.clip_ids)
            ],
            dim=0,
        )
        self.vision_masks = torch.stack(
            [self._vision_mask(tensor) for tensor in self.vision_batch],
            dim=0,
        )

    def _load_vision_features(self, vision_path):
        with open(vision_path, "rb") as file:
            all_features = pickle.load(file)

        ids = all_features[self.mode]["id"]
        visions = all_features[self.mode]["vision"]
        vision_shape = tuple(np.asarray(visions[0]).shape)
        vision_dict = {sample_id: vision for sample_id, vision in zip(ids, visions)}
        return vision_dict, vision_shape

    def _vision_key(self, video_id, clip_id):
        return f"{video_id}$_${clip_id}"

    def _vision_tensor(self, video_id, clip_id):
        key = self._vision_key(video_id, clip_id)
        feature = self.vision_dict.get(key)
        if feature is None:
            return torch.zeros(self.vision_shape, dtype=torch.float32)
        return torch.as_tensor(feature, dtype=torch.float32)

    def _vision_mask(self, vision_tensor):
        return (~(vision_tensor == 0).all(dim=-1)).to(torch.long)

    def _text_context(self, index):
        context = []
        for offset in range(1, self.text_context_length + 1):
            context_index = index - offset
            if context_index < 0 or self.video_ids[index] != self.video_ids[context_index]:
                break
            context.insert(0, self.texts[context_index])
        return "</s>".join(context)

    def _audio_context(self, index):
        segments = []
        for offset in range(1, self.audio_context_length + 1):
            context_index = index - offset
            if context_index < 0 or self.video_ids[index] != self.video_ids[context_index]:
                break
            sound, _ = torchaudio.load(self.audio_paths[context_index])
            segments.insert(0, torch.mean(sound, dim=0))
        if not segments:
            return torch.tensor([])
        return torch.cat(segments, dim=0)

    def _video_context(self, index):
        context_index = index - 1
        if context_index >= 0 and self.video_ids[index] == self.video_ids[context_index]:
            return self.vision_batch[context_index], self.vision_masks[context_index]
        return (
            torch.zeros(self.vision_shape, dtype=torch.float32),
            torch.zeros(self.vision_shape[0], dtype=torch.long),
        )

    def _tokenize(self, text):
        return self.tokenizer(
            text,
            max_length=96,
            padding="max_length",
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
        )

    def _audio_features(self, waveform):
        features = self.feature_extractor(
            waveform,
            sampling_rate=16000,
            max_length=96000,
            return_attention_mask=True,
            truncation=True,
            padding="max_length",
        )
        audio = torch.as_tensor(np.array(features["input_values"]), dtype=torch.float32).squeeze()
        mask = torch.as_tensor(np.array(features["attention_mask"]), dtype=torch.long).squeeze()
        return audio, mask

    def __getitem__(self, index):
        text = self.texts[index]
        text_context = self._text_context(index)

        tokenized_text = self._tokenize(text)
        tokenized_context = self._tokenize(text_context)

        sound, _ = torchaudio.load(self.audio_paths[index])
        waveform = torch.mean(sound, dim=0)
        audio_features, audio_masks = self._audio_features(waveform)

        audio_context = self._audio_context(index)
        if audio_context.numel() == 0:
            audio_context_features = torch.zeros(96000, dtype=torch.float32)
            audio_context_masks = torch.zeros(96000, dtype=torch.long)
        else:
            audio_context_features, audio_context_masks = self._audio_features(audio_context)

        video_context, video_context_mask = self._video_context(index)

        return {
            "text_tokens": torch.as_tensor(tokenized_text["input_ids"], dtype=torch.long),
            "text_masks": torch.as_tensor(tokenized_text["attention_mask"], dtype=torch.long),
            "text_context_tokens": torch.as_tensor(
                tokenized_context["input_ids"],
                dtype=torch.long,
            ),
            "text_context_masks": torch.as_tensor(
                tokenized_context["attention_mask"],
                dtype=torch.long,
            ),
            "audio_inputs": audio_features,
            "audio_masks": audio_masks,
            "audio_context_inputs": audio_context_features,
            "audio_context_masks": audio_context_masks,
            "video_inputs": self.vision_batch[index],
            "video_masks": self.vision_masks[index],
            "video_context_inputs": video_context,
            "video_context_masks": video_context_mask,
            "targets": torch.tensor(self.targets[index], dtype=torch.float32),
        }

    def __len__(self):
        return len(self.targets)


def _make_dataset(dataset, split, text_context_length, audio_context_length):
    dataset_root = Path("data") / dataset.upper()
    return MSADataset(
        dataset_root,
        split,
        text_context_length=text_context_length,
        audio_context_length=audio_context_length,
    )


def data_loader(
    batch_size,
    dataset,
    text_context_length=2,
    audio_context_length=1,
    num_workers=0,
):
    dataset = dataset.lower()
    if dataset not in {"mosi", "mosei"}:
        raise ValueError("dataset must be either 'mosi' or 'mosei'")

    train_data = _make_dataset(dataset, "train", text_context_length, audio_context_length)
    val_data = _make_dataset(dataset, "valid", text_context_length, audio_context_length)
    test_data = _make_dataset(dataset, "test", text_context_length, audio_context_length)

    generator = torch.Generator()
    generator.manual_seed(random.randint(0, 2**31 - 1))

    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        generator=generator,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return train_loader, test_loader, val_loader
