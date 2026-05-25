import torch
import torch.nn.functional as F
from torch import nn
from transformers import Data2VecAudioModel, RobertaModel

from utils.attention_layers import CrossAttentionBlock, SelfAttentionBlock


class IVIDModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_size = config.hidden_size

        self.text_encoder = RobertaModel.from_pretrained("roberta-large")
        self.audio_encoder = Data2VecAudioModel.from_pretrained(
            "facebook/data2vec-audio-large-960h"
        )
        self.video_encoder = self._video_encoder(config)

        self.iv_text_encoder = RobertaModel.from_pretrained("roberta-large")
        self.iv_audio_encoder = Data2VecAudioModel.from_pretrained(
            "facebook/data2vec-audio-large-960h"
        )
        self.iv_video_encoder = self._video_encoder(config)

        self.text_cls = nn.Embedding(1, hidden_size)
        self.audio_cls = nn.Embedding(1, hidden_size)
        self.video_cls = nn.Embedding(1, hidden_size)
        self.iv_text_cls = nn.Embedding(1, hidden_size)
        self.iv_audio_cls = nn.Embedding(1, hidden_size)
        self.iv_video_cls = nn.Embedding(1, hidden_size)

        self.text_context_layer = SelfAttentionBlock(
            hidden_size, config.num_attention_heads, config.dropout
        )
        self.audio_context_layer = SelfAttentionBlock(
            hidden_size, config.num_attention_heads, config.dropout
        )
        self.video_context_layer = SelfAttentionBlock(
            hidden_size, config.num_attention_heads, config.dropout
        )

        self.audio_guide = CrossAttentionBlock(
            hidden_size, config.num_attention_heads, config.dropout
        )
        self.video_guide = CrossAttentionBlock(
            hidden_size, config.num_attention_heads, config.dropout
        )

        self.iv_text_projector = SelfAttentionBlock(
            hidden_size, config.num_attention_heads, config.dropout
        )
        self.iv_audio_projector = SelfAttentionBlock(
            hidden_size, config.num_attention_heads, config.dropout
        )
        self.iv_video_projector = SelfAttentionBlock(
            hidden_size, config.num_attention_heads, config.dropout
        )

        self.text_pure_head = self._projection_head(config)
        self.text_bias_head = self._projection_head(config)
        self.audio_pure_head = self._projection_head(config)
        self.audio_bias_head = self._projection_head(config)
        self.video_pure_head = self._projection_head(config)
        self.video_bias_head = self._projection_head(config)

        self.rho = nn.Parameter(torch.tensor(0.5))
        self.reconstruction_loss = nn.MSELoss()
        self.temperature = config.temperature

        self.guide_original_weight = 0.01
        self.guide_attention_weight = 0.98
        self.guide_text_weight = 0.01

        self.fusion = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(hidden_size * 6, hidden_size),
            nn.GELU(),
        )
        self.regressor = nn.Linear(hidden_size, 1)

    def _video_encoder(self, config):
        return nn.Sequential(
            nn.Linear(config.video_feature_dim, config.hidden_size),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size, config.hidden_size),
        )

    def _projection_head(self, config):
        return nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
        )

    def freeze_audio_feature_extractors(self):
        for encoder in (self.audio_encoder, self.iv_audio_encoder):
            for parameter in encoder.feature_extractor.parameters():
                parameter.requires_grad = False

    def _prepend_cls(self, inputs, mask, embedding):
        index = torch.zeros(1, dtype=torch.long, device=inputs.device)
        cls = embedding(index).expand(inputs.size(0), 1, inputs.size(-1))
        cls_mask = torch.ones(inputs.size(0), 1, dtype=mask.dtype, device=mask.device)
        return torch.cat((cls, inputs), dim=1), torch.cat((cls_mask, mask), dim=1)

    def _audio_mask_from_attentions(self, audio_output, fallback_length):
        batch_size = audio_output.last_hidden_state.size(0)
        mask = torch.zeros(
            batch_size,
            fallback_length,
            dtype=torch.long,
            device=audio_output.last_hidden_state.device,
        )

        attentions = audio_output.attentions or []
        for batch_id in range(batch_size):
            valid_length = fallback_length
            for layer_attention in attentions:
                try:
                    valid_length = int((layer_attention[batch_id, 0, 0] != 0).sum().item())
                    break
                except (IndexError, RuntimeError):
                    continue
            valid_length = max(1, min(valid_length, fallback_length))
            mask[batch_id, :valid_length] = 1
        return mask

    def _mean_pool_audio(self, hidden_states, mask):
        mask = mask.to(hidden_states.dtype).unsqueeze(-1)
        lengths = mask.sum(dim=1).clamp(min=1.0)
        return (hidden_states * mask).sum(dim=1) / lengths

    def _encode_audio(self, audio_inputs, audio_mask):
        output = self.audio_encoder(
            audio_inputs,
            attention_mask=audio_mask,
            output_attentions=True,
        )
        sequence = output.last_hidden_state
        sequence_mask = self._audio_mask_from_attentions(output, sequence.size(1))
        pooled = self._mean_pool_audio(sequence, sequence_mask)
        return sequence, sequence_mask, pooled

    def _encode_iv_audio(self, audio_inputs, audio_mask):
        output = self.iv_audio_encoder(
            audio_inputs,
            attention_mask=audio_mask,
            output_attentions=True,
        )
        sequence = output.last_hidden_state
        sequence_mask = self._audio_mask_from_attentions(output, sequence.size(1))
        return sequence, sequence_mask

    def _encode_video(self, video_inputs, video_mask, encoder):
        sequence = encoder(video_inputs)
        mask = video_mask.to(dtype=torch.long, device=sequence.device)
        empty_rows = mask.sum(dim=1) == 0
        if empty_rows.any():
            mask = mask.clone()
            mask[empty_rows, 0] = 1
        return sequence, mask

    def _contrastive_alignment(self, iv_features, pure_features):
        iv_features = F.normalize(iv_features, dim=1)
        pure_features = F.normalize(pure_features, dim=1)
        logits = torch.matmul(iv_features, pure_features.T) / self.temperature
        labels = torch.arange(logits.size(0), device=logits.device)
        return F.cross_entropy(logits, labels)

    def _disentanglement_loss(self, pure, bias, iv_anchor, guided):
        alignment = self._contrastive_alignment(iv_anchor, pure)
        orthogonality = F.cosine_similarity(bias, iv_anchor, dim=1).pow(2).mean()
        completeness = self.reconstruction_loss(pure + bias, guided)
        return alignment, orthogonality, completeness

    def forward(
        self,
        text_inputs,
        text_mask,
        text_context_inputs,
        text_context_mask,
        audio_inputs,
        audio_mask,
        audio_context_inputs,
        audio_context_mask,
        video_inputs,
        video_mask,
        video_context_inputs,
        video_context_mask,
    ):
        text_output = self.text_encoder(text_inputs, text_mask, return_dict=True)
        text_sequence = text_output.last_hidden_state
        text_sequence, text_sequence_mask = self._prepend_cls(
            text_sequence,
            text_mask,
            self.text_cls,
        )

        text_context_output = self.text_encoder(
            text_context_inputs,
            text_context_mask,
            return_dict=True,
        )
        text_context_sequence, text_context_mask = self._prepend_cls(
            text_context_output.last_hidden_state,
            text_context_mask,
            self.text_cls,
        )
        text_context = self.text_context_layer(text_context_sequence, text_context_mask)

        audio_sequence, audio_sequence_mask, _ = self._encode_audio(audio_inputs, audio_mask)
        audio_sequence, audio_sequence_mask = self._prepend_cls(
            audio_sequence,
            audio_sequence_mask,
            self.audio_cls,
        )
        audio_context_sequence, audio_context_mask, _ = self._encode_audio(
            audio_context_inputs,
            audio_context_mask,
        )
        audio_context_sequence, audio_context_mask = self._prepend_cls(
            audio_context_sequence,
            audio_context_mask,
            self.audio_cls,
        )
        audio_context = self.audio_context_layer(audio_context_sequence, audio_context_mask)

        video_sequence, video_mask = self._encode_video(
            video_inputs,
            video_mask,
            self.video_encoder,
        )
        video_sequence, video_mask = self._prepend_cls(video_sequence, video_mask, self.video_cls)
        video_context_sequence, video_context_mask = self._encode_video(
            video_context_inputs,
            video_context_mask,
            self.video_encoder,
        )
        video_context_sequence, video_context_mask = self._prepend_cls(
            video_context_sequence,
            video_context_mask,
            self.video_cls,
        )
        video_context = self.video_context_layer(video_context_sequence, video_context_mask)

        iv_text_output = self.iv_text_encoder(text_inputs, text_mask, return_dict=True)
        iv_text, iv_text_mask = self._prepend_cls(
            iv_text_output.last_hidden_state,
            text_mask,
            self.iv_text_cls,
        )
        iv_audio, iv_audio_mask = self._encode_iv_audio(audio_inputs, audio_mask)
        iv_audio, iv_audio_mask = self._prepend_cls(iv_audio, iv_audio_mask, self.iv_audio_cls)
        iv_video, iv_video_mask = self._encode_video(
            video_inputs,
            video_mask[:, 1:],
            self.iv_video_encoder,
        )
        iv_video, iv_video_mask = self._prepend_cls(iv_video, iv_video_mask, self.iv_video_cls)

        audio_guided_attention = self.audio_guide(
            audio_sequence,
            audio_sequence_mask,
            text_sequence,
            text_sequence_mask,
        )
        video_guided_attention = self.video_guide(
            video_sequence,
            video_mask,
            text_sequence,
            text_sequence_mask,
        )

        text_guided = text_sequence
        text_cls = text_sequence[:, 0, :].unsqueeze(1)
        audio_guided = (
            self.guide_original_weight * audio_sequence
            + self.guide_attention_weight * audio_guided_attention
            + self.guide_text_weight * text_cls
        )
        video_guided = (
            self.guide_original_weight * video_sequence
            + self.guide_attention_weight * video_guided_attention
            + self.guide_text_weight * text_cls
        )

        iv_text = self.iv_text_projector(iv_text, iv_text_mask)
        iv_audio = self.iv_audio_projector(iv_audio, iv_audio_mask)
        iv_video = self.iv_video_projector(iv_video, iv_video_mask)

        text_guided_cls = text_guided[:, 0, :]
        audio_guided_cls = audio_guided[:, 0, :]
        video_guided_cls = video_guided[:, 0, :]
        text_iv_cls = iv_text[:, 0, :]
        audio_iv_cls = iv_audio[:, 0, :]
        video_iv_cls = iv_video[:, 0, :]

        text_pure = self.text_pure_head(text_guided_cls)
        text_bias = self.text_bias_head(text_guided_cls)
        audio_pure = self.audio_pure_head(audio_guided_cls)
        audio_bias = self.audio_bias_head(audio_guided_cls)
        video_pure = self.video_pure_head(video_guided_cls)
        video_bias = self.video_bias_head(video_guided_cls)

        text_align, text_orth, text_recon = self._disentanglement_loss(
            text_pure,
            text_bias,
            text_iv_cls,
            text_guided_cls,
        )
        audio_align, audio_orth, audio_recon = self._disentanglement_loss(
            audio_pure,
            audio_bias,
            audio_iv_cls,
            audio_guided_cls,
        )
        video_align, video_orth, video_recon = self._disentanglement_loss(
            video_pure,
            video_bias,
            video_iv_cls,
            video_guided_cls,
        )

        rho = torch.sigmoid(self.rho)
        text_final = rho * text_pure + (1 - rho) * text_iv_cls
        audio_final = rho * audio_pure + (1 - rho) * audio_iv_cls
        video_final = rho * video_pure + (1 - rho) * video_iv_cls

        fused = torch.cat(
            (
                text_final,
                text_context[:, 0, :],
                audio_final,
                audio_context[:, 0, :],
                video_final,
                video_context[:, 0, :],
            ),
            dim=1,
        )
        prediction = self.regressor(self.fusion(fused))

        return {
            "prediction": prediction,
            "L_pure": text_align + audio_align + video_align,
            "L_bias": text_orth + audio_orth + video_orth,
            "L_complete": text_recon + audio_recon + video_recon,
            "rho": rho.detach(),
        }
