import torch
import torchaudio
import comfy.model_management


_encoder_decoder = None
_encoder_decoder_device = None


def _get_encoder_decoder():
    global _encoder_decoder, _encoder_decoder_device

    device = comfy.model_management.get_torch_device()
    if _encoder_decoder is None or _encoder_decoder_device != device:
        try:
            from music2latent import EncoderDecoder
        except ImportError as exc:
            raise RuntimeError(
                "Music2Latent is not installed in ComfyUI's Python environment. "
                "Install this node's requirements.txt or run: pip install music2latent"
            ) from exc

        _encoder_decoder = EncoderDecoder(device=device)
        _encoder_decoder_device = device

    return _encoder_decoder


def _resample(waveform, source_rate, target_rate):
    if source_rate == target_rate:
        return waveform
    return torchaudio.functional.resample(waveform, source_rate, target_rate)


class AudioVAEDecodeNoAutoGain:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"samples": ("LATENT",), "vae": ("VAE",)}}

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "decode"
    CATEGORY = "latent/audio"

    def decode(self, vae, samples):
        waveform = vae.decode(samples["samples"]).movedim(-1, 1)
        return ({"waveform": waveform, "sample_rate": 44100},)


class Music2LatentEncode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "max_waveform_seconds": (
                    "INT",
                    {"default": 10, "min": 1, "max": 60, "step": 1},
                ),
            }
        }

    RETURN_TYPES = ("MUSIC2LATENT",)
    RETURN_NAMES = ("music2latent",)
    FUNCTION = "encode"
    CATEGORY = "audio/music2latent"

    def encode(self, audio, max_waveform_seconds):
        waveform = audio["waveform"].detach().float()
        if waveform.ndim == 2:
            waveform = waveform.unsqueeze(0)
        if waveform.ndim != 3:
            raise ValueError("AUDIO waveform must have shape [batch, channels, samples].")

        batch_size, channel_count, _ = waveform.shape
        if channel_count not in (1, 2):
            raise ValueError("Music2Latent supports mono or stereo audio input.")

        sample_rate = int(audio["sample_rate"])
        max_waveform_length = max_waveform_seconds * 44100
        model = _get_encoder_decoder()
        latent_samples = []
        encoded_lengths = []

        for batch_index in range(batch_size):
            sample = _resample(waveform[batch_index], sample_rate, 44100)
            encoded_lengths.append(sample.shape[-1])
            model_input = sample[0] if channel_count == 1 else sample
            latent = model.encode(
                model_input,
                max_waveform_length=max_waveform_length,
            )
            if not isinstance(latent, torch.Tensor):
                latent = torch.as_tensor(latent)
            if latent.ndim == 2:
                latent = latent.unsqueeze(0)
            latent_samples.append(latent.detach().float().cpu())

        return (
            {
                "samples": torch.stack(latent_samples),
                "sample_rate": sample_rate,
                "original_lengths": [waveform.shape[-1]] * batch_size,
                "encoded_lengths": encoded_lengths,
                "channels": channel_count,
            },
        )


class Music2LatentDecode:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"music2latent": ("MUSIC2LATENT",)}}

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "decode"
    CATEGORY = "audio/music2latent"

    def decode(self, music2latent):
        model = _get_encoder_decoder()
        samples = music2latent["samples"]
        sample_rate = music2latent["sample_rate"]
        original_lengths = music2latent["original_lengths"]
        encoded_lengths = music2latent["encoded_lengths"]
        channel_count = music2latent["channels"]
        decoded_samples = []

        for batch_index, latent in enumerate(samples):
            decoded = model.decode(latent)
            if not isinstance(decoded, torch.Tensor):
                decoded = torch.as_tensor(decoded)
            decoded = decoded.detach().float().cpu()
            if decoded.ndim == 3 and decoded.shape[0] == 1:
                decoded = decoded.squeeze(0)
            if decoded.ndim == 1:
                decoded = decoded.unsqueeze(0)
            elif decoded.ndim == 2 and decoded.shape[-1] == channel_count:
                decoded = decoded.transpose(0, 1)
            elif decoded.ndim != 2 or decoded.shape[0] != channel_count:
                raise ValueError("Music2Latent returned an unexpected decoded audio shape.")

            decoded = decoded[..., : encoded_lengths[batch_index]]
            decoded = _resample(decoded, 44100, sample_rate)
            decoded = decoded[..., : original_lengths[batch_index]]
            if decoded.shape[-1] < original_lengths[batch_index]:
                decoded = torch.nn.functional.pad(
                    decoded, (0, original_lengths[batch_index] - decoded.shape[-1])
                )
            decoded_samples.append(decoded)

        return (
            {
                "waveform": torch.stack(decoded_samples),
                "sample_rate": sample_rate,
            },
        )