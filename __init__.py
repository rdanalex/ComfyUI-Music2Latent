from .nodes import AudioVAEDecodeNoAutoGain, Music2LatentDecode, Music2LatentEncode


NODE_CLASS_MAPPINGS = {
    "AudioVAEDecodeNoAutoGain": AudioVAEDecodeNoAutoGain,
    "Music2LatentEncode": Music2LatentEncode,
    "Music2LatentDecode": Music2LatentDecode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AudioVAEDecodeNoAutoGain": "Audio VAE Decode (No Auto Gain)",
    "Music2LatentEncode": "Music2Latent Encode",
    "Music2LatentDecode": "Music2Latent Decode",
}