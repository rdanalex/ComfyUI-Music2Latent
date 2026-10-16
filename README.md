# ComfyUI Music2Latent Nodes

Adds `Audio VAE Decode (No Auto Gain)`, plus a matching `Music2Latent Encode` /
`Music2Latent Decode` pair for ComfyUI's `AUDIO` type.

`Audio VAE Decode (No Auto Gain)` accepts the same `LATENT` and `VAE` inputs as
the built-in audio decoder. It omits ComfyUI's post-decode standard-deviation
gain adjustment, so it is the direct alternative to try with an existing model
workflow. It does not change the VAE's own reconstruction.

The Music2Latent pair uses the standalone Music2Latent audio codec and its own
`MUSIC2LATENT` intermediate type.

Music2Latent does **not** replace ComfyUI's standard audio VAE decoder:
Music2Latent latents are incompatible with audio latents produced by ComfyUI
model VAEs. Encoding and decoding audio through any autoencoder is lossy; the
round trip is not guaranteed to preserve the source perfectly or eliminate
artifacts. The no-auto-gain VAE node only skips ComfyUI's post-decode level
adjustment; compare output loudness and clipping with the built-in node.

## Install

Copy or symlink this directory into `ComfyUI/custom_nodes/ComfyUI-Music2Latent`,
then install `requirements.txt` into the Python environment used by ComfyUI and
restart ComfyUI. The model checkpoint downloads automatically on first use.

The upstream Music2Latent package is licensed under CC BY-NC 4.0. Check that
license before using it for commercial work.