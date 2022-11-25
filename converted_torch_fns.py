from typing import Callable
import jax.numpy as jnp
import PIL
import numpy as np
import jax

# This is a JAX port of https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_inpaint.py
def prepare_mask_and_masked_image(image, mask):
    """
    Prepares a pair (image, mask) to be consumed by the Stable Diffusion pipeline.
    """
    if isinstance(image, jnp.ndarray):
        if not isinstance(mask, jnp.ndarray):
            raise TypeError(f"`image` is a jnp.ndarray but `mask` (type: {type(mask)} is not")

        # Batch single image
        if image.ndim == 3:
            assert image.shape[0] == 3, "Image outside a batch should be of shape (3, H, W)"
            image = image[None]

        # Batch and add channel dim for single mask
        if mask.shape == 2:
            mask = mask[None, None]

        # Batch single mask or add channel dim
        if mask.ndim == 3:
            # Single batched mask, no channel dim or single mask not batched but channel dim
            if mask.shape[0] == 1:
                mask = mask[None]

            # Batched masks no channel dim
            else:
                mask = mask[:, None]

        assert image.ndim == 4 and mask.ndim == 4, "Image and Mask must have 4 dimensions"
        assert (
            image.shape[-2:] == mask.shape[-2:]
        ), "Image and Mask must have the same spatial dimensions"
        assert image.shape[0] == mask.shape[0], "Image and Mask must have the same batch size"

        # Check image is in [-1, 1]
        if image.min() < -1 or image.max() > 1:
            raise ValueError("Image should be in [-1, 1] range")

        # Check mask is in [0, 1]
        if mask.min() < 0 or mask.max() > 1:
            raise ValueError("Mask should be in [0, 1] range")

        # Binarize mask
        mask[mask < 0.5] = 0
        mask[mask >= 0.5] = 1

        # Image as float32
        image = image.astype(np.float32)
    elif isinstance(mask, jnp.ndarray):
        raise TypeError(f"`mask` is a jnp.ndarray but `image` (type: {type(image)} is not")
    else:
        if isinstance(image, PIL.Image.Image):
            image = np.array(image.convert("RGB"))
        image = image[None].transpose(0, 3, 1, 2)
        image = jnp.array(image).astype(np.float32) / 127.5 - 1.0
        if isinstance(mask, PIL.Image.Image):
            mask = np.array(mask.convert("L"))
            mask = mask.astype(np.float32) / 255.0
        mask = mask[None, None]
        mask[mask < 0.5] = 0
        mask[mask >= 0.5] = 1
        mask = jnp.array(mask)

    masked_image = image * (mask < 0.5)

    return mask, masked_image


# https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_inpaint.py
def prepare_mask_latents(
    self,
    mask: jax.Array,
    masked_image,
    batch_size,
    height,
    width,
    generator: Callable[[int], jax.random.KeyArray],
    do_classifier_free_guidance,
):
    # resize the mask to latents shape as we concatenate the mask to the latents
    # we do that before converting to dtype to avoid breaking in case we're using cpu_offload
    # and half precision
    mask = jax.image.resize(
        mask,
        shape=(height // self.vae_scale_factor, width // self.vae_scale_factor),
        method="nearest",
    )
    # mask = mask.to(device=device, dtype=dtype)

    # masked_image = masked_image.to(device=device, dtype=dtype)

    # encode the mask image into latents space so we can concatenate it to the latents
    masked_image_latents = self.vae.encode(masked_image).latent_dist.sample(generator=generator)
    masked_image_latents = 0.18215 * masked_image_latents

    # duplicate mask and masked_image_latents for each generation per prompt, using mps friendly method
    mask = jnp.tile(mask, (batch_size, 1, 1, 1))
    masked_image_latents = jnp.tile(masked_image_latents, (batch_size, 1, 1, 1))

    mask = jnp.concatenate([mask] * 2) if do_classifier_free_guidance else mask
    masked_image_latents = (
        jnp.concatenate([masked_image_latents] * 2)
        if do_classifier_free_guidance
        else masked_image_latents
    )

    # aligning device to prevent device errors when concating it with the latent model input
    # masked_image_latents = masked_image_latents.to(device=device, dtype=dtype)
    return mask, masked_image_latents
