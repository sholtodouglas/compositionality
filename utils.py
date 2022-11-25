import PIL


def generate_mask(image: PIL.Image) -> PIL.Image:
    """
    Creates a mask. White means replace, black means keep.
    blended is a preview of the picture for debugging purposes.
    This function is probably not the fastest way of doing this at scale,
    jax.numpy can probably just create a giant vector of tensors with all the masks.
    """
    # Base mask is all black (keep)
    mask = PIL.Image.new("RGB", image.size, 0)
    white_mask = PIL.Image.new("RGB", (100, 100), "white")
    PIL.Image.Image.paste(mask, white_mask, (200, 200))
    blended = PIL.Image.blend(image, mask, 0.5)
    return mask, blended


mask_image, blended = generate_mask(init_image)
blended
