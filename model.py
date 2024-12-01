def model(prompt: str) -> bytes:
    """
    ``prompt``: ``str`` type. The prompt to be processed.

    ``return``: ``bytes`` type.
    """
    # TODO: Implement your model here.

    with open(prompt, 'rb') as image_file:
        image_bytes = image_file.read()

        return image_bytes
