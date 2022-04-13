import asyncio

from networks.module import group_images as _group_images, ModuleScheme


async def group_images(image_paths, img_c, img_sim, success_callback, progress_callback):
    result = _group_images(image_paths, ModuleScheme.ClassifierSiamese, progress_callback, img_c, img_sim)

    if success_callback:
        success_callback(result)
