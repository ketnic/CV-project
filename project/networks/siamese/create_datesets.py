import os
import random
import shutil
import tensorflow as tf

from networks.siamese.CustomDataset import CustomDataset

from utils.tools.log import log
from utils.tools.convertible import join_to_path
from utils.tools.inline_print import inline_print, complete_inline_print
from utils.read_and_process_image import process_and_save_image


# Создает test и train выборки
def _create_samples(source, dist, balanced, image_size, paired_with_itself=True):
    # Обрабатываем фотографии и кэшируем
    cache = join_to_path([os.path.dirname(dist), '_' + os.path.basename(dist)])
    log("photo processing...")
    _process_photos(
        source=source,
        dist=cache,
        image_size=image_size
    )
    # Создаем выборку из обработанных фото и кэшируем
    log("photo pairing...")
    _pair_photos(
        cache=cache,
        dist=dist,
        paired_with_itself=paired_with_itself,
        balanced=balanced
    )


# Обрабатывает и кэширует фото
def _process_photos(source, dist, image_size):
    if os.path.exists(dist):
        if  len(os.listdir(source)) == len(os.listdir(dist)):
            log('using cached photos')
            return
        log('removing cached photos...')
        shutil.rmtree(dist)
    os.makedirs(dist)
    if not os.path.exists(source):
        raise Exception('\'{}\' doesn\'t exist'.format(source))
    flats = os.listdir(source)
    count = len(flats)
    for i, flat in enumerate(flats):
        flat_path = join_to_path([source, flat]) # ссылка на квартиру
        rooms = os.listdir(flat_path) # список комнат
        inline_print(f'{i + 1}/{count} flats')
        for room in rooms:
            room_path = join_to_path([flat_path, room]) # ссылка на комнату
            room_photos = os.listdir(room_path) # список фото в комнате
            for room_photo in room_photos:
                room_photo_path = join_to_path([room_path, room_photo]) # ссылка на фото в комнате
                processed_photo_path = join_to_path([dist, flat, room, room_photo])
                os.makedirs(os.path.dirname(processed_photo_path), exist_ok=True)
                process_and_save_image(
                    img_path=room_photo_path, 
                    save_path=processed_photo_path, 
                    img_size=image_size
                )
    complete_inline_print()


# Создает выборки из обработанных фото и кэширует
def _pair_photos(cache, dist, paired_with_itself=False, balanced=True):
    if os.path.exists(dist) and not len(os.listdir(dist)) == 0:
        log('using cached pairs')
        return
    # Создаем позитивные и негативные пары фото
    flats = os.listdir(cache)
    pos = []
    neg = []
    for flat in flats:
        flat_path = join_to_path([cache, flat])  # ссылка на квартиру
        rooms = os.listdir(flat_path)  # список комнат в квартире
        # Создаем список ссылок на комнаты
        rooms_paths = list(map(lambda room: join_to_path([flat_path, room]), rooms))
        # Создаем список типа (ссылка на комнату, список фото в этой комнате)
        rooms_photos = list(map(lambda room_path: (room_path, os.listdir(room_path)), rooms_paths))
        # Создаем список, где элемент - список ссылок на фотографии комнаты в комнате
        rooms_photos_paths = list(map(lambda rooms_photo: list(map(lambda photo: join_to_path([rooms_photo[0], photo]), rooms_photo[1])), rooms_photos))
        # Создаем пары комнат (комната == массив фото этой комнаты)
        paired_rooms_photos_paths = _make_uniq_pairs(rooms_photos_paths, rooms_photos_paths)
        for paired_rooms in paired_rooms_photos_paths:
            left_room, right_room = paired_rooms
            if left_room[0] == right_room[0]:
                pos = pos + _make_uniq_pairs(left_room, right_room, itself=paired_with_itself)
            else:
                neg = neg +  _make_uniq_pairs(left_room, right_room, itself=paired_with_itself)
    # Сбалансировано отбираем рандомные негативные и позитивные пары (по хорошему это надо реализовать на уровне датасета)
    count = None
    sum = None
    if balanced:
        count = min(len(pos), len(neg))
        random.shuffle(pos)
        random.shuffle(neg)
        sum = list(zip(pos, [1] * count)) + list(zip(neg, [0] * count))
    else:
        count = len(pos) + len(neg)
        sum = list(zip(pos, [1] * len(pos))) + list(zip(neg, [0] * len(neg)))
    for i, ((left_photo, right_photo), value) in enumerate(sum):
        temp, left_photo_name = os.path.split(left_photo)
        temp, left_photo_room = os.path.split(temp)
        temp, left_photo_flat = os.path.split(temp)
        dist_left_photo = join_to_path([dist, str(value), str(i), left_photo_flat + '_' + left_photo_room + '_' + left_photo_name + '_left'])
        os.makedirs(os.path.dirname(dist_left_photo), exist_ok=True)
        shutil.copy(left_photo, dist_left_photo)
        temp, right_photo_name = os.path.split(right_photo)
        temp, right_photo_room = os.path.split(temp)
        temp, right_photo_flat = os.path.split(temp)
        dist_right_photo = join_to_path([dist, str(value), str(i), right_photo_flat + '_' + right_photo_room + '_' + right_photo_name + '_right'])
        os.makedirs(os.path.dirname(dist_right_photo), exist_ok=True)
        shutil.copy(right_photo, dist_right_photo)


# Создает уникальные пары элементов
def _make_uniq_pairs(left_arr, right_arr, itself=True):
    result = []
    for i, left_elem in enumerate(left_arr):
        for right_elem in right_arr[i if itself else i + 1:]:
            result.append((left_elem, right_elem))
    return result


# def _scale_image(image):
#     image = tf.image.per_image_standardization(image)
#     return image


def create_datasets(source, dist, balanced, image_size, dataset_dir, validation_split, paired_with_itself=True):
    log('sampling...')
    _create_samples(source, dist, balanced, image_size, paired_with_itself=paired_with_itself)
    
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)

    ds = CustomDataset(path=dist)
    ds.download_and_prepare(download_dir=dataset_dir)

    ds = ds.as_dataset()['train']
    ds = ds.map(lambda x: ((x['input_1_left'], x['input_2_right']), x['label']))
    # ds = ds.batch(32)
    # ds = ds.shuffle(buffer_size=32)

    DATASET_SIZE = ds.__len__().numpy()
    train_size = int(DATASET_SIZE * (1 - validation_split))
    train_ds = ds.take(train_size)
    val_ds = ds.skip(train_size)

    return train_ds, val_ds
