import os
import shutil
import pandas as pd

from utils.tools.log import log
from utils.tools.inline_print import inline_print, complete_inline_print
from utils.read_and_process_image import process_and_save_image


def process_photos(source, train, val, val_split, csv, image_size, balance):
    if os.path.exists(train) and os.listdir(train) and os.path.exists(val) and os.listdir(val):
        log('reuse the processed photos')
        return

    df = pd.read_csv(csv)
    df['photo_path'] = df.apply(lambda row: '{}\\{}\\{}.jpeg'.format(source, row['flat_id'], row['photo_num']), axis=1)

    flats = df['flat_id'].unique()
    threshold = int(val_split * len(flats))

    log('process for train')
    _process_photos(df[df['flat_id'].isin(flats[threshold:])], source, train, image_size, balance)
    log('process for validation')
    _process_photos(df[df['flat_id'].isin(flats[:threshold])], source, val, image_size, balance)


def _process_photos(df, source, dist, image_size, balance):
    max = None
    if balance:
        max = df.groupby('class').count().max().values[0]
        log(f'images per class: {max}')

    # Распределяем фото по папкам-класса + обрабатываем
    i = 0
    count = len(df['class'].unique()) * max if balance else len(df)
    for class_name in df['class'].unique():
        # Обрабатываем фото определенного класса
        temp = df[df['class'] == class_name]
        # Балансируем данные в классах
        while balance and len(temp.index) < max:
            temp = pd.concat([temp, temp.sample(frac=0.5)], ignore_index=True)
        if balance and len(temp.index) > max:
            temp = temp.head(max)
        # Обрабатываем и сохраняем в dist (кэш)
        for id, row in temp.iterrows():
            source = row['photo_path']
            photo = os.path.sep.join([dist, row['class'], f'{id}.jpeg'])
            os.makedirs(os.path.dirname(photo), exist_ok=True)
            shutil.copy(source, photo)
            process_and_save_image(photo, img_size=image_size)
            i += 1
            inline_print(f'{i}/{count}')
    complete_inline_print()
