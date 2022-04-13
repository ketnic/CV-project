import os
import pandas as pd


from config import FLAT_TYPES_CSV, SIMILAR_PHOTOS, MERGED_DATASET_CSV


def merge_datasets(strict_mode=True):
    df = pd.read_csv(FLAT_TYPES_CSV)
    df['room_num'] = None

    flats = os.listdir(SIMILAR_PHOTOS)
    for flat_id in flats:
        flat_path = os.path.sep.join([SIMILAR_PHOTOS, flat_id]) # ссылка на квартиру
        flat_id = int(flat_id)
        rooms = os.listdir(flat_path) # список комнат
        for room_num in rooms:
            room_path = os.path.sep.join([flat_path, room_num]) # ссылка на комнату
            room_photos = os.listdir(room_path) # список фото в комнате
            for photo_num in room_photos:
                photo_num = os.path.splitext(os.path.basename(photo_num))[0]
                photo_num = int(photo_num)
                condition = (df['photo_num'] == photo_num) & (df['flat_id'] == flat_id)
                if len(df[condition]) == 1:
                    index = df.loc[condition].index[0]
                    df.iat[index, 3] = room_num
                elif not strict_mode:
                    df = df.append({'flat_id': flat_id, 'photo_num': photo_num, 'class': None, 'room_num': room_num}, ignore_index=True)
    if strict_mode:
        df = df[~df['room_num'].isnull()]
    df = df.sort_values(['flat_id', 'photo_num'], ascending=[True, True])
    df.to_csv(MERGED_DATASET_CSV, index=False)
