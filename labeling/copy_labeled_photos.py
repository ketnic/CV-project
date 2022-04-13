import os
import shutil
import pandas as pd


photos_csv = './photos.csv'
input_dir = './photos'
output_dir = './output'

df = pd.read_csv(photos_csv)
df = df.set_index('id')
df['object_id'] = df.index.to_series().apply(lambda x: x.split('_')[0])
df['photo_num'] = df.index.to_series().apply(lambda x: x.split('_')[1])
df['output_dir'] = df.apply(lambda row: '{}/{}'.format(output_dir, row['object_id']), axis=1)
df['input_path'] = df.apply(lambda row: '{}/{}/{}.jpeg'.format(input_dir, row['object_id'], row['photo_num']), axis=1)
df['output_path'] = df.apply(lambda row: '{}/{}/{}.jpeg'.format(output_dir, row['object_id'], row['photo_num']), axis=1)

print(df)

if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
    print('Output dir removed')

os.mkdir(output_dir)

print('Copy start')

for _, row in df.iterrows():
    if not os.path.exists(row['output_dir']):
        os.mkdir(row['output_dir'])
    shutil.copyfile(row['input_path'], row['output_path'])

print('Copy complete!')
