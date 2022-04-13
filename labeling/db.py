import os
from shutil import copyfile
import pandas as pd
from pandas.io.formats.format import return_docstring


class Table:
    path = ''
    index_name = ''
    df = None

    def __init__(self, index_name, path, sep = ','):
        self.index_name = index_name
        self.path = path
        if not os.path.isfile(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            self.df = pd.DataFrame(columns=[index_name])
            self.df.set_index(self.index_name)
            self.df.to_csv(path)
        else:
            self.df = pd.read_csv(self.path, header=0, sep=sep)
            self.df = self.df.set_index(self.index_name)

    def set_table(self, df):
        if df.index.name != self.index_name:
            return
        self.df = df.copy()
        self.save()

    def add_value(self, row, col, value):
        self.df.at[row, col] = value
        self.save()
    
    def add_row(self, value, ignore_index=True):
        self.df = self.df.append(value, ignore_index=ignore_index)
        self.df.index.name = self.index_name
        self.save()
    
    def remove_row(self, index):
        self.df = self.df.drop(index)
        self.save()
    
    def save(self):
        self.df.to_csv(self.path)

class PhotoTable(Table):
    class_col = 'class'

    def __init__(self, path, sep = ','):
        self.index_name = 'id'
        self.path = path
        if not os.path.isfile(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            self.df = pd.DataFrame(columns=[self.index_name, self.class_col])
            self.df = self.df.set_index(self.index_name)
            self.df.to_csv(path)
        else:
            self.df = pd.read_csv(self.path, header=0, sep=sep)
            self.df = self.df.set_index(self.index_name)
    
    def index(self, object_id, photo_num):
        return object_id.__str__() + '_' + photo_num.__str__()
    
    def data_from_index(self, index):
        temp = index.split('_')
        return int(temp[0]), int(temp[1])

    def set_class(self, object_id, photo_num, class_name):
        if class_name is None:
            return
        index = self.index(object_id, photo_num)
        if index in self.df.index:
            self.add_value(index, self.class_col, class_name)
        else:
            df = pd.DataFrame(data={self.index_name: [index], self.class_col: [class_name]})
            df = df.set_index(self.index_name)
            self.add_row(df, ignore_index=False)
    
    def remove_class(self, object_id, photo_num):
        index = self.index(object_id, photo_num)
        if index in self.df.index:
            self.remove_row(index)
    
    def get_class(self, object_id, photo_num):
        if object_id is None or photo_num is None:
            return None
        index = self.index(object_id, photo_num)
        if index not in self.df.index:
            return None
        return self.df.at[index, self.class_col]

    def get_last_data(self):
        if len(self.df.index) == 0:
            return None, None
        return self.data_from_index(list(self.df.index)[-1])
