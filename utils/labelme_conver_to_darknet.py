
import os
import json
import numpy as np

from glob import glob
from time import sleep
from tqdm import tqdm
from shutil import copyfile, rmtree, move

class DarknetDatasetParser(object):
    def __init__(self, class_dict, data_dir):
        super().__init__()
        self._class_dict = class_dict
        self._data_dir = data_dir

    def _generate_train_instance(self, json_file):
        filename = os.path.basename(json_file)
        basename = os.path.splitext(filename)[0]
        basedir = os.path.dirname(json_file)
        jpeg_file = os.path.join(basedir, basename + '.jpg')
        # 1. 确认相关图片文件存在
        if not os.path.exists(jpeg_file):
            raise Exception('% 路径不存在.' % jpeg_file)
        
        if not os.path.exists(self._data_dir) or not os.path.isdir(self._data_dir):
            os.makedirs(self._data_dir)
        
        # 2. 拷贝文件到指定路径
        jpeg_target_file = os.path.join(self._data_dir, basename + '.jpg')
        copyfile(jpeg_file, jpeg_target_file)

        # 3. 从标签文件中读取ground truth标签信息，并写入到与图片文件同目录的同名txt文件中
        with open(json_file, 'r') as f:
            json_text = json.load(f)

        width = json_text['imageWidth']
        height = json_text['imageHeight']
        shapes = json_text['shapes']
        
        # txt文件使用utf-8编码
        txt_file = os.path.join(self._data_dir, basename + '.txt')
        with open(txt_file, 'w', encoding='utf-8') as f:
            for mark in shapes:
                class_name = mark['label']
                if class_name not in self._class_dict:
                    continue # 忽略不存在的类别
                
                # bbox的所属class name索引，取值范围在[0, classes - 1]
                class_idx = self._class_dict[class_name]
                # bbox的坐标，格式为[[x1, y1], [x2, y2]], shape=(2, 2)
                bbox_coords = mark['points']
                bbox_coords = np.array(bbox_coords, dtype=np.float)
                # Boundingbox
                x = bbox_coords[:, 0]
                y = bbox_coords[:, 1]
                xmin = np.min(x)
                xmax = np.max(x)
                ymin = np.min(y)
                ymax = np.max(y)
                center_x = (xmin + xmax) / 2. / width
                center_y = (ymin + ymax) / 2. / height
                bbox_width = (xmax - xmin) / width
                bbox_height = (ymax - ymin) / height
                
                # 写入类别索引、bbox坐标的格式：
                # <object-class> <x_center> <y_center> <width> <height>
                # <object-class>-从0到的整数对象编号(classes-1)
                # <x_center> <y_center> <width> <height>-浮动值相对于图片的宽度和高度，
                # 可以等于(0.0 to 1.0]
                # 注意：<x_center> <y_center>-矩形的中心（不是左上角）
                bbox = [str(class_idx), str(center_x), str(center_y),
                         str(bbox_width), str(bbox_height)]
                
                f.write(' '.join(bbox) + '\n')
        
        return jpeg_target_file   

    def parse_yolomark_txt(self, file_list, out_file):
        feat_cnt = 0
        with open(out_file, 'w', encoding='utf-8') as f:
            for i in tqdm(range(len(file_list)), ncols=75):
                json_file = file_list[i]
                # 生成data/img/xxx.jpg 和 data/img/xxx.txt
                jpeg_file = self._generate_train_instance(json_file)
                # 写入图片路径到data/train.txt 或 data/val.txt
                f.write(jpeg_file + '\n')
                sleep(0.01)


def main():
    data_dir = 'C:/Code/yolov3/data'
    train_dir = 'C:/Code/训练数据/海东检测/0_数据_train'
    val_dir = 'C:/Code/训练数据/海东检测/0_数据_val'
    class_names_file = 'C:/Code/tfLearn/data/my_data.names'
    
    # 拷贝class.names到 data目录下
    copyfile(class_names_file, os.path.join(data_dir, os.path.basename(class_names_file)))
    
    train_file_list = glob(os.path.join(train_dir, '**/*.json'), recursive=True)
    val_file_list = glob(os.path.join(val_dir, '**/*.json'), recursive=True)
    train_file_path = os.path.join(data_dir, 'train.txt')
    val_file_path = os.path.join(data_dir, 'val.txt')
    
    # 从文件读取类别列表
    class_index = 0
    class_dict = {}
    with open(class_names_file, 'r') as f:
        lines = f.readlines()
        for class_name in lines:
            class_name = class_name.strip()
            if class_name not in class_dict:
                class_dict[class_name] = class_index
                class_index += 1
    
    ds_parser = DarknetDatasetParser(class_dict, data_dir + '/img')
    ds_parser.parse_yolomark_txt(train_file_list, train_file_path)
    ds_parser.parse_yolomark_txt(val_file_list, val_file_path)


if __name__ == '__main__':
    main()