# -*- coding: utf-8 -*-

import os
import struct
import numpy as np
import PIL
import scipy.io as scio
from tkinter import *
from tkinter.filedialog import *
from osgeo import gdal
import matplotlib.pyplot as plt


def is_typed_file(full_file_name, exts=('.bmp', '.jpg', '.png', '.tif', '.tiff', '.jpeg', '.pgm')):
    """
    判断给定文件是否是某种类型的文件
    :param full_file_name: 文件全名
    :param exts: 某种文件类型的后缀名列表, 以'.'开头, 全部小写, 如('.jpg',)
    :return: 判定结果, bool值
    """
    if full_file_name is None or full_file_name == '' or exts is None or exts == '':
        return False
    ext = os.path.splitext(full_file_name)[1].lower()
    for e in exts:
        if e == ext:
            return True
    return False


def read_hyperspectral_file(data_file_name, label_file_name=None,
                            dest_directory=os.path.normpath(os.path.join(os.getcwd(), 'Data')),
                            normalized=True):
    """
    读取高光谱数据文件和标签文件，按照统一格式输出到一个整合后的mat文件中;
    输入可以是mat格式文件，也可以是其它图像格式的文件;
    整合输出后的mat文件格式为:
    WWL: 标准mat文件标志, 如果存在该字段且该字段的值为'WWLMat', 则该文件为标准化后的mat文件, 必选项
    Data: 存放数据, 维度为(行X列, 波段数)的二维数组, 32位float类型, 必选项
    Labels: 存放Data对应的标签, 维度为(行X列,)的一维数组, 32位int类型, 可选项
    LabelValues: 标签集, 标签值从小到大排列的维度为(类别数目,)的一维数组, 32位int类型, 可选项
    LabelNames: 存放标签名称, 维度为(类别数目,)的一维数组, string类型, 可选项
    LabelColors: 存放标签颜色, 维度为(类别数目, 3)的二维数组, 32位int类型的[Red, Green, Blue]颜色值, 可选项
    ImageShape: 存放图像形状, [图像高度(行数), 图像宽度(列数)], 32位int类型, 可选项
    :param data_file_name: 高光谱数据文件全名
    :param label_file_name: 高光谱标签文件全名
    :param dest_directory: 输出mat文件的存放目录, 默认为当前工作目录
    :param normalized: 是否将读取的高光谱数据进行归一化, 默认为True
    :return: 按照统一格式整合后的mat字典
    """
    if data_file_name is None or data_file_name == '':
        return None
    data_file_name = os.path.normpath(data_file_name)
    data = None
    labels = None
    label_values = None
    label_names = None
    label_colors = None
    image_shape = None
    if is_typed_file(data_file_name):
        dataset = gdal.Open(data_file_name, gdal.GA_ReadOnly)
        if dataset is not None:
            data = np.zeros((dataset.RasterYSize * dataset.RasterXSize, dataset.RasterCount),
                            dtype=np.float32)
            for i in range(dataset.RasterCount):
                band = dataset.GetRasterBand(i + 1)
                band_str = band.ReadRaster(yoff=0, xoff=0, ysize=band.YSize, xsize=band.XSize,
                                           buf_ysize=band.YSize, buf_xsize=band.XSize, buf_type=gdal.GDT_Float32)
                band_value = struct.unpack('f' * band.YSize * band.XSize, band_str)
                data[:, i] = np.array(band_value, dtype=np.float32)
            image_shape = np.array([dataset.RasterYSize, dataset.RasterXSize], dtype=np.int32)
    elif is_typed_file(data_file_name, ('.mat',)):
        mat = scio.loadmat(data_file_name)
        if 'WWL' in mat and mat['WWL'] == 'WWLMat':
            data = mat['Data']
            if 'Labels' in mat:
                labels = mat['Labels']
            if 'LabelValues' in mat:
                label_values = mat['LabelValues']
            if 'LabelNames' in mat:
                label_names = mat['LabelNames']
            if 'LabelColors' in mat:
                label_colors = mat['LabelColors']
            if 'ImageShape' in mat:
                image_shape = mat['ImageShape']
        elif isinstance(list(mat.values())[3], np.ndarray):
            data_temp = list(mat.values())[3]
            dims = data_temp.shape
            if 3 == len(dims):
                data = np.array(data_temp.reshape([dims[0]*dims[1], dims[2]]), dtype=np.float32)
                image_shape = np.array([dims[0], dims[1]], dtype=np.int32)
            elif 2 == len(dims):
                data = np.array(data_temp.reshape([dims[0]*dims[1], 1]), dtype=np.float32)
                image_shape = np.array([dims[0], dims[1]], dtype=np.int32)
    if data is None:
        return None
    if label_file_name is not None and label_file_name != '':
        label_file_name = os.path.normpath(label_file_name)
        if is_typed_file(label_file_name):
            dataset = gdal.Open(label_file_name, gdal.GA_ReadOnly)
            if dataset is not None:
                band = dataset.GetRasterBand(1)
                band_str = band.ReadRaster(yoff=0, xoff=0, ysize=band.YSize, xsize=band.XSize,
                                           buf_ysize=band.YSize, buf_xsize=band.XSize, buf_type=gdal.GDT_Float32)
                band_value = struct.unpack('f' * band.YSize * band.XSize, band_str)
                labels_temp = np.array(band_value, dtype=np.int32)
                if data.shape[0] == labels_temp.shape[0]:
                    labels = labels_temp
                    label_values = []
                    for label in labels:
                        if label not in label_values:
                            label_values.append(label)
                    label_values.sort()
                    label_values = np.array(label_values, dtype=np.int32)
                    label_colors = np.zeros((len(label_values), 3), dtype=np.int32)
                    for i in range(len(label_values)):
                        if 0 == label_values[i]:
                            label_colors[i] = np.array([0, 0, 0], dtype=np.int32)
                        else:
                            label_colors[i] = np.array(np.random.randint(0, 255, [3]), dtype=np.int32)
        elif is_typed_file(label_file_name, ('.mat',)):
            mat = scio.loadmat(label_file_name)
            if 'WWL' in mat and mat['WWL'] == 'WWLMat':
                if 'Labels' in mat and mat['Labels'].shape[0] == mat['Data'].shape[0]:
                    labels = mat['Labels']
                    if 'LabelValues' in mat:
                        label_values = mat['LabelValues']
                    else:
                        label_values = []
                        for label in labels:
                            if label not in label_values:
                                label_values.append(label)
                        label_values.sort()
                        label_values = np.array(label_values, dtype=np.int32)
                    if 'LabelColors' in mat:
                        label_colors = mat['LabelColors']
                    else:
                        label_colors = np.zeros((len(label_values), 3), dtype=np.int32)
                        for i in range(len(label_values)):
                            if 0 == label_values[i]:
                                label_colors[i] = np.array([0, 0, 0], dtype=np.int32)
                            else:
                                label_colors[i] = np.array(np.random.randint(0, 255, [3]), dtype=np.int32)
            elif isinstance(list(mat.values())[3], np.ndarray):
                labels_temp = list(mat.values())[3]
                dims = labels_temp.shape
                if 2 == len(dims) and data.shape[0] == dims[0]*dims[1]:
                    labels = np.array(labels_temp.reshape([dims[0]*dims[1]]), dtype=np.int32)
                    label_values = []
                    for label in labels:
                        if label not in label_values:
                            label_values.append(label)
                    label_values.sort()
                    label_values = np.array(label_values, dtype=np.int32)
                    label_colors = np.zeros((len(label_values), 3), dtype=np.int32)
                    for i in range(len(label_values)):
                        if 0 == label_values[i]:
                            label_colors[i] = np.array([0, 0, 0], dtype=np.int32)
                        else:
                            label_colors[i] = np.array(np.random.randint(0, 255, [3]), dtype=np.int32)
    if normalized:
        min_vector = np.min(data, axis=0)
        max_vector = np.max(data, axis=0)
        d_vector = max_vector - min_vector
        data = (data - min_vector) / d_vector
    dataset_mat = dict()
    dataset_mat['WWL'] = 'WWLMat'
    dataset_mat['Data'] = data
    if labels is not None:
        dataset_mat['Labels'] = labels
    if label_values is not None:
        dataset_mat['LabelValues'] = label_values
    if label_names is not None:
        dataset_mat['LabelNames'] = label_names
    if label_colors is not None:
        dataset_mat['LabelColors'] = label_colors
    if image_shape is not None:
        dataset_mat['ImageShape'] = image_shape
    file_name, _ = os.path.splitext(data_file_name)
    _, file_name = os.path.split(file_name)
    file_name = file_name + '.mat'
    if dest_directory is None or dest_directory == '':
        dest_directory = os.path.join(os.getcwd(), 'Data')
    dest_directory = os.path.normpath(dest_directory)
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    file_name = os.path.normpath(os.path.join(dest_directory, file_name))
    scio.savemat(file_name, dataset_mat)
    return dataset_mat


def mat_to_tif(mat_file_name,
               dest_directory=os.path.normpath(os.path.join(os.getcwd(), 'Data')),
               normalized=False):
    """
    将非标准化的mat格式高光谱文件转化为Geotiff图像文件
    :param mat_file_name: mat文件全名
    :param dest_directory: 存储tif文件的目标目录
    :param normalized: 转化过程中是否对数据进行归一化, 默认为False
    :return: 无
    """
    if mat_file_name is None or mat_file_name == '':
        return
    if dest_directory is None or dest_directory == '':
        dest_directory = os.path.normpath(os.path.join(os.getcwd(), 'Data'))
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    _, image_file_name = os.path.split(mat_file_name)
    image_file_name, _ = os.path.splitext(image_file_name)
    image_file_name = image_file_name + '.tif'
    image_file_name = os.path.normpath(os.path.join(dest_directory, image_file_name))
    mat_dict = scio.loadmat(mat_file_name)
    data = np.array(list(mat_dict.values())[3], dtype=np.float32)
    dims = data.shape
    band_count = 0
    if len(dims) > 2:
        band_count = dims[2]
    else:
        band_count = 1
    driver = gdal.GetDriverByName('GTiff')
    image = driver.Create(image_file_name, ysize=dims[0], xsize=dims[1],
                          bands=band_count, eType=gdal.GDT_Float32)
    band_data = None
    for i in range(band_count):
        if 1 == band_count:
            band_data = data
        else:
            band_data = data[:, :, i]
        if normalized:
            data_min = np.min(band_data)
            data_max = np.max(band_data)
            data_distance = data_max - data_min
            band_data = (band_data - data_min) / data_distance
        image.GetRasterBand(i + 1).WriteArray(band_data)
    image = None


def standard_mat_to_tif(mat_file_name,
                        dest_directory=os.path.normpath(os.path.join(os.getcwd(), 'Data'))):
    """
    将标准化的mat格式高光谱文件转化为Geotiff格式的图像文件
    :param mat_file_name: 标准化的mat格式高光谱文件全名
    :param dest_directory: 存放tif文件的目标目录
    :return: 无
    """
    if mat_file_name is None or mat_file_name == '':
        return
    mat = scio.loadmat(mat_file_name)
    if 'WWL' not in mat or mat['WWL'] != 'WWLMat':
        return
    if dest_directory is None or dest_directory == '':
        dest_directory = os.path.normpath(os.path.join(os.getcwd(), 'Data'))
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    _, file_name = os.path.split(mat_file_name)
    file_name, _ = os.path.splitext(file_name)
    image_shape = mat['ImageShape'][0]
    driver = gdal.GetDriverByName('GTiff')
    image_file_name = file_name + '.tif'
    image_file_name = os.path.normpath(os.path.join(dest_directory, image_file_name))
    data = mat['Data']
    band_count = data.shape[1]
    image = driver.Create(image_file_name, ysize=np.int(image_shape[0]), xsize=np.int(image_shape[1]),
                          bands=band_count, eType=gdal.GDT_Float32)
    for i in range(band_count):
        band_data = np.reshape(data[:, i], image_shape)
        image.GetRasterBand(i + 1).WriteArray(band_data)
    image = None
    if 'Labels' in mat:
        label_file_name = file_name + '_gt.tif'
        label_file_name = os.path.normpath(os.path.join(dest_directory, label_file_name))
        labels = mat['Labels'][0]
        band_data = np.reshape(labels, image_shape)
        image = driver.Create(label_file_name, ysize=np.int(image_shape[0]), xsize=np.int(image_shape[1]),
                              bands=1, eType=gdal.GDT_Float32)
        image.GetRasterBand(1).WriteArray(band_data)
        image = None


def show_label_image(image_shape, labels, label_values=None, label_colors=None):
    """
    可视化标签数据(图像分类, 聚类, 分割结果)
    :param image_shape: 图像尺寸([行, 列])
    :param labels: 标签数据, 维度为(行*列, )
    :param label_values: 从小到大排列的标签值, 维度为(类别数目, )
    :param label_colors: 标签颜色, 维度为(类别数目, 3)
    :return: 无
    """
    if image_shape is None or 2 != len(image_shape):
        return
    if labels is None:
        return
    if label_values is None or 0 == len(label_values):
        label_values = []
        for label in labels:
            if label not in label_values:
                label_values.append(label)
        label_values.sort()
        label_values = np.array(label_values, dtype=np.int32)
    if label_colors is None or label_colors.shape[0] != label_values.shape[0]:
        label_colors = np.zeros((label_values.shape[0], 3), dtype=np.int32)
        for i in range(len(label_values)):
            if 0 == label_values[i]:
                label_colors[i, :] = np.array([0, 0, 0], dtype=np.int32)
            else:
                label_colors[i, :] = np.array(np.random.randint(0, 255, [3]), dtype=np.int32)
    indexed_colors = dict(zip(label_values, label_colors))
    image = np.zeros((image_shape[0], image_shape[1], 3), dtype=np.int32)
    for i in range(image_shape[0]):
        for j in range(image_shape[1]):
            image[i, j, :] = indexed_colors[labels[i*image_shape[1]+j]]
    plt.imshow(image)
    plt.axis('off')
    plt.show()


def search_files(directory, file_list, exts=None):
    """
    获取指定目录及子目录下所有指定类型的文件名, 返回文件全名列表
    :param directory: 待搜索的文件目录
    :param file_list: 用于存放搜索结果的的列表, 由于函数是递归搜索, 该列表需要在函数外部预先定义
    :param exts: 文件后缀名列表(None表示所有文件), 单项以'.'开头, 如('.jpg', '.bmp'), ('.pgm',)
    :return: 参数 file_list
    """
    if directory is None or directory == '' or (not isinstance(file_list, list)):
        return
    for name in os.listdir(directory):
        name_temp = os.path.normpath(os.path.join(directory, name))
        if os.path.isdir(name_temp):
            search_files(name_temp, file_list, exts)
        elif os.path.isfile(name_temp):
            if exts is None or exts == '':
                file_list.append(name_temp)
            else:
                if is_typed_file(name_temp, exts):
                    file_list.append(name_temp)


class FileDialog(Tk):
    """
    自定义文件对话框, 可避免出现额外的root窗口
    """
    def __init__(self, style=0, title='打开文件', filetypes=(('所有文件', '*.*'),)):
        """
        构造函数
        :param style: 0表示打开文件对话框, 1表示保存文件对话框, 2表示打开目录对话框
        :param title: 对话框标题
        :param filetypes: 文件过滤器, list或者tuple类型, 如(('所有文件', '*.*'),)
        """
        Tk.__init__(self)
        self.withdraw()
        if 0 == style:
            # 打开单个文件可用askopenfilename或askopenfilenames，打开多个文件只能用askopenfilenames
            self.full_path = askopenfilenames(parent=self, title=title, filetypes=filetypes)
        elif 1 == style:
            self.full_path = asksaveasfilename(parent=self, title=title, filetypes=filetypes)
            self.full_path = os.path.normpath(self.full_path)
        elif 2 == style:
            self.full_path = askdirectory(title=title, mustexist=True)
            self.full_path = os.path.normpath(self.full_path)
        self.destroy()
        self.quit()
        self.mainloop()
            
            
def open_images_gui_aplied(dest_directory=os.path.normpath(os.path.join(os.getcwd(), 'Data')),
                           dest_file_name=None, file_type_exts=('.pgm',), normalized=True):
    """
    读取按目录分类存储的图像文件, 将读取结果统一标准存储为指定的mat文件
    :param dest_directory: 存储mat文件的目标目录
    :param dest_file_name: mat文件名(不需要带.mat后缀名)
    :param file_type_exts: 要读取的图像文件后缀名列表, 单项以'.'开头, 如('.jpg','.bmp','.pgm'), ('.bmp',)
    :param normalized: 是否将读取的数据进行归一化, 默认为True
    :return:
    """
    dlg = FileDialog(style=2, title='选择图像文件主目录')
    image_list = []
    search_files(dlg.full_path, image_list, file_type_exts)
    if not len(image_list):
        print('未找到指定类型的文件')
        return
    n = 0
    data = None
    labels = []
    label_values = []
    label_names = []
    image_shape = None
    label_dict = {}
    for i in range(len(image_list)):
        image_full_name = image_list[i]
        image_path, __ = os.path.split(image_full_name)
        image_path, label_name = os.path.split(image_path)
        __, dest_name = os.path.split(image_path)
        image = PIL.Image.open(image_full_name)
        image = np.array(image, dtype=np.float32)
        dims = image.shape
        num = 1
        for dim in dims:
            num = num * dim
        image = np.reshape(image, [1, num])
        if 0 == i:
            image_shape = np.array([dims[0], dims[1]], dtype=np.int32)
            data = image
            if dest_file_name is None or dest_file_name == '':
                dest_file_name = dest_name
        else:
            data = np.concatenate((data, image), axis=0)
        if label_name not in label_dict:
            label_dict[label_name] = n
            label_values.append(n)
            label_names.append(label_name)
            n = n + 1
        labels.append(label_dict[label_name])
        print(dims, '—>', image.shape, label_dict[label_name], label_name, image_full_name)
    labels = np.array(labels, dtype=np.int32)
    label_values = np.array(label_values, dtype=np.int32)
    label_names = np.array(label_names, dtype=np.str)
    if normalized:
        data_min = np.min(data, axis=len(data.shape)-1).reshape([data.shape[0], 1])
        data_max = np.max(data, axis=len(data.shape)-1).reshape([data.shape[0], 1])
        data_distance = data_max - data_min
        data = (data - data_min) / data_distance
    mat = {'Data': data,
           'Labels': labels,
           'LabelValues': label_values,
           'LabelNames': label_names,
           'ImageShape': image_shape}
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    dest_full_path = os.path.join(dest_directory, dest_file_name + '.mat')
    scio.savemat(dest_full_path, mat)
    print('读取完毕，保存至：')
    print(dest_full_path)


def open_images_gui(dest_directory=os.path.normpath(os.path.join(os.getcwd(), 'Data')),
                    dest_file_name=None, file_type_exts=('.png',), split_symbol='_', normalized=True):
    """
    读取按文件名分类存储的图像文件, 将读取结果统一标准存储为指定的mat文件
    :param dest_directory: 存储mat文件的目标目录
    :param dest_file_name: mat文件名(不需要带.mat后缀名)
    :param file_type_exts: 要读取的图像文件后缀名列表, 单项以'.'开头, 如('.jpg','.bmp','.pgm'), ('.bmp',)
    :param split_symbol: 文件名中类别和序号之间的分隔符号
    :param normalized: 是否将读取的数据进行归一化, 默认为True
    :return: 无
    """
    dlg = FileDialog(style=2, title='选择图像目录')
    image_list = []
    search_files(dlg.full_path, image_list, file_type_exts)
    if not len(image_list):
        print('未找到指定类型的文件')
        return
    n = 0
    data = None
    labels = []
    label_values = []
    label_names = []
    image_shape = None
    label_dict = {}
    for i in range(len(image_list)):
        image_full_name = image_list[i]
        image_path, label_name = os.path.split(image_full_name)
        label_name = label_name.split(split_symbol)[0]
        __, dest_name = os.path.split(image_path)
        image = PIL.Image.open(image_full_name)
        image = np.array(image, dtype=np.float32)
        dims = image.shape
        num = 1
        for dim in dims:
            num = num * dim
        image = np.reshape(image, [1, num])
        if 0 == i:
            image_shape = np.array([dims[0], dims[1]], dtype=np.int32)
            data = image
            if not dest_file_name:
                dest_file_name = dest_name
        else:
            data = np.concatenate((data, image), axis=0)
        if label_name not in label_dict:
            label_dict[label_name] = n
            label_values.append(n)
            label_names.append(label_name)
            n = n + 1
        labels.append(label_dict[label_name])
        print(dims, '—>', image.shape, label_dict[label_name], label_name, image_full_name)
    labels = np.array(labels, dtype=np.int32)
    label_values = np.array(label_values, dtype=np.int32)
    label_names = np.array(label_names, dtype=np.str)
    if normalized:
        data_min = np.min(data, axis=len(data.shape)-1).reshape([data.shape[0], 1])
        data_max = np.max(data, axis=len(data.shape)-1).reshape([data.shape[0], 1])
        data_distance = data_max - data_min
        data = (data - data_min) / data_distance
    mat = {'Data': data,
           'Labels': labels,
           'LabelValues': label_values,
           'LabelNames': label_names,
           'ImageShape': image_shape}
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    dest_full_path = os.path.join(dest_directory, dest_file_name + '.mat')
    scio.savemat(dest_full_path, mat)
    print('读取完毕，保存至：')
    print(dest_full_path)        

