# -*- coding: utf-8 -*-

import os
import utility as ut


root_directory = 'F:\\Data\\Image_Hyperspectral\\'
file_directories = ['Botswana',
                    'Cuprite',
                    'Indian Pines',
                    'Kennedy Space Center',
                    'Pavia',
                    'Pavia University',
                    'Salinas',
                    'SalinasA']
# file_directories = ['Cuprite']
dest_directory = 'D:\\Projects\\Python\\Data'


def generate_mat_file(root_directory=root_directory, directory_list=file_directories,
                      dest_directory=os.path.normpath(os.path.join(os.getcwd(), 'Data')), normalized=True):
    """
    从非标准mat文件批量生成标准mat文件
    :param root_directory: 存放非标准mat文件的主目录
    :param directory_list: 存放非标准mat文件的目录列表, 每个非标准数据mat及其标签mat存放在同一个目录
    :param dest_directory: 存放生成的标准mat的目标目录
    :param normalized: 指示生成过程中是否对数据进行归一化, 默认为True
    :return: 无
    """
    if root_directory is None or root_directory == '':
        return
    if directory_list is None or 0 == len(directory_list):
        return
    if dest_directory is None or dest_directory == '':
        dest_directory = os.path.normpath(os.path.join(os.getcwd(), 'Data'))
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    for directory_name in directory_list:
        search_directory = os.path.normpath(os.path.join(root_directory, directory_name))
        file_names = []
        ut.search_files(search_directory, file_names, ('.mat',))
        if len(file_names) > 0:
            data_file_names = []
            label_file_name = None
            for file_name in file_names:
                if '_gt.mat' in file_name:
                    label_file_name = file_name
                else:
                    data_file_names.append(file_name)
            if len(data_file_names) > 0:
                for data_file_name in data_file_names:
                    ut.read_hyperspectral_file(data_file_name, label_file_name, dest_directory, normalized)
                    _, name = os.path.split(data_file_name)
                    print(os.path.normpath(os.path.join(dest_directory, name)))


def generate_tif_file(root_directory=root_directory, directory_list=file_directories,
                      dest_directory=os.path.normpath(os.path.join(os.getcwd(), 'Data')), normalized=False):
    """
    从非标准mat文件批量生成tif文件
    :param root_directory: 存放非标准mat文件的主目录
    :param directory_list: 存放非标准mat文件的目录列表, 每个非标准数据mat及其标签mat存放在同一个目录
    :param dest_directory: 存放生成的tif文件的目标目录
    :param normalized: 指示生成过程中是否对数据进行归一化, 默认为False
    :return: 无
    """
    if root_directory is None or root_directory == '':
        return
    if directory_list is None or 0 == len(directory_list):
        return
    if dest_directory is None or dest_directory == '':
        dest_directory = os.path.normpath(os.path.join(os.getcwd(), 'Data'))
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    for directory_name in directory_list:
        search_directory = os.path.normpath(os.path.join(root_directory, directory_name))
        file_names = []
        ut.search_files(search_directory, file_names, ('.mat',))
        if len(file_names) > 0:
            for file_name in file_names:
                ut.mat_to_tif(file_name, dest_directory, normalized)
                _, name = os.path.split(file_name)
                name, _ = os.path.splitext(name)
                print(os.path.normpath(os.path.join(dest_directory, name + '.tif')))


if __name__ == '__main__':
    # generate_mat_file(root_directory, file_directories, dest_directory, normalized=True)
    # generate_tif_file(root_directory, file_directories, dest_directory, normalized=False)
    print('\nNothing has been done!\n'
          'If you want to do something,\n'
          'please comment out these lines and correct corresponding codes in the source file.')
