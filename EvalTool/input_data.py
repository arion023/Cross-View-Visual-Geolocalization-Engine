import cv2
import random
import numpy as np


class InputDataEval:

    # crop is tuple (top_crop, bottom_crop), if there is no crop, there should be (None, None)
    def __init__(self, root_dir, crop = (None, None), splits="splits/val-19zl.csv"):

        if crop[0] == 0 or crop[1] == 0:
            raise ValueError("If there is no crop, it should be (None, None)")

        self.crop = crop
        self.img_root = root_dir
        self.splits = splits
        self.dataset_list= self.img_root + self.splits

        print('InputData::__init__: load %s' % self.dataset_list)
        self.__cur_test_id = 0  # for training
        self.id_dataset_list = []
        self.id_dataset_idx_list = []
        self.id_matches_dict = {}
        with open(self.dataset_list, 'r') as file:
            idx = 0
            for line in file:
                data = line.split(',')
                pano_id = (data[1].split('/')[-1]).split('.')[0]
                # satellite filename, streetview filename, pano_id
                self.id_dataset_list.append([data[0], data[1], pano_id])


                self.id_dataset_idx_list.append(idx)
                idx += 1
        self.dataset_size = len(self.id_dataset_list)
        self.make_unique_sat_idx_list()
        print('InputData::__init__: load', self.dataset_list, ' data_size =', self.dataset_size)




    def next_batch_scan(self, batch_size):
        if self.__cur_test_id >= self.dataset_size:
            self.__cur_test_id = 0
            return None, None
        elif self.__cur_test_id + batch_size >= self.dataset_size:
            batch_size = self.dataset_size - self.__cur_test_id

        batch_grd = np.zeros([batch_size, 224, 1232, 3], dtype = np.float32)
        batch_sat = np.zeros([batch_size, 512, 512, 3], dtype=np.float32)
        for i in range(batch_size):
            img_idx = self.__cur_test_id + i

            # satellite
            img = cv2.imread(self.img_root + self.id_dataset_list[img_idx][0])
            img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA)

            img = img.astype(np.float32)
            # img -= 100.0
            img[:, :, 0] -= 103.939  # Blue
            img[:, :, 1] -= 116.779  # Green
            img[:, :, 2] -= 123.6  # Red
            batch_sat[i, :, :, :] = img

            # ground
            img = cv2.imread(self.img_root + self.id_dataset_list[img_idx][1])

            img = img[ self.crop[0]:-self.crop[1], :, :]
            img = cv2.resize(img, (1232, 224), interpolation=cv2.INTER_AREA)

            img = img.astype(np.float32)
            # img -= 100.0
            img[:, :, 0] -= 103.939  # Blue
            img[:, :, 1] -= 116.779  # Green
            img[:, :, 2] -= 123.6  # Red
            batch_grd[i, :, :, :] = img

        self.__cur_test_id += batch_size

        return batch_sat, batch_grd

    def get_crop(self):
        # crop is tuple (bottom_crop, top_crop)
        return self.crop

    def get_sat(self, idx):
        # 0 image is sat img path
        return self.id_dataset_list[idx][0]

    def get_pano(self, idx):
        # 1 image, is panorama img path
        return self.id_dataset_list[idx][1]

    def get_unique_sat_idx_list(self):
        # returns ids of unique satelite paths
        return self.unique_sat_idx_list

    def make_unique_sat_idx_list(self):
        unique_indices = {}
        for idx, elem in enumerate(self.id_dataset_list):
            if elem[0] not in unique_indices:
                unique_indices[elem[0]] = idx

        self.unique_sat_idx_list = list(unique_indices.values())


    def get_dataset_list(self):
        return self.id_dataset_list

    def get_dataset_size(self):
        return self.dataset_size

    def reset_scan(self):
        self.__cur_test_id = 0

