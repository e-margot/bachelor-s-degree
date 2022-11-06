import itertools
import json
import os
import sys
import time
import numpy as np
from collections import defaultdict

PYTHON_VERSION = sys.version_info[0]
if PYTHON_VERSION == 2:
    from urllib import urlretrieve
elif PYTHON_VERSION == 3:
    from urllib.request import urlretrieve


def _is_array_like(obj):
    return hasattr(obj, '__iter__') and hasattr(obj, '__len__')


"""# Overriding COCO and CocoDetection"""


class MyCOCO:
    def __init__(self, annotation_file=None):
        """
        Constructor of Microsoft COCO helper class for reading and visualizing annotations.
        :param annotation_file (str): location of annotation file
        :param image_folder (str): location to the folder that hosts images.
        :return:
        """
        # load dataset
        self.dataset, self.anns, self.cats, self.imgs = dict(), dict(), dict(), dict()
        self.img_to_anns, self.cat_to_imgs = defaultdict(list), defaultdict(list)
        if annotation_file:
            print('loading annotations into memory...')
            tic = time.time()
            dataset = json.load(open(annotation_file, 'r'))
            assert type(dataset) == dict, 'annotation file format {} not supported'.format(type(dataset))
            print('Done (t={:0.2f}s)'.format(time.time() - tic))
            self.dataset = dataset
            self.create_index()

    def create_index(self):
        # create index
        print('creating index...')
        anns, cats, imgs = {}, {}, {}
        img_to_anns, cat_to_imgs = defaultdict(list), defaultdict(list)
        if 'annotations' in self.dataset:
            for ann in self.dataset['annotations']:
                img_to_anns[ann['id']].append(ann)
                anns[ann['id']] = ann
                imgs[ann['id']] = ann

        if 'categories' in self.dataset:
            for cat in self.dataset['categories']:
                cats[cat['id']] = cat

        if 'annotations' in self.dataset and 'categories' in self.dataset:
            for ann in self.dataset['annotations']:
                cat_to_imgs[ann['category_id']].append(ann['id'])

        print('index created!')

        # create class members
        self.anns = anns
        self.img_to_anns = img_to_anns
        self.cat_to_imgs = cat_to_imgs
        self.imgs = imgs
        self.cats = cats

    def get_ann_ids(self, img_ids=[], cat_ids=[], iscrowd=None):
        """
        Get ann ids that satisfy given filter conditions. default skips that filter
        :param img_ids  (int array)     : get anns for given imgs
               cat_ids  (int array)     : get anns for given cats
               iscrowd (boolean)       : get anns for given crowd label (False or True)
        :return: ids (int array)       : integer array of ann ids
        """
        img_ids = img_ids if _is_array_like(img_ids) else [img_ids]
        cat_ids = cat_ids if _is_array_like(cat_ids) else [cat_ids]

        if len(img_ids) == len(cat_ids) == 0:
            anns = self.dataset['annotations']
        else:
            if not len(img_ids) == 0:
                lists = [self.img_to_anns[imgId] for imgId in img_ids if imgId in self.img_to_anns]
                anns = list(itertools.chain.from_iterable(lists))
            else:
                anns = self.dataset['annotations']
            anns = anns if len(cat_ids) == 0 else [ann for ann in anns if ann['category_id'] in cat_ids]
        if not iscrowd == None:
            ids = [ann['id'] for ann in anns if ann['iscrowd'] == iscrowd]
        else:
            ids = [ann['id'] for ann in anns]
        return ids

    def get_cat_ids(self, cat_nms=[], cat_ids=[]):
        """
        filtering parameters. default skips that filter.
        :param cat_nms (str array)  : get cats for given cat names
        :param cat_ids (int array)  : get cats for given cat ids
        :return: ids (int array)   : integer array of cat ids
        """
        cat_nms = cat_nms if _is_array_like(cat_nms) else [cat_nms]
        cat_ids = cat_ids if _is_array_like(cat_ids) else [cat_ids]

        if len(cat_nms) == len(cat_ids) == 0:
            cats = self.dataset['categories']
        else:
            cats = self.dataset['categories']
            cats = cats if len(cat_nms) == 0 else [cat for cat in cats if cat['name'] in cat_nms]
            cats = cats if len(cat_ids) == 0 else [cat for cat in cats if cat['id'] in cat_ids]
        ids = [cat['id'] for cat in cats]
        return ids

    def get_img_ids(self, img_ids=[], cat_ids=[]):
        """
        Get img ids that satisfy given filter conditions.
        :param img_ids (int array) : get imgs for given ids
        :param cat_ids (int array) : get imgs with all given cats
        :return: ids (int array)  : integer array of img ids
        """
        img_ids = img_ids if _is_array_like(img_ids) else [img_ids]
        cat_ids = cat_ids if _is_array_like(cat_ids) else [cat_ids]

        if len(img_ids) == len(cat_ids) == 0:
            ids = self.imgs.keys()
        else:
            ids = set(img_ids)
            for i, cat_id in enumerate(cat_ids):
                if i == 0 and len(ids) == 0:
                    ids = set(self.cat_to_imgs[cat_id])
                else:
                    ids &= set(self.cat_to_imgs[cat_id])
        return list(ids)

    def cls(self, ids=[]):
        target = [0] * 80
        a = ([self.anns[id]['category_id'] for id in ids])
        target[a[0] - 1] = 1
        return target

    def load_anns(self, ids=[]):
        """
        Load anns with the specified ids.
        :param ids (int array)       : integer ids specifying anns
        :return: anns (object array) : loaded ann objects
        """
        if _is_array_like(ids):
            return self.cls(ids)
        elif type(ids) == int:
            return [self.anns[ids]]

    def load_cats(self, ids=[]):
        """
        Load cats with the specified ids.
        :param ids (int array)       : integer ids specifying cats
        :return: cats (object array) : loaded cat objects
        """
        if _is_array_like(ids):
            return [self.cats[id] for id in ids]
        elif type(ids) == int:
            return [self.cats[ids]]

    def load_imgs(self, ids=[]):
        """
        Load anns with the specified ids.
        :param ids (int array)       : integer ids specifying img
        :return: imgs (object array) : loaded img objects
        """
        if _is_array_like(ids):
            return [self.imgs[id] for id in ids]
        elif type(ids) == int:
            return [self.imgs[ids]]

    def download(self, tar_dir=None, img_ids=[]):
        """
        Download COCO images from mscoco.org server.
        :param tar_dir (str): COCO results directory name
               img_ids (list): images to be downloaded
        :return:
        """
        if tar_dir is None:
            print('Please specify target directory')
            return -1
        if len(img_ids) == 0:
            imgs = self.imgs.values()
        else:
            imgs = self.load_imgs(img_ids)
        if not os.path.exists(tar_dir):
            os.makedirs(tar_dir)
        for i, img in enumerate(imgs):
            tic = time.time()
            file_name = os.path.join(tar_dir, img['file_name'])
            if not os.path.exists(file_name):
                urlretrieve(img['coco_url'], file_name)
            print('downloaded {}/{} images (t={:0.1f}s)'.format(i, len(imgs), time.time() - tic))

    def load_numpy_annotations(self, data):
        """
        Convert result data from a numpy array [Nx7] where each row contains {imageID,x1,y1,w,h,score,class}
        :param  data (numpy.ndarray)
        :return: annotations (python nested list)
        """
        print('Converting ndarray to lists...')
        assert (type(data) == np.ndarray)
        print(data.shape)
        assert (data.shape[1] == 7)
        N = data.shape[0]
        ann = []
        for i in range(N):
            if i % 1000000 == 0:
                print('{}/{}'.format(i, N))
            ann += [{
                'image_id': int(data[i, 0]),
                'bbox': [data[i, 1], data[i, 2], data[i, 3], data[i, 4]],
                'score': data[i, 5],
                'category_id': int(data[i, 6]),
            }]
        return ann
