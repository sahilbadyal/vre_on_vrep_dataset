import os
import json
import _pickle as cPickle
from PIL import Image
import base64
import numpy as np
import time
import logging

import torch
from torch.utils.data import Dataset
from external.pytorch_pretrained_bert import BertTokenizer

from common.utils.zipreader import ZipReader
from common.utils.create_logger import makedirsExist
from common.utils.bbox import bbox_iou_py_vectorized

class Refer():
    def __init__(self):
        self.ref_id_to_box = {}
    
    def getRefBox(self,ref_id):
        return self.ref_id_to_box[ref_id][1] 
    

class VRep(Dataset):
    def __init__(self, image_set, root_path, data_path, boxes='gt', proposal_source='official',
                 transform=None, test_mode=False,
                 zip_mode=False, cache_mode=False, cache_db=False, ignore_db_cache=True,
                 tokenizer=None, pretrained_model_name=None,
                 add_image_as_a_box=False, mask_size=(14, 14),
                 aspect_grouping=False, **kwargs):
        """
        VREP Dataset

        :param image_set: image folder name
        :param root_path: root path to cache database loaded from annotation file
        :param data_path: path to dataset
        :param boxes: boxes to use, 'gt' or 'proposal'
        :param transform: transform
        :param test_mode: test mode means no labels available
        :param zip_mode: reading images and metadata in zip archive
        :param cache_mode: cache whole dataset to RAM first, then __getitem__ read them from RAM
        :param ignore_db_cache: ignore previous cached database, reload it from annotation file
        :param tokenizer: default is BertTokenizer from pytorch_pretrained_bert
        :param add_image_as_a_box: add whole image as a box
        :param mask_size: size of instance mask of each object
        :param aspect_grouping: whether to group images via their aspect
        :param kwargs:
        """
        super(VRep, self).__init__()

        assert not cache_mode, 'currently not support cache mode!'
        self.data_json = 'obj_det_res.json'#'image_seg_test.json'#'obj_det_res.json'
        self.ref_json = 'ref_annotations.json'
        self.boxes = boxes
        self.refer = Refer()
        self.test_mode = test_mode
        self.data_path = data_path
        self.root_path = root_path
        self.transform = transform
        self.zip_mode = zip_mode
        self.cache_mode = cache_mode
        self.cache_db = cache_db
        self.ignore_db_cache = ignore_db_cache
        self.aspect_grouping = aspect_grouping
        self.cache_dir = os.path.join(root_path, 'cache')
        self.add_image_as_a_box = add_image_as_a_box
        self.mask_size = mask_size
        if not os.path.exists(self.cache_dir):
            makedirsExist(self.cache_dir)
        self.tokenizer = tokenizer if tokenizer is not None \
            else BertTokenizer.from_pretrained(
            'bert-base-uncased' if pretrained_model_name is None else pretrained_model_name,
            cache_dir=self.cache_dir)

        if zip_mode:
            self.zipreader = ZipReader()

        self.database = self.load_annotations()
        if self.aspect_grouping:
            self.group_ids = self.group_aspect(self.database)

    @property
    def data_names(self):
        if self.test_mode:
            return ['image', 'boxes', 'im_info', 'expression']
        else:
            return ['image', 'boxes', 'im_info', 'expression', 'label']

    def __getitem__(self, index):
        idb = self.database[index]

	#print(idb)

        # image related
        img_id = idb['image_id']
        image = self._load_image(idb['image_fn'])
        im_info = torch.as_tensor([idb['width'], idb['height'], 1.0, 1.0])
        if not self.test_mode:
            gt_box = torch.as_tensor(idb['gt_box'])
        flipped = False
        bb = self._load_json(os.path.join(self.data_path, 'bb.json'))
        if self.boxes == 'gt':
            boxes = torch.as_tensor(bb[img_id])

        if self.add_image_as_a_box:
            w0, h0 = im_info[0], im_info[1]
            image_box = torch.as_tensor([[0.0, 0.0, w0 - 1, h0 - 1]])
            boxes = torch.cat((image_box, boxes), dim=0)

        if self.transform is not None:
            if not self.test_mode:
                boxes = torch.cat((gt_box[None], boxes), 0)
            image, boxes, _, im_info, flipped = self.transform(image, boxes, None, im_info, flipped)
            if not self.test_mode:
                gt_box = boxes[0]
                boxes = boxes[1:]

        # clamp boxes
        w = im_info[0].item()
        h = im_info[1].item()
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(min=0, max=w - 1)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(min=0, max=h - 1)
        if not self.test_mode:
            gt_box[[0, 2]] = gt_box[[0, 2]].clamp(min=0, max=w - 1)
            gt_box[[1, 3]] = gt_box[[1, 3]].clamp(min=0, max=h - 1)

        # assign label to each box by its IoU with gt_box
        if not self.test_mode:
            boxes_ious = bbox_iou_py_vectorized(boxes, gt_box[None]).view(-1)
            label = (boxes_ious > 0.5).float()

        # expression
        exp_tokens = idb['tokens']
        exp_retokens = self.tokenizer.tokenize(' '.join(exp_tokens))
        if flipped:
            exp_retokens = self.flip_tokens(exp_retokens, verbose=True)
        exp_ids = self.tokenizer.convert_tokens_to_ids(exp_retokens)

        if self.test_mode:
            return image, boxes, im_info, exp_ids
        else:
            return image, boxes, im_info, exp_ids, label

    @staticmethod
    def flip_tokens(tokens, verbose=True):
        changed = False
        tokens_new = [tok for tok in tokens]
        for i, tok in enumerate(tokens):
            if tok == 'left':
                tokens_new[i] = 'right'
                changed = True
            elif tok == 'right':
                tokens_new[i] = 'left'
                changed = True
        if verbose and changed:
            logging.info('[Tokens Flip] {} -> {}'.format(tokens, tokens_new))
        return tokens_new

    @staticmethod
    def b64_decode(string):
        return base64.decodebytes(string.encode())

    def load_annotations(self):
        tic = time.time()
        database = []
        db_cache_name = 'vrep_boxes'#_{}_{}'.format(self.boxes, '+'.join(self.image_sets))
        if self.zip_mode:
            db_cache_name = db_cache_name + '_zipmode'
        if self.test_mode:
            db_cache_name = db_cache_name + '_testmode'
        db_cache_root = os.path.join(self.root_path, 'cache')
        db_cache_path = os.path.join(db_cache_root, '{}.pkl'.format(db_cache_name))
        dataset = self._load_json(os.path.join(self.data_path, self.data_json))
        ref = self._load_json(os.path.join(self.data_path, self.ref_json))
        if os.path.exists(db_cache_path):
            if not self.ignore_db_cache:
                # reading cached database
                print('cached database found in {}.'.format(db_cache_path))
                with open(db_cache_path, 'rb') as f:
                    print('loading cached database from {}...'.format(db_cache_path))
                    tic = time.time()
                    database = cPickle.load(f)
                    print('Done (t={:.2f}s)'.format(time.time() - tic))
                    return database
            else:
                print('cached database ignored.')

        # ignore or not find cached database, reload it from annotation file
        #print('loading database of split {}...'.format('+'.join(self.image_sets)))
        tic = time.time()

        refer_id = 0 
	
        for data_point in dataset['images']:
            iset  = 'full_images'
            image_name = data_point['file_name'].split('/')[3]
            if True:
            	for anno in data_point['annotations']:
                    if anno['id'] == data_point['ground_truth']:
                        gt_x, gt_y, gt_w, gt_h = anno['bbox']
            if self.zip_mode:
                image_fn = os.path.join(self.data_path, iset + '.zip@/' + iset, image_name)
            else:
                image_fn = os.path.join(self.data_path, iset, image_name)
            for sent in ref[image_name]:
                idb = {
                    #'sent_id': sent['sent_id'],
                    #'ann_id': ref['ann_id'],
                    'ref_id': refer_id,
                    'image_id': image_name,
                    'image_fn': image_fn,
                    'width': 1024,
                    'height': 576,
                    'raw': sent,
                    'sent': sent,
                    'tokens': self.tokenizer.tokenize(sent),
                    #'category_id': ref['category_id'],
                    'gt_box': [gt_x, gt_y, gt_x + gt_w, gt_y + gt_h] if not self.test_mode else None
                }
                self.refer.ref_id_to_box[refer_id] = [image_name, [gt_x, gt_y, gt_w, gt_h], sent]
                database.append(idb)
                refer_id += 1

        with open('./final_refer_testset', 'w') as f:
            json.dump(self.refer.ref_id_to_box, f)

        print('Done (t={:.2f}s)'.format(time.time() - tic))

        # cache database via cPickle
        if self.cache_db:
            print('caching database to {}...'.format(db_cache_path))
            tic = time.time()
            if not os.path.exists(db_cache_root):
                makedirsExist(db_cache_root)
            with open(db_cache_path, 'wb') as f:
                cPickle.dump(database, f)
            print('Done (t={:.2f}s)'.format(time.time() - tic))

        return database

    @staticmethod
    def group_aspect(database):
        print('grouping aspect...')
        t = time.time()

        # get shape of all images
        widths = torch.as_tensor([idb['width'] for idb in database])
        heights = torch.as_tensor([idb['height'] for idb in database])

        # group
        group_ids = torch.zeros(len(database))
        horz = widths >= heights
        vert = 1 - horz
        group_ids[horz] = 0
        group_ids[vert] = 1

        print('Done (t={:.2f}s)'.format(time.time() - t))

        return group_ids

    def __len__(self):
        return len(self.database)

    def _load_image(self, path):
        if '.zip@' in path:
            return self.zipreader.imread(path).convert('RGB')
        else:
            return Image.open(path).convert('RGB')

    def _load_json(self, path):
        if '.zip@' in path:
            f = self.zipreader.read(path)
            return json.loads(f.decode())
        else:
            with open(path, 'r') as f:
                return json.load(f)

