'''
Author: Naiyuan liu
Github: https://github.com/NNNNAI
Date: 2021-11-23 17:03:58
LastEditors: Naiyuan liu
LastEditTime: 2021-11-24 16:45:41
Description: 
'''
from __future__ import division
import collections
import numpy
import PIL
from PIL import Image
import glob
import os
import os.path as osp
import cv2
from insightface.model_zoo import model_zoo
from insightface_func.utils import face_align_ffhqandnewarc as face_align
# from insightface.utils import ensure_available

__all__ = ['Face_detect_crop', 'Face']

Face = collections.namedtuple('Face', [
    'bbox', 'kps', 'det_score', 'embedding', 'gender', 'age',
    'embedding_norm', 'normed_embedding',
    'landmark'
])

Face.__new__.__defaults__ = (None, ) * len(Face._fields)


class Face_detect_crop:
    # def __init__(self, name, root='~/.insightface', allowed_modules=None, **kwargs):
    #     # onnxruntime.set_default_logger_severity(3)
    #     self.models = {}
    #     self.model_dir = ensure_available('models', name, root=root)
    #     onnx_files = glob.glob(osp.join(self.model_dir, '*.onnx'))
    #     onnx_files = sorted(onnx_files)
    #     for onnx_file in onnx_files:
    #         model = model_zoo.get_model(onnx_file, **kwargs)
    #         if model is None:
    #             print('model not recognized:', onnx_file)
    #         elif allowed_modules is not None and model.taskname not in allowed_modules:
    #             print('model ignore:', onnx_file, model.taskname)
    #             del model
    #         elif model.taskname not in self.models and (allowed_modules is None or model.taskname in allowed_modules):
    #             print('find model:', onnx_file, model.taskname, model.input_shape, model.input_mean, model.input_std)
    #             self.models[model.taskname] = model
    #         else:
    #             print('duplicated model task type, ignore:', onnx_file, model.taskname)
    #             del model
    #     assert 'detection' in self.models
    #     self.det_model = self.models['detection']
    def __init__(self, name, root='~/.insightface_func/models'):
        self.models = {}
        root = os.path.expanduser(root)
        onnx_files = glob.glob(osp.join(root, name, '*.onnx'))
        onnx_files = sorted(onnx_files)
        for onnx_file in onnx_files:
            if onnx_file.find('_selfgen_')>0:
                #print('ignore:', onnx_file)
                continue
            model = model_zoo.get_model(onnx_file)
            if model.taskname not in self.models:
                print('find model:', onnx_file, model.taskname)
                self.models[model.taskname] = model
            else:
                print('duplicated model task type, ignore:', onnx_file, model.taskname)
                del model
        assert 'detection' in self.models
        self.det_model = self.models['detection']


    def prepare(self, ctx_id, det_thresh=0.5, det_size=(640, 640), mode ='None', crop_size = 512, ratio = 0.3):
        self.det_thresh = det_thresh
        self.mode = mode
        self.crop_size = crop_size
        self.min_size = int(crop_size*ratio)
        assert det_size is not None
        print('set det-size:', det_size)
        self.det_size = det_size
        for taskname, model in self.models.items():
            if taskname=='detection':
                model.prepare(ctx_id, input_size=det_size)
            else:
                model.prepare(ctx_id)

    def get(self, img, max_num=0):
        bboxes, kpss = self.det_model.detect(img,
                                             threshold=self.det_thresh,
                                             max_num=max_num,
                                             metric='default')
        crop_size = self.crop_size
        if bboxes.shape[0] == 0:
            return None
        align_img_list = []
        M_list = []

        for i in range(bboxes.shape[0]):
            kps = None
            if kpss is not None:
                kps = kpss[i]
            width = bboxes[i][2] - bboxes[i][0]
            height = bboxes[i][3] - bboxes[i][1]
            if max(width,height) < self.min_size:
                # print("The detected face (%d,%d) is smaller than the minimum value %d"%(width,height,self.min_size))
                continue
            # print("width:%d, height:%d"%(width, height))
            M, _        = face_align.estimate_norm(kps, crop_size, mode = self.mode) 
            align_img   = cv2.warpAffine(img, M, (crop_size, crop_size), flags=cv2.INTER_LANCZOS4, borderValue=0.0)
            align_img   = Image.fromarray(cv2.cvtColor(align_img,cv2.COLOR_BGR2RGB))
            # align_img   = align_img.resize((crop_size, crop_size), PIL.Image.LANCZOS)
            # # align_img   = cv2.cvtColor(numpy.asarray(align_img),cv2.COLOR_RGB2BGR)

            align_img_list.append(align_img)
            M_list.append(M)
        if len(align_img_list)<1:
            return None
        
        return align_img_list, M_list
