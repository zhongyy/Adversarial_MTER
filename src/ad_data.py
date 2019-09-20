from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random
import logging
import sys
import numbers
import math
import sklearn
import datetime
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib
import mxnet as mx
from mxnet import ndarray as nd
from mxnet import io
from mxnet import recordio
sys.path.append(os.path.join(os.path.dirname(__file__), 'common'))
import face_preprocess
import multiprocessing
sys.path.append(os.path.join(os.path.dirname(__file__), 'symbols'))
import fresnet_adv
logger = logging.getLogger()


class FaceImageIter(io.DataIter):

    def __init__(self, batch_size_src, data_shape,
                 path_imgrec = None, model = None,
                 shuffle=False, aug_list=None, mean = None,
                 rand_mirror = False, cutoff = 0,
                 data_name='data', label_name='softmax_label',
                 emb_size = None, num_layers = None, version_se = None,
                 version_input = None, version_output = None,
                 version_unit = None, version_act = None,
                 ctx= None, ctxnum= None, main_ctx_num = None, adv_round= None, adv_thd = None, adv_sigma= None, **kwargs):

        super(FaceImageIter, self).__init__()
        assert path_imgrec
        if path_imgrec:
            logging.info('loading recordio %s...',
                         path_imgrec)
            path_imgidx = path_imgrec[0:-4]+".idx"
            self.imgrec = recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, 'r')  # pylint: disable=redefined-variable-type
            s = self.imgrec.read_idx(0)
            header, _ = recordio.unpack(s)
            if header.flag>0:
              print('header0 label', header.label)
              self.header0 = (int(header.label[0]), int(header.label[1]))
              self.imgidx = range(1, int(header.label[0]))
              self.id2range = {}
              self.seq_identity = range(int(header.label[0]), int(header.label[1]))
              for identity in self.seq_identity:
                s = self.imgrec.read_idx(identity)
                header, _ = recordio.unpack(s)
                a,b = int(header.label[0]), int(header.label[1])
                self.id2range[identity] = (a,b)
              print('id2range', len(self.id2range))
            else:
              self.imgidx = list(self.imgrec.keys)
            self.seq = None
        self.thd = adv_thd
        self.sigma = adv_sigma
        self.round = adv_round
        self.emb_size = emb_size
        self.num_layers = num_layers
        self.version_se = version_se
        self.version_input = version_input
        self.version_output = version_output
        self.version_unit = version_unit
        self.version_act = version_act
        self.mean = mean
        self.nd_mean = None
        if self.mean:
          self.mean = np.array(self.mean, dtype=np.float32).reshape(1,1,3)
          self.nd_mean = mx.nd.array(self.mean).reshape((1,1,3))
        self.check_data_shape(data_shape)
        self.provide_data = [(data_name, (3 * batch_size_src,) + data_shape)]
        self.batch_size = batch_size_src*3
        self.batch_size_src = batch_size_src
        self.data_shape = data_shape
        self.shuffle = shuffle
        self.image_size = '%d,%d'%(data_shape[1],data_shape[2])
        self.rand_mirror = rand_mirror
        print('rand_mirror', rand_mirror)
        self.cutoff = cutoff
        self.provide_label = [(label_name, (batch_size_src*3,))]
        self.provide_label_srctar = [(label_name, (batch_size_src*2,))]
        self.provide_label_adv = [(label_name, (batch_size_src,))]
        self.cur1 = 0
        self.cur2 = 0
        self.nbatch = 0
        self.is_init = False
        self.oid = 1
        self.oseq1 = None
        self.oseq2 = None
        self.ctx = ctx
        self.ctx_num = ctxnum
        self.main_ctx_num = main_ctx_num
        self.model = model
        self.model_adv = None
        self.model_adv_init = False
        self.allow_missing = False
        self.force_rebind = False
        self.force_init = False
        self.inputs_need_grad = True
        self.for_training = True

    def reset(self):
      print('oseq_reset')
      self.cur1 = 0
      self.cur2 = 0
      ids = []
      for k in self.id2range:
        ids.append(k)
      random.shuffle(ids)
      self.oseq1 = []
      self.oseq2 = []
      for _id in ids:
        v = self.id2range[_id]
        _list = range(*v)
        random.shuffle(_list)
        if self.oid==1:
            self.oseq1 += _list
            self.oid = 2
        else:
            self.oseq2 += _list
            self.oid = 1
      random.shuffle(self.oseq1)
      random.shuffle(self.oseq2)
      print("oseq1: ", len(self.oseq1))
      print("oseq2: ", len(self.oseq2))
      if self.model_adv_init == False:
          arg_t, aux_t = self.model.get_params()
          sym, arg_params, aux_params = self.get_symbol(arg_t, aux_t)
          self.model_adv = mx.mod.Module(context=self.ctx, symbol=sym)
          provide_data = [('data', (2 * self.batch_size_src, 3, self.data_shape[1], self.data_shape[2]))]
          provide_label = [('softmax_label', (2 * self.batch_size_src,))]
          self.model_adv.bind(data_shapes=provide_data, label_shapes=provide_label,
                              for_training=self.for_training, inputs_need_grad=self.inputs_need_grad, force_rebind=self.force_rebind)
          self.model_adv.init_params(arg_params=arg_params, aux_params=aux_params,
                                     allow_missing=self.allow_missing, force_init=self.force_init)
          self.model_adv_init == True
          print("init model_adv params")
      else:
          arg_t, aux_t = self.model.get_params()
          self.model_adv.set_params(arg_t, aux_t)
          print("update model_adv params")

    def next_sample1(self):
        if self.oseq1 is not None:
          while True:
            if self.cur1 >= len(self.oseq1):
                raise StopIteration
            idx1 = self.oseq1[self.cur1]
            self.cur1 += 1
            if self.imgrec is not None:
              s = self.imgrec.read_idx(idx1)
              header, img = recordio.unpack(s)
              label = header.label
              if not isinstance(label, numbers.Number):
                label = label[0]
              return label, img, None, None

    def next_sample2(self):
        if self.oseq2 is not None:
          while True:
            if self.cur2 >= len(self.oseq2):
                raise StopIteration
            idx2 = self.oseq2[self.cur2]
            self.cur2 += 1
            if self.imgrec is not None:
              s = self.imgrec.read_idx(idx2)
              header, img = recordio.unpack(s)
              label = header.label
              if not isinstance(label, numbers.Number):
                label = label[0]
              return label, img, None, None

    def next(self):
        if not self.is_init:
          self.reset()
          self.is_init = True
        """Returns the next batch of data."""
        self.nbatch +=1
        c, h, w = self.data_shape
        batch_data_srctar = nd.empty((2 * self.batch_size_src, c, h, w))
        batch_data_t = nd.empty((2 * self.batch_size_src, c, h, w))
        batch_label_srctar = nd.empty(self.provide_label_srctar[0][1])
        batch_label_t = nd.empty(self.provide_label_srctar[0][1])

        batch_data_tar = nd.empty((self.batch_size_src, c, h, w))
        batch_data_adv = nd.empty((self.batch_size_src, c, h, w))
        batch_data = nd.empty((3 * self.batch_size_src, c, h, w))
        batch_label = nd.empty(self.provide_label[0][1])

        #time_now1 = datetime.datetime.now()
        arg_t, aux_t = self.model.get_params()
        self.model_adv.set_params(arg_t, aux_t)
        #print("update model_adv params")
        #time_now2 = datetime.datetime.now()
        #print("update params time", time_now2-time_now1)

        i = 0
        try:
            while i < self.batch_size_src:
                label, s, bbox, landmark = self.next_sample1()
                _data = self.imdecode(s)
                if self.rand_mirror:
                  _rd = random.randint(0,1)
                  if _rd==1:
                    _data = mx.ndarray.flip(data=_data, axis=1)
                if self.nd_mean is not None:
                    _data = _data.astype('float32')
                    _data -= self.nd_mean
                    _data *= 0.0078125
                if self.cutoff>0:
                  centerh = random.randint(0, _data.shape[0]-1)
                  centerw = random.randint(0, _data.shape[1]-1)
                  half = self.cutoff//2
                  starth = max(0, centerh-half)
                  endh = min(_data.shape[0], centerh+half)
                  startw = max(0, centerw-half)
                  endw = min(_data.shape[1], centerw+half)
                  _data = _data.astype('float32')
                  _data[starth:endh, startw:endw, :] = 127.5
                data = [_data]
                try:
                    self.check_valid_image(data)
                except RuntimeError as e:
                    logging.debug('Invalid image, skipping:  %s', str(e))
                    continue

                for datum in data:
                    assert i < self.batch_size_src, 'Batch size must be multiples of augmenter output length'
                    batch_data_srctar[i][:] = self.postprocess_data(datum)
                    batch_label_srctar[i][:] = label
                    i += 1
        except StopIteration:
            if i< self.batch_size_src:
                raise StopIteration
        try:
            while i < 2 * self.batch_size_src:
                label, s, bbox, landmark = self.next_sample2()
                _data = self.imdecode(s)
                if self.rand_mirror:
                  _rd = random.randint(0,1)
                  if _rd==1:
                    _data = mx.ndarray.flip(data=_data, axis=1)
                if self.nd_mean is not None:
                    _data = _data.astype('float32')
                    _data -= self.nd_mean
                    _data *= 0.0078125
                if self.cutoff>0:
                  centerh = random.randint(0, _data.shape[0]-1)
                  centerw = random.randint(0, _data.shape[1]-1)
                  half = self.cutoff//2
                  starth = max(0, centerh-half)
                  endh = min(_data.shape[0], centerh+half)
                  startw = max(0, centerw-half)
                  endw = min(_data.shape[1], centerw+half)
                  _data = _data.astype('float32')
                  _data[starth:endh, startw:endw, :] = 127.5
                data = [_data]
                try:
                    self.check_valid_image(data)
                except RuntimeError as e:
                    logging.debug('Invalid image, skipping:  %s', str(e))
                    continue

                for datum in data:
                    assert i < 2 * self.batch_size_src, 'Batch size must be multiples of augmenter output length'
                    batch_data_srctar[i][:] = self.postprocess_data(datum)
                    batch_label_srctar[i][:] = label
                    i += 1
        except StopIteration:
            if i< 2 * self.batch_size_src:
                raise StopIteration

        #print("batch_label_srctar:", batch_label_srctar)
        margin = self.batch_size_src//self.ctx_num
        #print("margin: ",margin)
        for i in xrange(self.ctx_num):
            batch_data_t[2 * i * margin:(2 * i + 1) * margin][:] = batch_data_srctar[i * margin:(i + 1) * margin][:]
            batch_data_t[(2 * i + 1) * margin:2 * (i + 1) * margin][:] = batch_data_srctar[self.batch_size_src + i * margin:self.batch_size_src + (i + 1) * margin][:]
        for i in xrange(self.ctx_num):
            batch_label_t[2 * i * margin:(2 * i + 1) * margin][:] = batch_label_srctar[i * margin:(i + 1) * margin][:]
            batch_label_t[(2 * i + 1) * margin:2 * (i + 1) * margin][:] = batch_label_srctar[self.batch_size_src + i * margin:self.batch_size_src + (i + 1) * margin][:]

        #print("batch_label_t:", batch_label_t)

        db = mx.io.DataBatch([batch_data_t])
        self.model_adv.forward(db, is_train=True)
        ori_out = self.model_adv.get_outputs()[-1].asnumpy()
        #print("ori_dis: ", ori_out)
        self.model_adv.backward()
        grad = self.model_adv.get_input_grads()[0]
        #print("grad: ", grad)
        grad = mx.nd.array(grad)
        #print("batch_data_t: ", batch_data_t.asnumpy().shape)

        for i in xrange(self.ctx_num):
            batch_data_t[2 * i * margin:(2 * i + 1) * margin][:] -= self.sigma * mx.nd.sign(grad[2 * i * margin:(2 * i + 1) * margin][:])
            batch_data_t[2 * i * margin:(2 * i + 1) * margin][:] = mx.nd.maximum(batch_data_t[2 * i * margin:(2 * i + 1) * margin][:], mx.nd.zeros_like(batch_data_t[2 * i * margin:(2 * i + 1) * margin][:]))
            batch_data_t[2 * i * margin:(2 * i + 1) * margin][:] = mx.nd.minimum(batch_data_t[2 * i * margin:(2 * i + 1) * margin][:], 255 * mx.nd.ones_like(batch_data_t[2 * i * margin:(2 * i + 1) * margin][:]))
        #print("first")
        for i in range(0, self.round - 1):
            db = mx.io.DataBatch([batch_data_t])
            self.model_adv.forward(db, is_train=True)
            adv_out = self.model_adv.get_outputs()[-1].asnumpy()
            #print("adv_dis: ", i, adv_out, np.max(adv_out))
            if np.max(adv_out) > self.thd:
                self.model_adv.backward()
                grad = self.model_adv.get_input_grads()[0]
                grad = mx.nd.array(grad)
                for i in xrange(self.ctx_num):
                    batch_data_t[2 * i * margin:(2 * i + 1) * margin][:] -= self.sigma * mx.nd.sign(
                        grad[2 * i * margin:(2 * i + 1) * margin][:])
                    batch_data_t[2 * i * margin:(2 * i + 1) * margin][:] = mx.nd.maximum(
                        batch_data_t[2 * i * margin:(2 * i + 1) * margin][:],
                        mx.nd.zeros_like(batch_data_t[2 * i * margin:(2 * i + 1) * margin][:]))
                    batch_data_t[2 * i * margin:(2 * i + 1) * margin][:] = mx.nd.minimum(
                        batch_data_t[2 * i * margin:(2 * i + 1) * margin][:],
                        255 * mx.nd.ones_like(batch_data_t[2 * i * margin:(2 * i + 1) * margin][:]))
            else:
                #print("adv_dis: ", i)
                break
        db = mx.io.DataBatch([batch_data_t])
        self.model_adv.forward(db, is_train=True)
        adv_out = self.model_adv.get_outputs()[-1].asnumpy()
        #print("adv_dis: ", adv_out)

        #imgadv_show = np.squeeze(batch_data_t[0][:].asnumpy())
        #imgadv_show = imgadv_show.astype(np.uint8)
        # print("imgadv_show.type: ", imgadv_show.astype)
        #imgadv_show = np.transpose(imgadv_show, (1, 2, 0))
        #plt.imshow(imgadv_show)
        #plt.show()

        for i in xrange(self.ctx_num):
            batch_data_adv[i * margin: (i + 1) * margin][:] = batch_data_t[2 * i * margin:(2 * i + 1) * margin][:]

        batch_data_src = batch_data_srctar[0:self.batch_size_src][:]
        batch_data_tar = batch_data_srctar[self.batch_size_src:2*self.batch_size_src][:]

        #for i in xrange(self.ctx_num):
        #    batch_data_tar[i * margin: (i + 1) * margin][:] = batch_data_t[(2 * i + 1) * margin:2 * (i + 1) * margin][:]

        batch_label_src = batch_label_srctar[0:self.batch_size_src][:]
        batch_label_tar = batch_label_srctar[self.batch_size_src:2 * self.batch_size_src][:]
        #print("labels: " , batch_label_src , batch_label_tar)


        margin = self.batch_size_src // self.main_ctx_num # 30
        for i in xrange(self.main_ctx_num): # 0 1 2 3
            batch_data[margin * 3 * i : margin * 3 * i + margin][:] = batch_data_src[margin * i :margin * i + margin][:]
            batch_data[margin * 3 * i + margin: margin * 3 * i + 2 * margin][:] = batch_data_tar[margin * i :margin * i + margin][:]
            batch_data[margin * 3 * i + 2 * margin: margin * 3 * i + 3 * margin][:] = batch_data_adv[margin * i :margin * i + margin][:]


        for i in xrange(self.main_ctx_num):
            batch_label[margin * 3 * i : margin * 3 * i + margin][:] = batch_label_src[margin * i :margin * i + margin][:]
            batch_label[margin * 3 * i + margin: margin * 3 * i + 2 * margin][:] = batch_label_tar[margin * i :margin * i + margin][:]
            batch_label[margin * 3 * i + 2 * margin: margin * 3 * i + 3 * margin][:] = batch_label_src[margin * i :margin * i + margin][:]

        #print("batch labels: ", batch_label)
        return io.DataBatch([batch_data], [batch_label])
        #return io.DataBatch([batch_data], [batch_label], batch_size - i)

    def check_data_shape(self, data_shape):
        """Checks if the input data shape is valid"""
        if not len(data_shape) == 3:
            raise ValueError('data_shape should have length 3, with dimensions CxHxW')
        if not data_shape[0] == 3:
            raise ValueError('This iterator expects inputs to have 3 channels.')

    def check_valid_image(self, data):
        """Checks if the input data is valid"""
        if len(data[0].shape) == 0:
            raise RuntimeError('Data shape is wrong')

    def imdecode(self, s):
        """Decodes a string or byte string to an NDArray.
        See mx.img.imdecode for more details."""
        img = mx.image.imdecode(s) #mx.ndarray
        return img

    def postprocess_data(self, datum):
        """Final postprocessing step before image is loaded into the batch."""
        return nd.transpose(datum, axes=(2, 0, 1))

    def get_symbol(self, arg_params, aux_params):
        print('init resnet', self.num_layers)
        embedding = fresnet_adv.get_symbol(self.emb_size, self.num_layers,
                                           version_se=self.version_se, version_input=self.version_input,
                                           version_output=self.version_output, version_unit=self.version_unit,
                                           version_act=self.version_act)
        embedding1 = mx.sym.slice_axis(embedding, axis=0, begin=0, end=self.batch_size_src // self.ctx_num)
        embedding2 = mx.sym.slice_axis(embedding, axis=0, begin=self.batch_size_src // self.ctx_num, end= 2 * self.batch_size_src // self.ctx_num)
        nembedding1 = mx.sym.L2Normalization(embedding1, mode='instance')
        nembedding2 = mx.sym.L2Normalization(embedding2, mode='instance')
        #nembedding2_t = mx.sym.transpose(nembedding2)
        #cosdis = 1 - mx.sym.dot(nembedding1, nembedding2_t)
        cos = mx.sym.broadcast_mul(nembedding1,nembedding2)
        cosdis = 1 - mx.sym.sum(cos,axis=1)
        all_label = mx.symbol.Variable('softmax_label')
        gt_label = all_label
        out_list = [mx.sym.BlockGrad(gt_label)]
        out_list.append(mx.symbol.BlockGrad(nembedding1))
        out_list.append(mx.symbol.BlockGrad(nembedding2))
        out_list.append(mx.sym.MakeLoss(cosdis))
        #out_list.append(mx.symbol.BlockGrad(cos))
        out = mx.symbol.Group(out_list)
        return (out, arg_params, aux_params)



