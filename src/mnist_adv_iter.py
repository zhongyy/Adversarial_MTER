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
import multiprocessing
sys.path.append(os.path.join(os.path.dirname(__file__), 'symbols'))
from symbol_resnet_adv import resnet_o_adv
from symbol_lenet import lenet
logger = logging.getLogger()


class MnistADVIter(io.DataIter):

    def __init__(self, train_data, train_label, batch_size_src, data_shape, model = None,
                 data_name='data', label_name='softmax_label',
                 emb_size = None, num_layers = None, network= None,
                 ctx= None, ctxnum= None, main_ctx_num = None,
                 adv_round= None, adv_thd=None, adv_sigma=None, aug_list=None, **kwargs):

        super(MnistADVIter, self).__init__()
        logging.info("init")
        self.network = network
        self.train_data = train_data
        self.train_label = train_label

        self.sigma = adv_sigma
        self.thd = adv_thd
        self.round = adv_round
        self.emb_size = emb_size
        self.num_layers = num_layers
        self.provide_data = [(data_name, (3 * batch_size_src,) + data_shape)]
        self.batch_size = batch_size_src*3
        self.batch_size_src = batch_size_src
        self.data_shape = data_shape
        self.image_size = '%d,%d'%(data_shape[1],data_shape[2])
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
      id = np.arange(10)
      random.shuffle(id)
      id1 = id[0:5]
      id2 = id[5:10]
      print(id1, id2)
      idx_1 = None
      idx_2 = None
      for i in xrange(5):
          if i == 0:
              idx_1 = np.where(np.array(self.train_label) == id1[i])[0]
          else:
              idx_1 = np.concatenate((idx_1, np.where(np.array(self.train_label) == id1[i])[0]), axis=0)
      for i in xrange(5):
          if i == 0:
              idx_2 = np.where(np.array(self.train_label) == id2[i])[0]
          else:
              idx_2 = np.concatenate((idx_2, np.where(np.array(self.train_label) == id2[i])[0]), axis=0)
      #print(idx_1, idx_1.shape)
      #print(idx_2, idx_2.shape)
      #print(np.array(self.train_label).shape)
      random.shuffle(idx_1)
      random.shuffle(idx_2)
      self.train_data1 = np.delete(self.train_data, idx_2, axis=0)
      self.train_data2 = np.delete(self.train_data, idx_1, axis=0)
      self.train_label1 = np.delete(self.train_label, idx_2, axis=0)
      self.train_label2 = np.delete(self.train_label, idx_1, axis=0)
      print("oseq1: ", len(self.train_label1))
      print("oseq2: ", len(self.train_label2))

      if self.model_adv_init == False:
          arg_t, aux_t = self.model.get_params()
          sym, arg_params, aux_params = self.get_symbol(arg_t, aux_t)
          self.model_adv = mx.mod.Module(context=self.ctx, symbol=sym)
          provide_data = [('data', (2 * self.batch_size_src, 1, self.data_shape[1], self.data_shape[2]))]
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
        #print("s1")
        if self.train_label1 is not None:
          while True:
            if self.cur1 + self.batch_size_src >= len(self.train_label1):
                raise StopIteration
            self.cur1 += 1
            img = self.train_data1[self.cur1][:]
            label = self.train_label1[self.cur1]
            return label, img

    def next_sample2(self):
        #print("s2")
        if self.train_label2 is not None:
          while True:
            if self.cur2 + self.batch_size_src >= len(self.train_label2):
                raise StopIteration
            self.cur2 += 1
            img = self.train_data2[self.cur2][:]
            label = self.train_label2[self.cur2]
            return label, img

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

        arg_t, aux_t = self.model.get_params()
        self.model_adv.set_params(arg_t, aux_t)
        #print("update model_adv params")
        #time_now2 = datetime.datetime.now()
        #print("update params time", time_now2-time_now1)

        i = 0
        try:
            while i < self.batch_size_src:
                label, img = self.next_sample1()

                batch_data_srctar[i][:] = img
                batch_label_srctar[i][:] = label
                i += 1
        except StopIteration:
            if i< self.batch_size_src:
                raise StopIteration
        try:
            while i < 2 * self.batch_size_src:
                label, img = self.next_sample2()
                #print(img)
                #img_show = np.squeeze(img)
                #img_show = img_show.astype(np.uint8)
                #print("img.shape:", img_show.shape)
                #plt.imshow(img_show)
                #plt.show()
                batch_data_srctar[i][:] = img
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

        #print("batch_data_t:", batch_data_t[0][:],batch_data_t)
        batch_data_t_o = batch_data_t
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
            batch_data_t[2 * i * margin:(2 * i + 1) * margin][:] -= 1/255 * mx.nd.sign(grad[2 * i * margin:(2 * i + 1) * margin][:])
            batch_data_t[2 * i * margin:(2 * i + 1) * margin][:] = mx.nd.maximum(mx.nd.maximum(batch_data_t[2 * i * margin:(2 * i + 1) * margin][:], batch_data_t_o[2 * i * margin:(2 * i + 1) * margin][:] - self.sigma),mx.nd.zeros_like(batch_data_t[2 * i * margin:(2 * i + 1) * margin][:]))
            batch_data_t[2 * i * margin:(2 * i + 1) * margin][:] = mx.nd.minimum(batch_data_t[2 * i * margin:(2 * i + 1) * margin][:], mx.nd.minimum(batch_data_t_o[2 * i * margin:(2 * i + 1) * margin][:] + self.sigma, mx.nd.ones_like(batch_data_t[2 * i * margin:(2 * i + 1) * margin][:])))
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
                    batch_data_t[2 * i * margin:(2 * i + 1) * margin][:] -= 1 / 255 * mx.nd.sign(
                        grad[2 * i * margin:(2 * i + 1) * margin][:])
                    batch_data_t[2 * i * margin:(2 * i + 1) * margin][:] = mx.nd.maximum(
                        mx.nd.maximum(batch_data_t[2 * i * margin:(2 * i + 1) * margin][:],
                                      batch_data_t_o[2 * i * margin:(2 * i + 1) * margin][:] - self.sigma),
                        mx.nd.zeros_like(batch_data_t[2 * i * margin:(2 * i + 1) * margin][:]))
                    batch_data_t[2 * i * margin:(2 * i + 1) * margin][:] = mx.nd.minimum(
                        batch_data_t[2 * i * margin:(2 * i + 1) * margin][:],
                        mx.nd.minimum(batch_data_t_o[2 * i * margin:(2 * i + 1) * margin][:] + self.sigma,
                                      mx.nd.ones_like(batch_data_t[2 * i * margin:(2 * i + 1) * margin][:])))
            else:
                #print("adv_dis: ", i)
                break
        db = mx.io.DataBatch([batch_data_t])
        self.model_adv.forward(db, is_train=True)
        adv_out = self.model_adv.get_outputs()[-1].asnumpy()
        #print("adv_dis: ", adv_out)
        '''
        for i in xrange(5):
            imgadv_show = np.squeeze(batch_data_t[i][0][:].asnumpy())
            imgadv_show = imgadv_show.astype(np.uint8)
            print("imgadv_show.type: ", imgadv_show.astype)
            #imgadv_show = np.transpose(imgadv_show, (1, 2, 0))
            plt.imshow(imgadv_show)
            plt.show()
        '''
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

        #print('batch_label: ', batch_label)
        '''
        for i in xrange(2):
            imgadv_show = np.squeeze(batch_data[i][0][:].asnumpy())
            print(imgadv_show)
            #imgadv_show = imgadv_show.astype(np.uint8)
            print(imgadv_show)
            print("imgadv_show.type: ", imgadv_show.astype)
            #imgadv_show = np.transpose(imgadv_show, (1, 2, 0))
            plt.imshow(imgadv_show)
            plt.show()
        '''
        return io.DataBatch([batch_data], [batch_label])

    def get_symbol(self, arg_params, aux_params):
        print('init resnet', self.num_layers)
        if self.network[0] == 'o':
            if self.num_layers == 10:
                units = [1, 1, 1, 1]
            elif self.num_layers == 18:
                units = [2, 2, 2, 2]
            elif self.num_layers == 34:
                units = [3, 4, 6, 3]
            elif self.num_layers == 50:
                units = [3, 4, 6, 3]
            embedding = resnet_o_adv(units=units, num_stage=4, filter_list=[64, 256, 512, 1024, 2048] if self.num_layers >= 50
                else [64, 64, 128, 256, 512], num_class=10, data_type="cifar10", bottle_neck=True
                if self.num_layers >= 50 else False, bn_mom=0.9, workspace=512,
                                     memonger=False)
        elif self.network[0] == 'l':
            embedding = lenet()
        elif self.network[0] == 's':
            if self.num_layers == 5:
                units = [1, 1]
                embedding = resnet_o_adv(units=units, num_stage=2, filter_list=[32, 64, 128], num_class=10,
                                     data_type="cifar10", bottle_neck=True
                    if self.num_layers >= 50 else False, bn_mom=0.9, workspace=512,
                                     memonger=False)
            if self.num_layers == 8:
                units = [1, 1, 1]
                embedding = resnet_o_adv(units=units, num_stage=3, filter_list=[32, 64, 64, 128],
                                     num_class=10, data_type="cifar10", bottle_neck=True
                    if self.num_layers >= 50 else False, bn_mom=0.9, workspace=512,
                                     memonger=False)

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



