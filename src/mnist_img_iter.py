# THIS FILE IS FOR EXPERIMENTS, USE image_iter.py FOR NORMAL IMAGE LOADING.
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

import mxnet as mx
from mxnet import ndarray as nd
from mxnet import io

logger = logging.getLogger()

import ctypes
import threading

try:
    import h5py
except ImportError:
    h5py = None
import numpy as np

from mxnet.base import _LIB
from mxnet.base import c_str_array, mx_uint, py_str
from mxnet.base import DataIterHandle, NDArrayHandle
from mxnet.base import mx_real_t
from mxnet.base import check_call, build_param_doc as _build_param_doc
from mxnet.ndarray import NDArray
from mxnet.ndarray.sparse import CSRNDArray
from mxnet.ndarray.sparse import array as sparse_array
from mxnet.ndarray import _ndarray_cls
from mxnet.ndarray import array
from mxnet.ndarray import concatenate
from mxnet.ndarray import arange
from mxnet.ndarray.random import shuffle as random_shuffle
from mxnet.io import _init_data, _has_instance, _shuffle, DataDesc, DataBatch


class mnistIter(io.DataIter):
    def __init__(self, data, label=None, batch_size=1, shuffle=True,
                 last_batch_handle='pad', data_name='data',
                 label_name='softmax_label'):
        super(mnistIter, self).__init__(batch_size)


        print(data.shape, label.shape)
        self.data = _init_data(data, allow_empty=False, default_name=data_name)
        self.label = _init_data(label, allow_empty=True, default_name=label_name)

        # shuffle data
        if shuffle:
            tmp_idx = arange(self.data[0][1].shape[0], dtype=np.int32)
            self.idx = random_shuffle(tmp_idx, out=tmp_idx).asnumpy()
            self.data = _shuffle(self.data, self.idx)
            self.label = _shuffle(self.label, self.idx)
        else:
            self.idx = np.arange(self.data[0][1].shape[0])

        self.data_list = [x[1] for x in self.data] + [x[1] for x in self.label]
        #print('data_list :', self.data_list)
        self.num_source = len(self.data_list)
        #print('num_source: ', self.num_source)
        self.num_data = self.idx.shape[0]
        #print('num_data: ', self.num_data)
        self.cursor = -batch_size
        self.batch_size = batch_size
        self.last_batch_handle = last_batch_handle
        data_shape = (1, 28, 28)
        self.provide_data = [(data_name, (batch_size,) + data_shape)]
        self.provide_label = [(label_name, (batch_size,))]

    def reset(self):
        logging.debug("reset")
        tmp_idx = arange(self.data[0][1].shape[0], dtype=np.int32)
        self.idx = random_shuffle(tmp_idx, out=tmp_idx).asnumpy()
        self.data = _shuffle(self.data, self.idx)
        self.label = _shuffle(self.label, self.idx)
        self.cursor = 0
        logging.debug("self.cursor: ", self.cursor)

    def iter_next(self):
        #print("cursor: ", self.cursor)
        self.cursor += self.batch_size
        #print("cursor+: ", self.cursor)
        return self.cursor + self.batch_size < self.num_data

    def next(self):
        if self.iter_next():
            #print("return batch")
            return DataBatch(self.getdata(), self.getlabel())
        else:
            # print("raise StopIteration")
            # self.reset()
            raise StopIteration

    def _getdata(self, data_source):
        assert (self.cursor + self.batch_size < self.num_data)
        """Load data from underlying arrays, internal use only."""
        return [
            # np.ndarray or NDArray case
            x[1][self.cursor:self.cursor + self.batch_size] for x in data_source
        ]

    def getdata(self):
        return self._getdata(self.data)

    def getlabel(self):
        return self._getdata(self.label)


