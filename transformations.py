import abc
import itertools
import numpy as np
from keras.preprocessing.image import apply_affine_transform
# The code is adapted from https://github.com/izikgo/AnomalyDetectionTransformations/blob/master/transformations.py

def get_transformer(type_trans):
    if type_trans == 'complicated':
        tr_x, tr_y = 8, 8
        transformer = Transformer(tr_x, tr_y)
        return transformer
    elif type_trans == 'simple':
        transformer = SimpleTransformer()
        return transformer

class AffineTransformation_non90(object):
    def __init__(self, flip, tx, ty, k_90_rotate, angle):
        self.flip = flip
        self.theta = k_90_rotate * angle

    def __call__(self, x):
        res_x = x
        if self.flip:
            res_x = np.fliplr(res_x)
        zx = np.random.randint(4,9)/10
        zy = np.random.randint(4,9)/10
        res_x = apply_affine_transform(res_x,theta= self.theta, zx=zx, zy=zy, channel_axis=2, fill_mode='reflect')
        return res_x
        
class AffineTransformation(object):
    def __init__(self, flip, tx, ty, k_90_rotate):
        self.flip = flip
        self.tx = tx
        self.ty = ty
        self.k_90_rotate = k_90_rotate

    def __call__(self, x):
        res_x = x
        if self.flip:
            res_x = np.fliplr(res_x)
        if self.tx != 0 or self.ty != 0:
            res_x = apply_affine_transform(res_x,
            tx=self.tx, ty=self.ty, channel_axis=2, fill_mode='reflect')
        if self.k_90_rotate != 0:
            res_x = np.rot90(res_x, self.k_90_rotate)
        return res_x


class AbstractTransformer(abc.ABC):
    def __init__(self):
        self._transformation_list = None
        self._create_transformation_list()

    @property
    def n_transforms(self):
        return len(self._transformation_list)

    @abc.abstractmethod
    def _create_transformation_list(self):
        return

    def transform_batch(self, x_batch, t_inds):
        assert len(x_batch) == len(t_inds)

        transformed_batch = x_batch.copy()
        for i, t_ind in enumerate(t_inds):
            transformed_batch[i] = self._transformation_list[t_ind](transformed_batch[i])
        return transformed_batch


class Transformer(AbstractTransformer):
    def __init__(self, translation_x=8, translation_y=8):
        self.max_tx = translation_x
        self.max_ty = translation_y
        super().__init__()

    def _create_transformation_list(self):
        transformation_list = []
        for is_flip, tx, ty, k_rotate in itertools.product((False, True),
                                                           (0, -self.max_tx, self.max_tx),
                                                           (0, -self.max_ty, self.max_ty),
                                                           range(4)):
            transformation = AffineTransformation(is_flip, tx, ty, k_rotate)
            transformation_list.append(transformation)
        self._transformation_list = transformation_list
        return transformation_list


class SimpleTransformer(AbstractTransformer):
    def _create_transformation_list(self):
        transformation_list = []
        for is_flip, k_rotate in itertools.product((False, True),
                                                    range(4)):
            transformation = AffineTransformation(is_flip, 0, 0, k_rotate)
            transformation_list.append(transformation)
        self._transformation_list = transformation_list
        return transformation_list

class Transformer_non90(AbstractTransformer):
    def __init__(self, translation_x=8, translation_y=8, angle=90):
        self.max_tx = translation_x
        self.max_ty = translation_y
        self.angle = angle
        super().__init__()

    def _create_transformation_list(self):
        transformation_list = []
        if self.max_tx !=0:
            for is_flip, tx, ty, k_rotate in itertools.product((False, True),
                                                            (0, -self.max_tx, self.max_tx),
                                                            (0, -self.max_ty, self.max_ty),
                                                            range(4)):
                transformation = AffineTransformation_non90(is_flip, tx, ty, k_rotate, self.angle)
                transformation_list.append(transformation)
        else:
            for is_flip, k_rotate in itertools.product((False, True),
                                                            (-1,0,1)):
                transformation = AffineTransformation_non90(is_flip, 0, 0, k_rotate, self.angle)
                transformation_list.append(transformation)

        self._transformation_list = transformation_list