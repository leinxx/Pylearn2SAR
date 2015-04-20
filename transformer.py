import numpy as np
from theano import config
from scipy import ndimage, misc
from pylearn2.train_extensions import TrainExtension
from pylearn2.monitor import Monitor
from pylearn2.gui import patch_viewer

# Decorator to reshape batch data before and after transformations.
# Transformations need a shape of ['b',0,1,'c']
# The class to which belongs the wrapped function must have an attribute
# input_space which is a Conv2DSpace object.
def reshape(default=['b',0,1,'c']):
    def decorate(perform):
        def call(self,X):

            # convert X shape
            shape = X.shape
            if self.input_space:
                # needs reshaping
                if len(self.input_space.axes) != len(shape):
                    X = X.reshape([X.shape[0]]+self.input_space.shape+[self.input_space.num_channels])

                # dimension transposition
                else:
                    # How can we detect axes of X?
                    X = self.input_space.convert_numpy(X,self.input_space.axes,default)

            # apply perform function

            result = perform(self,X)

            # revert X shape

            if self.input_space:
                # needs reshaping
                if len(self.input_space.axes) != len(shape):
                    X = X.reshape(shape)

                # dimension transposition
                else:
                    # How can we detect axes of X?
                    X = self.input_space.convert_numpy(X,default,self.input_space.axes)

            return X

        return call

    return decorate


class TransformationPipeline:
    """
        Apply transformations sequentially
    """
    # shape should we switched to Conv2DSpace or something similar to make space conversion more general
    def __init__(self,transformations,seed=None,shape=None,input_space=None):
        self.transformations = transformations
        self.input_space = input_space

        if seed:
            np.random.RandomState(seed)

    @reshape()
    def perform(self, X):

        for transformation in self.transformations:
            X = transformation.perform(X)

        return X

class TransformationPool:
    """
        Apply one of the transformations given the probability distribution
        default : equal probability for every transformations
    """
    def __init__(self,transformations,p_distribution=None,seed=None,input_space=None):
        self.transformations = transformations
        self.input_space = input_space

        if p_distribution:
            if np.sum(p_distribution)<0.9999 or np.sum(p_distribution)>1.0001:
                raise ValueError("p_distribution does not sum to 1")
            self.p_distribution = p_distribution
        else:
            self.p_distribution = np.ones(len(transformations))/(.0+len(transformations))

        if seed:
            np.random.RandomState(seed)

    @reshape()
    def perform(self, X):

        for i in xrange(X.shape[0]):
            # randomly pick one transformation according to probability distribution
            t = np.array(self.p_distribution).cumsum().searchsorted(np.random.sample(1))

            X[i] = self.transformations[t[0]].perform(X[i])

        return X

def gen_mm_random(f,args,min,max,max_iteration=50):
    if min==None or min=="None":
        min = -1e99

    if max==None or max=="None":
        max = 1e99

    i = 0
    r = f(**args)
    while (r < min or max < r) and i < max_iteration:
        r = f(**args)
        i += 1

    if i >= max_iteration:
        raise RuntimeError("Were not able to generate a random value between "+str(min)+" and "+str(max)+" in less than "+str(i)+" iterations")

    return r

class RandomTransformation:
    def __init__(self,p):
        self.p = p

    def perform(self,X):

        for i in xrange(X.shape[0]):
            p = np.random.random(1)[0]
            if p < self.p:
                X[i] = self.transform(X[i])

        return X

    def transform(self,X):
        return X

class Translation(RandomTransformation):
    def __init__(self,p=1,random_fct='normal',fct_settings=None,min=None,max=None):
        if not fct_settings:
            fct_settings = dict(loc=0.0,scale=2.0)
        RandomTransformation.__init__(self,p)
        self.__dict__.update(locals())

        self.I = np.identity(3)

    def transform(self,x):

        rx = gen_mm_random(np.random.__dict__[self.random_fct],self.fct_settings,self.min,self.max)
        ry = gen_mm_random(np.random.__dict__[self.random_fct],self.fct_settings,self.min,self.max)

        return ndimage.affine_transform(x,self.I,offset=(rx,ry,0))


class Croping(RandomTransformation):
    def __init__(self,p, shape):

        RandomTransformation.__init__(self,p)
        self.__dict__.update(locals())

    def transform(self, x):

        im = np.zeros(x.shape)
        # crop channels individually
        print x.shape
        for c in xrange(x.shape[2]):
            im_c = x[:,:,c]

            try:
                im[:,:,c] = self._crop(im_c,shape)
            except ValueError as e:
                # won't crop so better return it already
                return x

        # balance channels

        return im

    def _crop(self,x,shape):
        """
            in 2D (x,y)
        """
        y = np.zeros(shape)


        if x.shape[0] < shape[0]:
            y[int((y.shape[0]-x.shape[0])/2.0+0.5):int(-(y.shape[0]-x.shape[0])/2.0),
              int((y.shape[1]-x.shape[1])/2.0+0.5):int(-(y.shape[1]-x.shape[1])/2.0)] = x
        else:
            y = x[int((x.shape[0]-y.shape[0])/2.0+0.5):int(-(x.shape[0]-y.shape[0])/2.0),
                  int((x.shape[1]-y.shape[1])/2.0+0.5):int(-(x.shape[1]-y.shape[1])/2.0)]

        return y

class Scaling(RandomTransformation):
    def __init__(self,p=1,random_fct='normal',fct_settings=None,min=0.1,max=None):
        if not fct_settings:
            fct_settings = dict(loc=1.0,scale=0.1)

        RandomTransformation.__init__(self,p)
        self.__dict__.update(locals())

        pass

    def transform(self, x):

        zoom = gen_mm_random(np.random.__dict__[self.random_fct],self.fct_settings,self.min,self.max)
        shape = x.shape
        im = np.zeros(x.shape)
        # crop channels individually
        print x.shape
        for c in xrange(x.shape[2]):
            im_c = ndimage.zoom(x[:,:,c],zoom)

            try:
                im[:,:,c] = self._crop(im_c,shape[:2])
            except ValueError as e:
                # won't crop so better return it already
                return x

        # balance channels

        return im

    def _crop(self,x,shape):
        """
            in 2D (x,y)
        """
        y = np.zeros(shape)


        if x.shape[0] < shape[0]:
            y[int((y.shape[0]-x.shape[0])/2.0+0.5):int(-(y.shape[0]-x.shape[0])/2.0),
              int((y.shape[1]-x.shape[1])/2.0+0.5):int(-(y.shape[1]-x.shape[1])/2.0)] = x
        else:
            y = x[int((x.shape[0]-y.shape[0])/2.0+0.5):int(-(x.shape[0]-y.shape[0])/2.0),
                  int((x.shape[1]-y.shape[1])/2.0+0.5):int(-(x.shape[1]-y.shape[1])/2.0)]

        return y

class Rotation(RandomTransformation):
    def __init__(self,p=1,random_fct='normal',fct_settings=None,min=None,max=None):
        if not fct_settings:
            fct_settings = dict(loc=0.0,scale=10.0)

        RandomTransformation.__init__(self,p)
        self.__dict__.update(locals())

        pass

    def transform(self,x):

        degree = gen_mm_random(np.random.__dict__[self.random_fct],self.fct_settings,self.min,self.max)

        return ndimage.rotate(x,degree,reshape=False)

class SaveTransform(TrainExtension):
    """
        Save a png of the transform every i epoch
    """
    def __init__(self, freq, save_path, rows=20, cols=20, rescale='global'):
        """
            freq: how often a png is saved
            save_path: should contain %d to save the epoch number
        """

        self.__dict__.update(locals())
        del self.self
        self._initialized = False

    def on_monitor(self, model, dataset, algorithm):

        if self.rescale == 'none':
            global_rescale = False
            patch_rescale = False
        elif self.rescale == 'global':
            global_rescale = True
            patch_rescale = False
        elif self.rescale == 'individual':
            global_rescale = False
            patch_rescale = True
        else:
            assert False

        # implement saving from show_examples
        examples = dataset.get_batch_topo(self.rows*self.cols)
        examples = dataset.adjust_for_viewer(examples)

        if global_rescale:
            examples /= np.abs(examples).max()

        if examples.shape[3] == 1:
            is_color = False
        elif examples.shape[3] == 3:
            is_color = True

        pv = patch_viewer.PatchViewer( (self.rows, self.cols), examples.shape[1:3], is_color=is_color)
        for i in xrange(self.rows*self.cols):
            pv.add_patch(examples[i,:,:,:], activation = 0.0, rescale = patch_rescale)

#        pv.show()

        pv.save(self.save_path % len(model.monitor.channels[model.monitor.channels.keys()[0]].val_record))
        print "dataset image saved"


class GaussianNoise(RandomTransformation):
    def __init__(self,p,sigma=1.0):
        """
            Sigma: from 1.0 to 5.0
        """
        RandomTransformation.__init__(self,p)
        self.__dict__.update(locals())
        pass

    def transform(self,x):

        return ndimage.gaussian_filter(x,self.sigma)

class Sharpening(RandomTransformation):
    def __init__(self,p,sigma=[3,1],alpha=30):
        RandomTransformation.__init__(self,p)
        self.__dict__.update(locals())
        pass

    def transform(self,x):

        blurred_l = ndimage.gaussian_filter(x, int(self.sigma[0]))

        filter_blurred_l = ndimage.gaussian_filter(blurred_l, int(self.sigma[1]))
        return blurred_l + int(self.alpha) * (blurred_l - filter_blurred_l)

class Noise(RandomTransformation):
    def __init__(self,p,alpha=1.0):
        RandomTransformation.__init__(self,p)
        self.__dict__.update(locals())

    def transform(self,x):
        return x + self.alpha.get_value()*x.std()*np.random.random(x.shape)

class Denoising(RandomTransformation):
    def __init__(self,p,sigma=[2,3],alpha=0.4):
        RandomTransformation.__init__(self,p)
        self.__dict__.update(locals())
        pass

    def transform(self,x):
        noisy = x + self.alpha * x.std() * np.random.random(x.shape)

        gauss_denoised = ndimage.gaussian_filter(noisy, self.sigma[0])

        return ndimage.median_filter(noisy, self.sigma[1])

class Flipping(RandomTransformation):
    def __init__(self,p=0.25):
        RandomTransformation.__init__(self,p)
        self.__dict__.update(locals())
        pass

    def transform(self,x):

        return np.fliplr(x)

class Occlusion:

    # Random function given as a parameter controls the size of the boxes, not their positions
    # Boxes potisions are sampled with numpy.random.random_sample

    def __init__(self,p=0.05,nb=10,random_fct='uniform',fct_settings=None,min=2,max=None):
        if not fct_settings:
            fct_settings = dict(low=2.0,high=20.0)

        self.__dict__.update(locals())
        pass

    def perform(self,X):

        for i in xrange(X.shape[0]):
            X[i] = self.transform(X[i])

        return X

    def transform(self,image):
        nb = np.sum(np.random.random(self.nb)<self.p)
        for i in xrange(nb):

            dx = int(gen_mm_random(np.random.__dict__[self.random_fct],
                               self.fct_settings,
                               self.min,image.shape[0]-2))
            dy = int(gen_mm_random(np.random.__dict__[self.random_fct],
                               self.fct_settings,
                               self.min,image.shape[1]-2))

            x = int(gen_mm_random(np.random.uniform,dict(low=0,high=image.shape[0]-dx),
                                  0,image.shape[0]-dx))

            y = int(gen_mm_random(np.random.uniform,dict(low=0,high=image.shape[0]-dy),
                                  0,image.shape[1]-dy))

            image[x:(x+dx),y:(y+dy)] = np.zeros((dx,dy,image.shape[2])).astype(int)

        return image

class HalfFace(RandomTransformation):
    def __init__(self,p=0.05):
        RandomTransformation.__init__(self,p)
        self.__dict__.update(locals())
        pass

    def transform(self,image):
        x_center = image.shape[0]/2
        y_center = image.shape[1]/2
        if np.random.rand(1) >= 0.5:
            image[:,:y_center] = np.zeros(image[:,:y_center].shape)#(np.random.rand(*image[:x_center,:y_center].shape)*255).astype(int)
        else:
            image[:,y_center:] = np.zeros(image[:,:y_center].shape)#(np.random.rand(*image[:,y_center:].shape)*255).astype(int)
        return image

if __name__ == "__main__":

    import matplotlib.pyplot as plt

    d = np.load('/data/afew/facetubes/sample.npy')
    print d.shape

#    plt.imshow(d)
#    plt.show()

    # translation
#    t = Translation(p=1,random_fct='normal',fct_settings=dict(loc=0.0,scale=10.0),min=None,max=None)
#    plt.imshow(t.transform(d))
#    plt.show()

    # scale
#    t = Scaling(1,random_fct='normal',fct_settings=dict(loc=1.0,scale=0.2),min=0.1,max=None)
#    i = t.transform(d).astype('uint8')
#    plt.imshow(i)
#    plt.show()

    # rotation
#    t = Rotation(p=1,random_fct='normal',fct_settings=dict(loc=0.0,scale=20.0),min=None,max=None)
#    i = t.transform(d)
#    plt.imshow(i)
#    plt.show()

    # occlusion
#    t = Occlusion(p=0.5,nb=10,random_fct='uniform',fct_settings=None,min=2,max=None)
#    i = t.transform(d)
#    plt.imshow(i)
#    plt.show()
