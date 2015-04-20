# orginal author Ian Goodfellow.
# modified by Lei Wang, alphaleiw@gmail.com

from pylearn2.datasets.dataset import Dataset

# Modified by plcarrier so that when get_test_set is called, 
# the raw dataset is returned instead of the TransformerDataset so
# that no transformation is performed on the test data

class TransformerDataset(Dataset):
    """
        A dataset that applies a transformation on the fly
        as examples are requested.
    """
    def __init__(self, raw, transformer, cpu_only = False,
            space_preserving=False):
        """
            raw: a pylearn2 Dataset that provides raw data
            transformer: a pylearn2 Block to transform the data
        """
        self.__dict__.update(locals())
        self.view_converter = self.raw.view_converter #########################
        self.y = self.raw.y
        del self.self

    def get_batch_design(self, batch_size, include_labels=False):
        raw = self.raw.get_batch_design(batch_size, include_labels)
        if include_labels:
            X, y = raw
        else:
            X = raw
        pX = X.copy()
        X = self.transformer.perform(X)
        if include_labels:
            return X, y
        return X

    def get_topo_batch_axis(self):
        return self.view_converter.axes.index('b')

    def get_test_set(self):
        return self.raw.get_test_set()


    def get_batch_topo(self, batch_size):
        """
        If the transformer has changed the space, we don't have a good
        idea of how to do topology in the new space.
        If the transformer just changes the values in the original space,
        we can have the raw dataset provide the topology.
        """
        X = self.get_batch_design(batch_size)
        if self.space_preserving:
            return self.raw.get_topological_view(X)
        return X.reshape(X.shape[0],X.shape[1],1,1)

    def iterator(self, mode=None, batch_size=None, num_batches=None,
                 topo=None, targets=None, rng=None, data_specs=None, return_tuple=False):

        raw_iterator = self.raw.iterator(mode, batch_size, num_batches, rng, data_specs, return_tuple)

        final_iterator = TransformerIterator(raw_iterator, self)

        return final_iterator

    def has_targets(self):
        return self.raw.y is not None

    def adjust_for_viewer(self, X):
        if self.space_preserving:
            return self.raw.adjust_for_viewer(X)
        return X

    def get_weights_view(self, *args, **kwargs):
        if self.space_preserving:
            return self.raw.get_weights_view(*args, **kwargs)
        raise NotImplementedError()

    def get_topological_view(self, *args, **kwargs):
        if self.space_preserving:
            return self.raw.get_weights_view(*args, **kwargs)
        raise NotImplementedError()

    def adjust_to_be_viewed_with(self, *args, **kwargs):
        return self.raw.adjust_to_be_viewed_with(*args, **kwargs)

    def get_num_examples(self):
      return  self.raw.get_num_examples()

class TransformerIterator(object):

    def __init__(self, raw_iterator, transformer_dataset):
        self.raw_iterator = raw_iterator
        self.transformer_dataset = transformer_dataset
        self.stochastic = raw_iterator.stochastic
        # self.uneven = raw_iterator.uneven

    def __iter__(self):
        return self

    def next(self):

        raw_batch = self.raw_iterator.next()

        # if self.raw_iterator._targets:
        rval = (self.transformer_dataset.transformer.perform(raw_batch[0]), raw_batch[1])
        # else:
        #    rval = self.transformer_dataset.transformer.perform(raw_batch)

        return rval

    @property
    def num_examples(self):
        return self.raw_iterator.num_examples
