from pylearn2.datasets.preprocessing import ExamplewisePreprocessor
import cPickle
import os

class Standardize(ExamplewisePreprocessor):
    """
    Subtracts the mean and divides by the standard deviation.

    Parameters
    ----------
    global_mean : bool, optional
        If `True`, subtract the (scalar) mean over every element
        in the design matrix. If `False`, subtract the mean from
        each column (feature) separately. Default is `False`.
    global_std : bool, optional
        If `True`, after centering, divide by the (scalar) standard
        deviation of every element in the design matrix. If `False`,
        divide by the column-wise (per-feature) standard deviation.
        Default is `False`.
    std_eps : float, optional
        Stabilization factor added to the standard deviations before
        dividing, to prevent standard deviations very close to zero
        from causing the feature values to blow up too much.
        Default is `1e-4`.
    """

    def __init__(self, global_mean=True, global_std=True, std_eps=1e-4, mean_std_file = None):
        self._global_mean = global_mean
        self._global_std = global_std
        self._std_eps = std_eps
        self._mean = None
        self._std = None
        self._mean_std_file = mean_std_file
        if os.path.exists(mean_std_file):
          stats = cPickle.load(open(mean_std_file,'r'))
          self._mean = stats['mean']
          self._std = stats['std']

    def apply(self, dataset, can_fit=False):
        """
        .. todo::

            WRITEME
        """
        X = dataset.get_design_matrix()
        if can_fit:
            self._mean = X.mean() if self._global_mean else X.mean(axis=0)
            self._std = X.std() if self._global_std else X.std(axis=0)
            cPickle.dump({'mean': self._mean,'std': self._std}, open(self._mean_std_file,'w'))
        else:
            if self._mean is None or self._std is None:
                raise ValueError("can_fit is False, but Standardize object "
                                 "has no stored mean or standard deviation")
        new = (X - self._mean) / (self._std_eps + self._std)
        dataset.set_design_matrix(new)

    def as_block(self):
        """
        .. todo::

            WRITEME
        """
        if self._mean is None or self._std is None:
            raise ValueError("can't convert %s to block without fitting"
                             % self.__class__.__name__)
        return ExamplewiseAddScaleTransform(add=-self._mean,
                                            multiply=self._std ** -1)
