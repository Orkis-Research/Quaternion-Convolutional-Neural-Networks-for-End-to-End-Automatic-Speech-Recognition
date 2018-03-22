import tables
from collections import OrderedDict

from fuel.datasets.hdf5 import PytablesDataset

from utils import make_local_copy


class Timit(PytablesDataset):
    """TIMIT dataset.

    Parameters
    ----------
    which_set : str, opt
        either 'train', 'dev' or 'test'.
    alignment : bool
        Whether return alignment.
    features : str
        The features to use. They will lead to the correct h5 file.

    """

    def __init__(self, which_set='train', local_copy=False, **kwargs):
        #self.path = '/home/parcollt/projects/rpp-bengioy/parcollt/Deep-Quaternary-Convolutional-Neural-Networks/TIMIT/timit_fbank_energy_deltas.h5'
        self.path = '/u/parcollt/WORKSPACE/QCNN/Deep-Quaternary-Convolutional-Neural-Networks/TIMIT/timit_fbank_energy_deltas.h5'
        #self.path = '/Users/titouanparcollet/CloudStation/LABO/WORKSPACE/EXPS/QCNN/timit_fbank_energy_deltas.h5'
        if local_copy and not self.path.startswith('/Tmp'):
            self.path = make_local_copy(self.path)
        self.which_set = which_set
        self.sources = ('features', 'features_shapes', 'phonemes')
        super(Timit, self).__init__(
            self.path, self.sources, data_node=which_set, **kwargs)

    def get_phoneme_dict(self):
        phoneme_list = self.h5file.root._v_attrs.phones_list
        return OrderedDict(enumerate(phoneme_list))

    def get_phoneme_ind_dict(self):
        phoneme_list = self.h5file.root._v_attrs.phones_list
        return OrderedDict(zip(phoneme_list, range(len(phoneme_list))))

    def get_normalization_factors(self):
        means = self.h5file.root._v_attrs.means
        stds = self.h5file.root._v_attrs.stds
        return means, stds

    def open_file(self, path):
        self.h5file = tables.open_file(path, mode="r")
        node = self.h5file.get_node('/', self.data_node)

        self.nodes = [getattr(node, source) for source in self.sources_in_file]
        if self.stop is None:
            self.stop = self.nodes[0].nrows
        self.num_examples = self.stop - self.start
