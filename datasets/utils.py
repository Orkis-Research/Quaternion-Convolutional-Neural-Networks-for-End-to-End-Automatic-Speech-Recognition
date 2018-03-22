import hashlib
import os
import os
import shutil
from fuel.streams import DataStream
from fuel.transformers import (Mapping, ForceFloatX, Padding,
                               SortMapping, Cast)
from fuel.schemes import ShuffledScheme
from schemes import SequentialShuffledScheme
from transformers import (MaximumFrameCache, Transpose, Normalize,
                          Reshape, Subsample, ConvReshape, DictRep)

phone_to_phoneme_dict = {'ao':   'aa',
                         'ax':   'ah',
                         'ax-h': 'ah',
                         'axr':  'er',
                         'hv':   'hh',
                         'ix':   'ih',
                         'el':   'l',
                         'em':   'm',
                         'en':   'n',
                         'nx':   'n',
                         'eng':   'ng',
                         'zh':   'sh',
                         'pcl':  'sil',
                         'tcl':  'sil',
                         'kcl':  'sil',
                         'bcl':  'sil',
                         'dcl':  'sil',
                         'gcl':  'sil',
                         'h#':   'sil',
                         'pau':  'sil',
                         'epi':  'sil',
                         'ux':   'uw'}

def file_hash(afile, blocksize=65536):
    buf = afile.read(blocksize)
    hasher = hashlib.md5()
    while len(buf) > 0:
        hasher.update(buf)
        buf = afile.read(blocksize)
    return hasher.digest()

def make_local_copy(filename):
    local_name = os.path.join('/Tmp/', os.environ['USER'],
                              os.path.basename(filename))
    if (not os.path.isfile(local_name) or
                file_hash(open(filename)) != file_hash(open(local_name))):
        print '.. made local copy at', local_name
        shutil.copy(filename, local_name)
    return local_name

def key(x):
    return x[0].shape[0]

def construct_conv_stream(dataset, rng, pool_size, maximum_frames,
                          quaternion=False, **kwargs):
    """Construct data stream.
    Parameters:
    -----------
    dataset : Dataset
        Dataset to use.
    rng : numpy.random.RandomState
    Random number generator.
    pool_size : int
        Pool size for TIMIT dataset.
    maximum_frames : int
        Maximum frames for TIMIT datset.
    subsample : bool, optional
        Subsample features.
    """
    stream = DataStream(
        dataset,
        iteration_scheme=SequentialShuffledScheme(dataset.num_examples,
                                                  pool_size, rng))
    stream = Reshape('features', 'features_shapes', data_stream=stream)
    means, stds = dataset.get_normalization_factors()
    stream = Normalize(stream, means, stds)
    stream.produces_examples = False
    stream = Mapping(stream,
                     SortMapping(key=key))
    stream = MaximumFrameCache(max_frames=maximum_frames, data_stream=stream,
                               rng=rng)
    stream = Padding(data_stream=stream,
                     mask_sources=['features', 'phonemes'])
    stream = Transpose(stream, [(0, 1, 2), (1, 0), (0, 1), (1, 0)])
    stream = ConvReshape('features',
                          data_stream=stream, quaternion=quaternion)
    stream = Transpose(stream, [(0, 2, 3, 1), (0, 1), (0, 1), (0, 1)])
    stream.produces_examples = False
    stream = Cast(stream, 'int32', which_sources=('phonemes',))
    stream = ForceFloatX(stream)
    return DictRep(stream)
