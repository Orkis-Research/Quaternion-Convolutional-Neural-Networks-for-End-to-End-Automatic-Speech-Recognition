import six
import math
import numpy
from picklable_itertools import imap
from picklable_itertools.extras import partition_all
from fuel.schemes import BatchScheme


class SequentialShuffledScheme(BatchScheme):
    """Sequential batches iterator.

    Iterate over all the examples in a dataset of fixed size sequentially
    in batches of a given size.

    Notes
    -----
    The batch size isn't enforced, so the last batch could be smaller.

    """
    def __init__(self, num_examples, batch_size, rng):
        self.num_examples = num_examples
        self.batch_size = batch_size
        self.rng = rng

    def get_request_iterator(self):
        return SequentialShuffledIterator(self.num_examples, self.batch_size,
                                          self.rng)

class SequentialShuffledIterator(six.Iterator):
    def __init__(self, num_examples, batch_size, rng):
        self.num_examples = num_examples
        self.batch_size = batch_size
        self.rng = rng
        self.batch_indexes = range(int(math.ceil(num_examples/ float(batch_size))))
        self.rng.shuffle(self.batch_indexes)
        self.current = 0
        self.current_batch = 0

    def __iter__(self):
        self.rng.shuffle(self.batch_indexes)
        return self

    def __next__(self):
        if self.current >= self.num_examples:
            raise StopIteration
        current_index = self.batch_indexes[self.current_batch]
        slice_ = slice(current_index * self.batch_size,
                       min(self.num_examples,
                           (current_index + 1) * self.batch_size))
        self.current += self.batch_size
        self.current_batch += 1
        return slice_

