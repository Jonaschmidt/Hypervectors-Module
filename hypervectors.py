# TODO: docstrings, especially of the static methods
# TODO: determine class/function structure
# TODO: support for binary and polar Hypervectors?

from typing import Literal, Optional, Tuple
import tensorflow as tf
import math
import numpy as np
from scipy.stats import qmc

# globals for Sobol generation...
sobol_dim = 0
sobol_indexer = 0
sobol_seqs = None
# TODO: give function better name
def _comp_each_to_val(x, val):
    return 1 - (2 * (x > val))
_comp_each_to_val = np.vectorize(_comp_each_to_val)
# ...

# TODO: be more clear about value everywhere, should it just always be a ratio?
def gen_tensor_sobol(value, size, value_range: Optional[Tuple[int, int]] = None):
    pass

# TODO: full implementation of Sobol sequences
# TODO: fix or remove or update the value_range and value attribute:
#           (consider that these can/should be "overridden" when generating LHVs)
# TODO: docstrings
class Hypervector:
    # TODO: update these docstrings
    '''
    A class to represent a Hypervector.

    Attributes:
        size (int): The size of the Hypervector
        value (int): The value represented by the Hypervector
        value_range (Tuple[int, int]): Default (0, size), is used when generating the tensor and is most useful when the Hypervector can represent negative numbers
        tensor (tensorflow.python.framework.ops.EagerTensor): The actual Hypervector, represented as a tensor
    '''

    def __init__(self, 
                 size: Optional[int]=256, 
                 value: Optional[int]=None, 
                 value_range: Optional[Tuple[int, int]]=None, 
                 tensor: Optional[tf.Tensor]=None,
                 random_method: Literal["tf_random", "Sobol"]="tf_random"):
        '''
        Initializes a Hypervector with the given size, value, and value range.

        Args:
            size (int): The size of the Hypervector, must be positive.
            value (Optional[int]): The value represented by the Hypervector. Default is None, which sets the value to half the size.
            value_range (Optional[Tuple[int, int]]): The range of possible values the Hypervector can represent. Default is None, which sets the range to (0, size).
            tensor (Optional[tf.Tensor]): Can be used to initialize the tensor of a Hypervector with finer control.
            random_method (Literal["built-in", "Sobol"]): Determines method of Hypervector generation:
                - "tf_random" specifies utilization of the built-in tensorflow random method
                - "Sobol" specifies utilization of Sobol sequences to generate Hypervectors, as described in the literature [TODO: citation]
        '''

        if size <= 0:
            raise ValueError("Size must be a positive integer.")        

        if value != None and tensor != None:
            raise ValueError("'tensor' and 'value' cannot simultaneously be provided.")
        
        self.size = size
        self.value = value

        if tensor != None:
            self.tensor = tensor
            self.value = self._generate_value()
            self.size = self.tensor.shape[0]
        else:
            self.tensor = self._generate_tensor(value=self.value, value_range=value_range, random_method=random_method)

    def _generate_tensor(self, value, value_range: Optional[Tuple[int, int]] = None, random_method: Literal["tf_random", "Sobol"]="tf_random") -> tf.Tensor:
        '''
        Generates the tensor representation of the Hypervector.
        Two generation methods are supported
        (TODO: describe these methods in detail:)
        - tf_random
        - Sobol

        Args:
            value_range (Optional[Tuple[int, int]]): The range of possible values the Hypervector can represent. Default is None, which sets the range to (0, size).

        Returns:
            tf.Tensor: The generated tensor representing the Hypervector.
        '''
        if value is None:
            value = self.size / 2

        if value_range is None:
            value_range = (0, self.size)

        min_value, max_value = value_range

        # tf_random GENERATION
        if random_method == "tf_random":
            num_ones = int(self.size * (value + min_value) / max_value)
            corr_vec = tf.constant([-1] * self.size)
            indices = tf.random.shuffle(tf.range(self.size))[:num_ones]
            updates = tf.ones(num_ones, dtype=tf.int32)

            tensor = tf.tensor_scatter_nd_update(corr_vec, tf.expand_dims(indices, 1), updates)
        
        # Sobol GENERATION
        elif random_method == "Sobol":
            global sobol_dim, sobol_indexer, sobol_seqs
    
            if sobol_seqs is None or sobol_dim < self.size:
                m = math.ceil(math.log(self.size, 2))
                sobol_dim = 2**m

                sobol_engine = qmc.Sobol(d=21_201, scramble=False)  # the maximum number of sequences that can be "generated" by this implementation is 21_201
                                                                    # TODO: uncap this maximum number of sequences

                sobol_samples = sobol_engine.random_base2(m=m)
                sobol_seqs = np.transpose(sobol_samples)
            
            s_val = ((value - min_value) / (max_value - min_value))
            sobol_indexer += 1

            tensor = tf.convert_to_tensor(_comp_each_to_val(sobol_seqs[sobol_indexer - 1], s_val)[:self.size])

        # TODO: handle this case better
        else:
            print("WARNING PLACEHOLDER")

            tensor = None

        return tensor
    
    def _generate_value(self) -> int:
        '''
        Generates the value of the a passed bitstream (Tensor).

        Returns:
            int: The generated value based on the number of 1's in the Tensor passed.
        '''
        return tf.reduce_sum(tf.cast(tf.equal(self.tensor, 1), tf.int32)).numpy()

    # TODO: def regen_tensor()

    def binarize(self):
        '''
        Converts the tensor to a binary representation where all positive values become 1 and non-positive values become 0.
        '''
        self.tensor = tf.where(self.tensor > 0, tf.ones_like(self.tensor), tf.zeros_like(self.tensor))

    def polarize(self):
        '''
        Converts the tensor to a polar representation where all non-positive values become -1 and positive values become 1.
        '''
        self.tensor = tf.where(self.tensor <= 0, tf.constant(-1, dtype=tf.int32), tf.ones_like(self.tensor))

# generates a Dictionary of "level" or "L" hypervectors for some integer value range
# TODO: docstrings
# TODO: non-integer support
# TODO: implement custom increment parameter (e.g., the value range is from 0-256 but in increments of 0.5)
def gen_L_HVs(hv_size: Optional[int]=256, value_range: Optional[Tuple[int, int]]=None):
    if value_range == None:
        value_range = (0, hv_size)

    min_val = int(value_range[0])
    max_val = int(value_range[1])

    L_HVs = {}
    L_HVs[min_val] = Hypervector(value=0, size=hv_size)

    num_flips_per = hv_size / (max_val - min_val)
    num_flips = num_flips_per

    for i in range((min_val + 1), max_val):
        neg_indices = tf.where(tf.equal(L_HVs[i - 1].tensor, -1))
        random_indices = tf.random.shuffle(neg_indices)[:int(num_flips)]
        next_tensor = tf.identity(L_HVs[i - 1].tensor)

        for idx in random_indices:
            index = idx[0]  # Extract the index from the shape tuple
            next_tensor = tf.tensor_scatter_nd_update(next_tensor, [[index]], [1])

        L_HVs[i] = Hypervector(tensor=next_tensor)

        '''
        if num_flips < 1:
            num_flips += num_flips_per
        else:
            num_flips = num_flips_per
        optimized to:'''
        num_flips = (num_flips * (num_flips < 1)) + num_flips_per

    L_HVs[max_val] = Hypervector(value=hv_size, size=hv_size)

    return L_HVs

# performs element-wise addition on two Hypervectors and returns the resulting Hypervector
def add(hv1, hv2):
    res_tensor = tf.add(hv1.tensor, hv2.tensor)
    return Hypervector(tensor=res_tensor)

# performs element-wise mulitplication on two Hypervectors and returns the resulting Hypervector
def multiply(hv1, hv2):
    res_tensor = tf.multiply(hv1.tensor, hv2.tensor)
    return Hypervector(tensor=res_tensor)

# returns [-1, 1], with 0 denoting complete orthogonality
def cos_similarity(hv1, hv2) -> float:
    '''
    Computes the cosine similarity between two Hypervectors.

    Args:
        hv1 (Hypervector): The first Hypervector.
        hv2 (Hypervector): The second Hypervector.

    Returns:
        float: The cosine similarity between hv1 and hv2, ranging from -1 to 1, where 0 denotes complete orthogonality.
    '''
    tensor1 = hv1.tensor
    tensor2 = hv2.tensor

    tensor1 = tf.cast(tensor1, tf.float32)
    tensor2 = tf.cast(tensor2, tf.float32)

    dot_product = tf.reduce_sum(tensor1 * tensor2)

    norm_tensor1 = tf.norm(tensor1)
    norm_tensor2 = tf.norm(tensor2)

    cos_sim = dot_product / (norm_tensor1 * norm_tensor2)

    return float(cos_sim)

