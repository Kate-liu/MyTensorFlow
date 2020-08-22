# encoding: utf-8

import tensorflow as tf

# 0阶张量
mammal = tf.Variable("Elephant", tf.string)
ignition = tf.Variable(451, tf.int16)
floating = tf.Variable(3.14159265359, tf.float64)
its_complicated = tf.Variable(12.3 - 4.85j, tf.complex64)

print([mammal, ignition, floating, its_complicated])
# [
# <tf.Variable 'Variable:0' shape=() dtype=string_ref>,
# <tf.Variable 'Variable_1:0' shape=() dtype=int32_ref>,
# <tf.Variable 'Variable_2:0' shape=() dtype=float32_ref>,
# <tf.Variable 'Variable_3:0' shape=() dtype=complex128_ref>
# ]


# 1阶张量
mystr = tf.Variable(["Hello", "World"], tf.string)
cool_numbers = tf.Variable([3.14159, 2.71828], tf.float32)
first_primes = tf.Variable([2, 3, 5, 7, 11], tf.int32)
its_very_complicated = tf.Variable([12.3 - 4.85j, 7.5 - 6.23j], tf.complex64)

print([mystr, cool_numbers, first_primes, its_very_complicated])
# [
# <tf.Variable 'Variable_4:0' shape=(2,) dtype=string_ref>,
# <tf.Variable 'Variable_5:0' shape=(2,) dtype=float32_ref>,
# <tf.Variable 'Variable_6:0' shape=(5,) dtype=int32_ref>,
# <tf.Variable 'Variable_7:0' shape=(2,) dtype=complex128_ref>
# ]


# 2阶张量
mymat = tf.Variable([[7], [11]], tf.int16)
myxor = tf.Variable([[False, True], [True, False]], tf.bool)
linear_squares = tf.Variable([[4], [9], [16], [25]], tf.int32)
squarish_squares = tf.Variable([[4, 9], [16, 25]], tf.int32)
rank_of_squares = tf.rank(squarish_squares)
mymatC = tf.Variable([[7], [11]], tf.int32)

print([mymat, myxor, linear_squares, squarish_squares, rank_of_squares, mymatC])
# [
# <tf.Variable 'Variable_8:0' shape=(2, 1) dtype=int32_ref>,
# <tf.Variable 'Variable_9:0' shape=(2, 2) dtype=bool_ref>,
# <tf.Variable 'Variable_10:0' shape=(4, 1) dtype=int32_ref>,
# <tf.Variable 'Variable_11:0' shape=(2, 2) dtype=int32_ref>,
# <tf.Tensor 'Rank:0' shape=() dtype=int32>,
# <tf.Variable 'Variable_12:0' shape=(2, 1) dtype=int32_ref>
# ]


# 4阶张量
my_image = tf.zeros([10, 299, 299, 3])  # batch x height x width x color

print(my_image)
# Tensor("zeros:0", shape=(10, 299, 299, 3), dtype=float32)
