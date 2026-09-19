## Unit Testing

### Sigmoid

The sigmoid fuction has been tested three different ways:

1. Comparisson againt known values. Inputs 0, 1, and -1 are used, for which expected outputs are:

    * sigmoid(0) = 0.5,
    * sigmoid(1) ≈ 0.7311,
    * sigmoid(-1) ≈ 0.2689,

    The results are compared with the help of numpy.testing.assert_allclose function, which allows for small flating-point differences.


HAVE NOT FINISHED YET
