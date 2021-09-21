from collections import namedtuple


RCNNFeatures = namedtuple("RCNNFeatures", ["global_features",
                                           "boxes_features",
                                           "boxes",
                                           "labels",
                                           "scores"])
