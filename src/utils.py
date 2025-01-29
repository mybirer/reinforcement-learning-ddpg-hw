
def hidden_init(layer):
    """ outputs the limits for the values in the hidden layer for initialisation"""
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)