def initialize(ndim):
    global N_DIM
    global MY_EPS
    global L2_GRAD_CLIP
    N_DIM=ndim
    # MY_EPS=1e-6
    MY_EPS=1e-8
    # L2_GRAD_CLIP=1e-2
    L2_GRAD_CLIP=1e-4
