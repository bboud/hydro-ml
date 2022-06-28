import numpy as np
from numba import jit

@jit
def kernel(x, x0):
    sigma = 0.8
    protonFraction = 0.4
    norm = protonFraction / (np.sqrt(2. * np.pi) * sigma)
    return norm * np.exp(-(x - x0) ** 2. / (2. * sigma ** 2.))


@jit
def fake_kernel(x, x0, s):
    sigma = s
    protonFraction = 0.4
    norm = protonFraction / (np.sqrt(2. * np.pi) * sigma)
    return norm * np.exp(-(x - x0) ** 2. / (2. * sigma ** 2.))


@jit
def test_data_gen(fakekernel=False, sigma=0.4):
    A = 197
    yBeam = 5.36
    slope = 0.5
    sigmaEtas = 0.2

    # generate input data
    nBaryons = np.random.randint(0, 2 * A)
    randX = np.random.uniform(0, 1, size=nBaryons)
    etasBaryon = 1. / slope * np.arcsinh((2. * randX - 1) * np.sinh(slope * yBeam))
    etasArr = np.linspace(-6.4, 6.4, 128)
    dNBdetas = np.zeros(len(etasArr))
    norm = 1. / (np.sqrt(2. * np.pi) * sigmaEtas)
    for iB in etasBaryon:
        dNBdetas += norm * np.exp(-(etasArr - iB) ** 2. / (2. * sigmaEtas ** 2.))

    # generate test data with convolution with a kernel
    dNpdy = np.zeros(len(etasArr))
    detas = etasArr[1] - etasArr[0]
    for i in range(len(etasArr)):
        dNpdy[i] = sum(kernel(etasArr, etasArr[i]) * dNBdetas) * detas

    if fakekernel:
        # dNBdetasFake = np.random.uniform(0.0, dNBdetas.max(), size=len(etasArr))
        dNpdyFake = np.zeros(len(etasArr))
        detas = etasArr[1] - etasArr[0]
        for i in range(len(etasArr)):
            dNpdyFake[i] = sum(fake_kernel(etasArr, etasArr[i], sigma) * dNBdetas) * detas

        return etasArr, dNBdetas, dNpdy, dNBdetas, dNpdyFake
    else:
        # generate fake data with random noise
        dNBdetasFake = np.random.uniform(0.0, dNBdetas.max(), size=len(etasArr))
        dNpdyFake = np.random.uniform(0.0, dNpdy.max(), size=len(etasArr))

        return etasArr, dNBdetas, dNpdy, dNBdetasFake, dNpdyFake


def generate_data(size=128):
    real_data = []
    fake_data = []

    for iev in range(size):
        x, y1, y2, y3, y4 = test_data_gen()

        # real data - Block
        x = y2
        real_data.append(x)

        # fake data: random - Block
        x = y4
        fake_data.append(x)

    return np.array(real_data, dtype=np.float32), np.array(fake_data, dtype=np.float32)
