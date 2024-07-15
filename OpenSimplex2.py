import random
import sys


def _fastFloor(x) -> int:
    xi = int(x)
    return xi - 1 if x < xi else xi


class OpenSimplexNoise:
    PRIME_X = 0x5205402B9270C86F
    PRIME_Y = 0x598CD327003817B5
    HASH_MULTIPLIER = 0x53A3F72DEEC546F5
    N_GRADS_2D_EXPONENT = 2  # max 25
    N_GRADS_2D = 50 << N_GRADS_2D_EXPONENT
    UNSKEW_2D = -0.21132486540518713
    NORMALIZER_2D = 1  # 0.01001634121365712
    RSQUARED_2D = 0.5
    GRADIENTS_2D = []
    SKEW_2D = 0.366025403784439
    ROOT2OVER2 = 0.7071067811865476

    def __init__(self, seed=None):
        self._initGradients()
        self.seed = seed if seed is not None else random.randint(0, sys.maxsize)

    def _initGradients(self):
        grad2 = [
            0.38268343236509, 0.923879532511287,
            0.923879532511287, 0.38268343236509,
            0.923879532511287, -0.38268343236509,
            0.38268343236509, -0.923879532511287,
            -0.38268343236509, -0.923879532511287,
            -0.923879532511287, -0.38268343236509,
            -0.923879532511287, 0.38268343236509,
            -0.38268343236509, 0.923879532511287,
            # ----------------------------------- #
            0.130526192220052, 0.99144486137381,
            0.608761429008721, 0.793353340291235,
            0.793353340291235, 0.608761429008721,
            0.99144486137381, 0.130526192220051,
            0.99144486137381, -0.130526192220051,
            0.793353340291235, -0.60876142900872,
            0.608761429008721, -0.793353340291235,
            0.130526192220052, -0.99144486137381,
            -0.130526192220052, -0.99144486137381,
            -0.608761429008721, -0.793353340291235,
            -0.793353340291235, -0.608761429008721,
            -0.99144486137381, -0.130526192220052,
            -0.99144486137381, 0.130526192220051,
            -0.793353340291235, 0.608761429008721,
            -0.608761429008721, 0.793353340291235,
            -0.130526192220052, 0.99144486137381,
        ]

        for i in range(len(grad2)):
            grad2[i] = (grad2[i] / self.NORMALIZER_2D)

        len_gradients_2d = self.N_GRADS_2D * 2

        self.GRADIENTS_2D += grad2 * (len_gradients_2d // len(grad2))
        self.GRADIENTS_2D += grad2[0:(len_gradients_2d % len(grad2))]

        # j = 0
        # for i in range(self.N_GRADS_2D * 2):
        #     if j == len(grad2):
        #         j = 0
        #     self.GRADIENTS_2D.append(grad2[j])
        #     j += 1

        # for (int i = 0, j = 0; i < GRADIENTS_2D.length; i++, j++) {
        #     if (j == grad2.length) j = 0;
        #     GRADIENTS_2D[i] = grad2[j];
        # }

    def _grad(self, xsvp, ysvp, dx, dy) -> float:
        hash = self.seed ^ xsvp ^ ysvp
        hash *= self.HASH_MULTIPLIER
        hash ^= hash >> (64 - self.N_GRADS_2D_EXPONENT + 1)
        gi = hash & ((self.N_GRADS_2D - 1) << 1)
        return self.GRADIENTS_2D[gi | 0] * dx + self.GRADIENTS_2D[gi | 1] * dy

    def noise2_UnskewedBase(self, xs, ys) -> float:
        # Get base points and offsets.
        xsb = _fastFloor(xs)
        ysb = _fastFloor(ys)

        xi = xs - xsb
        yi = ys - ysb

        # Prime pre-multiplication for hash.
        ysbp = ysb * self.PRIME_Y
        xsbp = xsb * self.PRIME_X

        # Unskew.
        t = (xi + yi) * self.UNSKEW_2D
        dx0 = xi + t
        dy0 = yi + t

        # First vertex.
        value = 0
        a0 = self.RSQUARED_2D - dx0 * dx0 - dy0 * dy0
        if a0 > 0:
            value = (a0 * a0) * (a0 * a0) * self._grad(xsbp, ysbp, dx0, dy0)

        # Second vertex.
        a1 = (2 * (1 + 2 * self.UNSKEW_2D) * (1 / self.UNSKEW_2D + 2)) * t + (
                (-2 * (1 + 2 * self.UNSKEW_2D) * (1 + 2 * self.UNSKEW_2D)) + a0)
        if a1 > 0:
            dx1 = dx0 - (1 + 2 * self.UNSKEW_2D)
            dy1 = dy0 - (1 + 2 * self.UNSKEW_2D)
            value += (a1 * a1) * (a1 * a1) * self._grad(xsbp + self.PRIME_X, ysbp + self.PRIME_Y, dx1, dy1)

        # Third vertex.
        if dy0 > dx0:
            dx2 = dx0 - self.UNSKEW_2D
            dy2 = dy0 - (self.UNSKEW_2D + 1)
            a2 = self.RSQUARED_2D - dx2 * dx2 - dy2 * dy2
            if a2 > 0:
                value += (a2 * a2) * (a2 * a2) * self._grad(xsbp, ysbp + self.PRIME_Y, dx2, dy2)

        else:
            dx2 = dx0 - (self.UNSKEW_2D + 1)
            dy2 = dy0 - self.UNSKEW_2D
            a2 = self.RSQUARED_2D - dx2 * dx2 - dy2 * dy2
            if a2 > 0:
                value += (a2 * a2) * (a2 * a2) * self._grad(xsbp + self.PRIME_X, ysbp, dx2, dy2)

        return value

    def noise2(self, x, y) -> float:

        # Get points for A2* lattice
        s = self.SKEW_2D * (x + y)
        xs = x + s
        ys = y + s

        return self.noise2_UnskewedBase(xs, ys)

    def noise2_ImproveX(self, x, y) -> float:

        # Skew transform and rotation baked into one.
        xx = x * self.ROOT2OVER2
        yy = y * (self.ROOT2OVER2 * (1 + 2 * self.SKEW_2D))

        return self.noise2_UnskewedBase(yy + xx, yy - xx)
