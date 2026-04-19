import numpy as np


class ChannelCodec:
    """Rate-1/2 convolutional codec with block interleaving."""

    CONSTRAINT = 7
    POLY = (0o133, 0o171)
    N_STATES = 1 << (CONSTRAINT - 1)

    def __init__(self):
        self._next_state = np.zeros((self.N_STATES, 2), dtype=np.int32)
        self._out = np.zeros((self.N_STATES, 2, 2), dtype=np.uint8)
        mask = (1 << (self.CONSTRAINT - 1)) - 1
        for state in range(self.N_STATES):
            for bit in (0, 1):
                reg = (bit << (self.CONSTRAINT - 1)) | state
                self._next_state[state, bit] = reg >> 1
                self._out[state, bit, 0] = self._parity(reg & self.POLY[0])
                self._out[state, bit, 1] = self._parity(reg & self.POLY[1])
        self._mask = mask

    @staticmethod
    def _parity(x: int) -> int:
        return x.bit_count() & 1

    def encode(self, bits):
        bits = np.asarray(bits, dtype=np.uint8)
        tail = np.zeros(self.CONSTRAINT - 1, dtype=np.uint8)
        in_bits = np.concatenate([bits, tail])
        state = 0
        out = np.zeros(len(in_bits) * 2, dtype=np.uint8)
        o = 0
        for bit in in_bits:
            reg = (int(bit) << (self.CONSTRAINT - 1)) | state
            out[o] = self._parity(reg & self.POLY[0])
            out[o + 1] = self._parity(reg & self.POLY[1])
            o += 2
            state = (reg >> 1) & self._mask
        return out

    def viterbi_decode(self, coded_bits):
        coded_bits = np.asarray(coded_bits, dtype=np.uint8)
        n_pairs = len(coded_bits) // 2
        if n_pairs == 0:
            return np.zeros(0, dtype=np.uint8)
        coded_bits = coded_bits[: n_pairs * 2].reshape(n_pairs, 2)

        inf = 10**9
        metrics = np.full(self.N_STATES, inf, dtype=np.int32)
        metrics[0] = 0
        prev_state = np.zeros((n_pairs, self.N_STATES), dtype=np.int16)
        prev_bit = np.zeros((n_pairs, self.N_STATES), dtype=np.uint8)

        for t in range(n_pairs):
            r0, r1 = int(coded_bits[t, 0]), int(coded_bits[t, 1])
            new_metrics = np.full(self.N_STATES, inf, dtype=np.int32)
            for s in range(self.N_STATES):
                m = metrics[s]
                if m >= inf:
                    continue
                for b in (0, 1):
                    ns = self._next_state[s, b]
                    o0, o1 = int(self._out[s, b, 0]), int(self._out[s, b, 1])
                    dist = (o0 ^ r0) + (o1 ^ r1)
                    cand = m + dist
                    if cand < new_metrics[ns]:
                        new_metrics[ns] = cand
                        prev_state[t, ns] = s
                        prev_bit[t, ns] = b
            metrics = new_metrics

        state = 0
        decoded = np.zeros(n_pairs, dtype=np.uint8)
        for t in range(n_pairs - 1, -1, -1):
            decoded[t] = prev_bit[t, state]
            state = prev_state[t, state]

        if len(decoded) > (self.CONSTRAINT - 1):
            return decoded[: -(self.CONSTRAINT - 1)]
        return np.zeros(0, dtype=np.uint8)

    @staticmethod
    def interleave(bits, depth=32):
        bits = np.asarray(bits, dtype=np.uint8)
        depth = max(1, int(depth))
        n = len(bits)
        cols = int(np.ceil(n / depth))
        padded_len = depth * cols
        padded = np.pad(bits, (0, padded_len - n), mode="constant")
        matrix = padded.reshape(depth, cols)
        return matrix.T.flatten()[:n]

    @staticmethod
    def deinterleave(bits, depth=32):
        bits = np.asarray(bits, dtype=np.uint8)
        depth = max(1, int(depth))
        n = len(bits)
        cols = int(np.ceil(n / depth))
        padded_len = depth * cols
        padded = np.pad(bits, (0, padded_len - n), mode="constant")
        matrix = padded.reshape(cols, depth).T
        return matrix.flatten()[:n]
