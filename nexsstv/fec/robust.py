import numpy as np


class ChannelCodec:
    """Punctured convolutional codec with block interleaving."""

    CONSTRAINT = 7
    POLY = (0o133, 0o171)
    N_STATES = 1 << (CONSTRAINT - 1)
    PUNCTURE_PATTERNS = {
        "1/2": np.array([1, 1], dtype=np.uint8),
        "2/3": np.array([1, 1, 1, 0], dtype=np.uint8),
        "3/4": np.array([1, 1, 0, 1, 1, 0], dtype=np.uint8),
    }

    def __init__(self, rate="2/3"):
        if rate not in self.PUNCTURE_PATTERNS:
            raise ValueError(f"Unsupported coding rate: {rate}")
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
        self.rate = rate
        self.puncture = self.PUNCTURE_PATTERNS[rate]

    @staticmethod
    def _parity(x: int) -> int:
        return x.bit_count() & 1

    def _puncture_bits(self, mother_bits):
        pattern = np.resize(self.puncture, len(mother_bits))
        return mother_bits[pattern == 1]

    def _depuncture_bits(self, coded_bits):
        coded_bits = np.asarray(coded_bits, dtype=np.uint8)
        received_idx = 0
        mother = []
        valid = []
        step = 0
        plen = len(self.puncture)

        while received_idx < len(coded_bits):
            keep = self.puncture[step % plen]
            if keep:
                mother.append(coded_bits[received_idx])
                valid.append(1)
                received_idx += 1
            else:
                mother.append(0)
                valid.append(0)
            step += 1

        if len(mother) % 2 != 0:
            keep = self.puncture[step % plen]
            mother.append(0)
            valid.append(1 if keep else 0)

        return np.array(mother, dtype=np.uint8), np.array(valid, dtype=np.uint8)

    def encoded_length(self, n_input_bits):
        n_input_bits = int(max(0, n_input_bits))
        n_total = n_input_bits + (self.CONSTRAINT - 1)
        mother_len = n_total * 2
        pattern = np.resize(self.puncture, mother_len)
        return int(np.sum(pattern))

    def max_input_bits_for_capacity(self, capacity_bits):
        capacity_bits = int(max(0, capacity_bits))
        lo, hi = 0, capacity_bits
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self.encoded_length(mid) <= capacity_bits:
                lo = mid
            else:
                hi = mid - 1
        return lo

    def encode(self, bits):
        bits = np.asarray(bits, dtype=np.uint8)
        tail = np.zeros(self.CONSTRAINT - 1, dtype=np.uint8)
        in_bits = np.concatenate([bits, tail])
        state = 0
        mother = np.zeros(len(in_bits) * 2, dtype=np.uint8)
        o = 0
        for bit in in_bits:
            reg = (int(bit) << (self.CONSTRAINT - 1)) | state
            mother[o] = self._parity(reg & self.POLY[0])
            mother[o + 1] = self._parity(reg & self.POLY[1])
            o += 2
            state = (reg >> 1) & self._mask
        return self._puncture_bits(mother)

    def viterbi_decode(self, coded_bits):
        mother_bits, valid = self._depuncture_bits(coded_bits)
        n_pairs = len(mother_bits) // 2
        if n_pairs == 0:
            return np.zeros(0, dtype=np.uint8)
        coded_bits = mother_bits[: n_pairs * 2].reshape(n_pairs, 2)
        valid = valid[: n_pairs * 2].reshape(n_pairs, 2)

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
                    dist = (valid[t, 0] * (o0 ^ r0)) + (valid[t, 1] * (o1 ^ r1))
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
