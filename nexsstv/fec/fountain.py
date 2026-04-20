import struct
import zlib


class StripeFramer:
    """
    Packet format:
      [magic 2B][version 1B][stripe_id 1B][repeat_idx 1B][total_stripes 1B]
      [len 2B][payload N][crc32 4B]
    """

    MAGIC = b"NX"
    VERSION = 1
    HEADER_STRUCT = ">2sBBBBH"
    CRC_STRUCT = ">I"

    @staticmethod
    def pack_stripe(stripe_id, data, repeat_idx=0, total_stripes=50):
        header = struct.pack(
            StripeFramer.HEADER_STRUCT,
            StripeFramer.MAGIC,
            StripeFramer.VERSION,
            int(stripe_id) & 0xFF,
            int(repeat_idx) & 0xFF,
            int(total_stripes) & 0xFF,
            len(data),
        )
        crc = zlib.crc32(header + data) & 0xFFFFFFFF
        return header + data + struct.pack(StripeFramer.CRC_STRUCT, crc)

    @staticmethod
    def unpack_stripe(packet):
        min_len = struct.calcsize(StripeFramer.HEADER_STRUCT) + struct.calcsize(StripeFramer.CRC_STRUCT)
        if len(packet) < min_len:
            return None

        try:
            magic, version, stripe_id, repeat_idx, total_stripes, length = struct.unpack(
                StripeFramer.HEADER_STRUCT,
                packet[: struct.calcsize(StripeFramer.HEADER_STRUCT)],
            )
        except struct.error:
            return None

        if magic != StripeFramer.MAGIC or version != StripeFramer.VERSION:
            return None

        payload_end = struct.calcsize(StripeFramer.HEADER_STRUCT) + length
        crc_end = payload_end + struct.calcsize(StripeFramer.CRC_STRUCT)
        if crc_end > len(packet):
            return None

        payload = packet[struct.calcsize(StripeFramer.HEADER_STRUCT) : payload_end]
        received_crc = struct.unpack(StripeFramer.CRC_STRUCT, packet[payload_end:crc_end])[0]
        expected = zlib.crc32(packet[:payload_end]) & 0xFFFFFFFF
        if received_crc != expected:
            return None

        return {
            "stripe_id": int(stripe_id),
            "repeat_idx": int(repeat_idx),
            "total_stripes": int(total_stripes),
            "payload": payload,
        }


class RedundancyPlanner:
    @staticmethod
    def build_schedule(num_stripes, redundancy_factor=1.3):
        if num_stripes <= 0:
            return []

        total_frames = int(round(num_stripes * float(redundancy_factor)))
        total_frames = max(num_stripes, total_frames)

        # First pass guarantees baseline progressive decode.
        schedule = list(range(num_stripes))

        # Extra passes prioritize center stripes less than edges; this helps
        # speaker/mic multipath because each stripe gets temporal diversity.
        extras = total_frames - num_stripes
        if extras <= 0:
            return [(sid, 0) for sid in schedule]

        ring = []
        left, right = 0, num_stripes - 1
        while left <= right:
            ring.append(left)
            if left != right:
                ring.append(right)
            left += 1
            right -= 1

        repeats = {i: 0 for i in range(num_stripes)}
        out = [(sid, 0) for sid in schedule]
        for i in range(extras):
            sid = ring[i % len(ring)]
            repeats[sid] += 1
            out.append((sid, repeats[sid]))

        return out
