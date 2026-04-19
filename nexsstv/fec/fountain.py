import struct
import zlib

class StripeFramer:
    # [Sync Barker 13] [Stripe ID 1B] [Length 2B] [Data] [CRC16 2B]
    HEADER_STRUCT = ">BH" # ID, Length

    @staticmethod
    def pack_stripe(stripe_id, data):
        """Wraps stripe data with a header and CRC."""
        header = struct.pack(StripeFramer.HEADER_STRUCT, stripe_id, len(data))
        crc = zlib.crc32(header + data) & 0xFFFF
        return header + data + struct.pack(">H", crc)

    @staticmethod
    def unpack_stripe(packet):
        """Unpacks stripe data and verifies CRC."""
        if len(packet) < 5: return None, None
        
        stripe_id, length = struct.unpack(StripeFramer.HEADER_STRUCT, packet[:3])
        if len(packet) < 3 + length + 2: return None, None
        
        data = packet[3 : 3 + length]
        received_crc = struct.unpack(">H", packet[3 + length : 3 + length + 2])[0]
        
        if zlib.crc32(packet[: 3 + length]) & 0xFFFF == received_crc:
            return stripe_id, data
        return None, None
