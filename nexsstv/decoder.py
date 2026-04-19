import sys
import os

# Path resolution
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from scipy.io import wavfile
import argparse
from nexsstv.config import Config
from nexsstv.image.progressive import ImageProcessor
from nexsstv.fec.fountain import StripeFramer
from nexsstv.fec.robust import ChannelCodec
from nexsstv.modem.ofdm import Modem
from nexsstv.sync.preamble import Preamble

def main():
    parser = argparse.ArgumentParser(description="NexSSTV NextGen Decoder")
    parser.add_argument("input_audio", help="Path to input audio")
    parser.add_argument("output_image", help="Path to output image")
    parser.add_argument("--mode", choices=["normal", "ultra", "classic"], default="normal")
    args = parser.parse_args()

    fs, audio = wavfile.read(args.input_audio)
    audio = audio.astype(np.float32) / 32767.0
    
    params = Config.get_mode_params(args.mode)
    modem = Modem(Config.FS, params['f_min'], params['f_max'], params['n_subcarriers'], Config.CP_RATIO)
    codec = ChannelCodec()
    preamble = Preamble(samples_per_bit=16)
    sync_template = preamble.generate_signal(Config.FS, f_center=(params['f_min'] + params['f_max']) / 2)
    
    received_stripes = {}
    dropped_packets = 0
    symbol_len = modem.fft_size + int(modem.fft_size * modem.cp_ratio)
    
    print(f"Decoding {args.mode} nextgen stream...")
    
    ptr = 0
    while ptr < len(audio) - len(sync_template):
        # 1. Slide and search for Barker pulse
        search_range = min(len(audio) - ptr, Config.FS) 
        idx, confidence = Preamble.detect(audio[ptr : ptr + search_range], sync_template)
        
        if confidence > 0.4:
            ptr += idx + len(sync_template)
            
            # 2. Pilot Reset
            if ptr + symbol_len > len(audio): break
            modem.demodulate_symbol(audio[ptr : ptr + symbol_len], is_pilot=True)
            ptr += symbol_len
            
            # 3. Read Fixed Payload
            bits_acc = []
            for _ in range(Config.SYMBOLS_PER_STRIPE):
                if ptr + symbol_len > len(audio): break
                symbol_data = audio[ptr : ptr + symbol_len]
                ptr += symbol_len
                diff_symbols = modem.demodulate_symbol(symbol_data)
                bits_acc.extend(modem.dqpsk_demap(diff_symbols))
                
            # 4. Unpack Stripe
            if len(bits_acc) >= 8:
                coded_bits = np.array(bits_acc, dtype=np.uint8)
                deinterleaved = codec.deinterleave(coded_bits, depth=Config.INTERLEAVER_DEPTH)
                decoded_bits = codec.viterbi_decode(deinterleaved)
                usable = (len(decoded_bits) // 8) * 8
                if usable == 0:
                    # Not enough recovered bits to reconstruct even one byte.
                    dropped_packets += 1
                    continue
                byte_data = np.packbits(decoded_bits[:usable]).tobytes()
                stripe_id, data = StripeFramer.unpack_stripe(byte_data)
                
                if stripe_id is not None:
                    if stripe_id not in received_stripes:
                        stripe_img = ImageProcessor.decode_stripe(data)
                        if stripe_img:
                            received_stripes[stripe_id] = stripe_img
                            if len(received_stripes) % 15 == 0 or len(received_stripes) == 1:
                                print(f"   Decoded {len(received_stripes)}/{Config.NUM_STRIPES} stripes...")
                        else:
                            dropped_packets += 1
                else:
                    dropped_packets += 1
        else:
            ptr += 512 # Skip

    if received_stripes:
        print(f"Finalizing ({len(received_stripes)} stripes found)...")
        final_img = ImageProcessor.merge_stripes(received_stripes)
        final_img.save(args.output_image)
        print(f"Success! Image saved to {args.output_image}")
        if dropped_packets:
            print(f"Note: {dropped_packets} packet(s) failed integrity checks and were discarded.")
    else:
        print("Failed to decode any stripes. Verify signal quality.")

if __name__ == "__main__":
    main()
