import io
from PIL import Image, ImageOps
import numpy as np
from nexsstv.config import Config

class ImageProcessor:
    TARGET_RES = Config.TARGET_RES
    STRIPE_HEIGHT = Config.STRIPE_HEIGHT
    NUM_STRIPES = Config.NUM_STRIPES
    SCALE_FACTORS = (1.0, 0.75, 0.5, 0.375, 0.25)
    QUALITY_STEP = 3
    MIN_REDUCED_WIDTH = 64

    @staticmethod
    def encode_image(image_path, quality=30, max_bytes=None):
        """Resizes image to 800x600 and splits into independent WebP stripes."""
        img = Image.open(image_path)
        # Use 'fit' to fill 800x600 without stretching (crops if necessary)
        img = ImageOps.fit(img, ImageProcessor.TARGET_RES, Image.Resampling.LANCZOS)
        
        stripes = []
        for i in range(ImageProcessor.NUM_STRIPES):
            box = (0, i * ImageProcessor.STRIPE_HEIGHT, 800, (i + 1) * ImageProcessor.STRIPE_HEIGHT)
            stripe_img = img.crop(box)

            best = None
            for scale in ImageProcessor.SCALE_FACTORS:
                if scale < 1.0:
                    w = max(1, int(ImageProcessor.TARGET_RES[0] * scale))
                    if w < ImageProcessor.MIN_REDUCED_WIDTH:
                        continue
                    reduced = stripe_img.resize((w, ImageProcessor.STRIPE_HEIGHT), Image.Resampling.LANCZOS)
                    candidate_img = reduced.resize((ImageProcessor.TARGET_RES[0], ImageProcessor.STRIPE_HEIGHT), Image.Resampling.BILINEAR)
                else:
                    candidate_img = stripe_img

                # Keep perfectly clean stripes when lossless fits in the budget.
                if max_bytes is not None:
                    output = io.BytesIO()
                    candidate_img.save(output, format='WEBP', lossless=True, method=6)
                    lossless = output.getvalue()
                    best = lossless if best is None or len(lossless) < len(best) else best
                    if len(lossless) <= max_bytes:
                        stripes.append(lossless)
                        break

                q = int(quality)
                while q >= 1:
                    output = io.BytesIO()
                    candidate_img.save(output, format='WEBP', quality=q, method=6)
                    data = output.getvalue()
                    best = data if best is None or len(data) < len(best) else best
                    if max_bytes is None or len(data) <= max_bytes:
                        stripes.append(data)
                        break
                    q -= ImageProcessor.QUALITY_STEP
                else:
                    continue
                break
            else:
                arr = np.array(stripe_img).reshape(-1, 3)
                avg = tuple(np.mean(arr, axis=0).astype(np.uint8).tolist())
                flat = Image.new('RGB', (ImageProcessor.TARGET_RES[0], ImageProcessor.STRIPE_HEIGHT), avg)
                output = io.BytesIO()
                flat.save(output, format='WEBP', quality=1, method=6)
                data = output.getvalue()
                if best is None or len(data) <= len(best):
                    stripes.append(data)
                else:
                    stripes.append(best)
            
        return stripes

    @staticmethod
    def decode_stripe(data):
        """Decodes a single WebP stripe."""
        try:
            return Image.open(io.BytesIO(data))
        except:
            return None

    @staticmethod
    def merge_stripes(stripe_dict):
        """Merges received stripes into a final 800x600 image."""
        final_img = Image.new('RGB', ImageProcessor.TARGET_RES, color=(0, 0, 0))
        for i in range(ImageProcessor.NUM_STRIPES):
            if i in stripe_dict and stripe_dict[i] is not None:
                final_img.paste(stripe_dict[i], (0, i * ImageProcessor.STRIPE_HEIGHT))
        return final_img
