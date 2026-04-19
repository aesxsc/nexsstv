import io
from PIL import Image, ImageOps
import numpy as np

class ImageProcessor:
    TARGET_RES = (800, 600)
    STRIPE_HEIGHT = 8  # 600 / 75 = 8
    NUM_STRIPES = 75

    @staticmethod
    def encode_image(image_path, quality=30):
        """Resizes image to 800x600 and splits into independent WebP stripes."""
        img = Image.open(image_path)
        # Use 'fit' to fill 800x600 without stretching (crops if necessary)
        img = ImageOps.fit(img, ImageProcessor.TARGET_RES, Image.Resampling.LANCZOS)
        
        stripes = []
        for i in range(ImageProcessor.NUM_STRIPES):
            box = (0, i * ImageProcessor.STRIPE_HEIGHT, 800, (i + 1) * ImageProcessor.STRIPE_HEIGHT)
            stripe_img = img.crop(box)
            
            output = io.BytesIO()
            # Independent WebP per stripe
            stripe_img.save(output, format='WEBP', quality=quality)
            stripes.append(output.getvalue())
            
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
