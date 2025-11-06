from util.utils import get_som_labeled_img, get_caption_model_processor, get_yolo_model, check_ocr_box
import torch
from PIL import Image
import io
import base64
import binascii
import re
from typing import Dict
class Omniparser(object):
    def __init__(self, config: Dict):
        self.config = config
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.som_model = get_yolo_model(model_path=config['som_model_path'])
        self.caption_model_processor = get_caption_model_processor(model_name=config['caption_model_name'], model_name_or_path=config['caption_model_path'], device=device)
        print('Omniparser initialized!!!')

    def parse(self, image_base64: str):
        image = open_image_from_base64(image_base64)
        print('image size:', image.size)
        
        box_overlay_ratio = max(image.size) / 3200
        draw_bbox_config = {
            'text_scale': 0.8 * box_overlay_ratio,
            'text_thickness': max(int(2 * box_overlay_ratio), 1),
            'text_padding': max(int(3 * box_overlay_ratio), 1),
            'thickness': max(int(3 * box_overlay_ratio), 1),
        }

        (text, ocr_bbox), _ = check_ocr_box(image, display_img=False, output_bb_format='xyxy', easyocr_args={'text_threshold': 0.8}, use_paddleocr=False)
        dino_labled_img, label_coordinates, parsed_content_list = get_som_labeled_img(image, self.som_model, BOX_TRESHOLD = self.config['BOX_TRESHOLD'], output_coord_in_ratio=True, ocr_bbox=ocr_bbox,draw_bbox_config=draw_bbox_config, caption_model_processor=self.caption_model_processor, ocr_text=text,use_local_semantics=True, iou_threshold=0.7, scale_img=False, batch_size=128)

        return dino_labled_img, parsed_content_list

def _maybe_strip_data_url_prefix(s: str) -> str:
    if s.startswith('data:'):
        parts = s.split(',', 1)
        if len(parts) == 2:
            return parts[1]
    return s

def _is_base64_text(b: bytes) -> bool:
    # Heuristic: consider small prefix chars
    allowed = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=\r\n"
    sample = b[:128]
    return all(c in allowed for c in sample)

def _maybe_double_base64(b: bytes) -> bytes:
    try:
        if _is_base64_text(b):
            b2 = base64.b64decode(b, validate=True)
            # If second decode yields plausible image header, accept
            if b2.startswith(b"\x89PNG") or b2.startswith(b"\xff\xd8") or b2[:6] in (b"GIF87a", b"GIF89a"):
                return b2
    except Exception:
        pass
    return b

def _maybe_unicode_escaped_bytes(b: bytes) -> bytes:
    # Detect presence of unicode escape sequences like \u00.. or \x..
    if b.startswith(b"\\u00") or b.startswith(b"\\x") or b.find(b"\\u00") != -1:
        try:
            txt = b.decode('utf-8')
            # decode unicode escapes to actual characters, then map to bytes 0-255
            fixed = txt.encode('utf-8').decode('unicode_escape').encode('latin1')
            return fixed
        except Exception:
            return b
    return b

def open_image_from_base64(image_base64: str) -> Image.Image:
    s = _maybe_strip_data_url_prefix(image_base64)
    try:
        b = base64.b64decode(s, validate=True)
    except binascii.Error:
        b = base64.b64decode(s)
    # Handle double-encoded base64
    b = _maybe_double_base64(b)
    # Handle unicode-escaped payloads
    b = _maybe_unicode_escaped_bytes(b)
    return Image.open(io.BytesIO(b))