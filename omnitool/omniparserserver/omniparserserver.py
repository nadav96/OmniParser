'''
python -m omniparserserver --som_model_path ../../weights/icon_detect/model.pt --caption_model_name florence2 --caption_model_path ../../weights/icon_caption_florence --device cuda --BOX_TRESHOLD 0.05
'''

import sys
import os
import time
from fastapi import FastAPI, Request
from pydantic import BaseModel
import logging
import argparse
import uvicorn
import base64
import io
import uuid
from PIL import Image
from fastapi.middleware.cors import CORSMiddleware
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_dir)
from util.omniparser import Omniparser

def parse_arguments():
    parser = argparse.ArgumentParser(description='Omniparser API')
    parser.add_argument('--som_model_path', type=str, default='../../weights/icon_detect/model.pt', help='Path to the som model')
    parser.add_argument('--caption_model_name', type=str, default='florence2', help='Name of the caption model')
    parser.add_argument('--caption_model_path', type=str, default='../../weights/icon_caption_florence', help='Path to the caption model')
    parser.add_argument('--device', type=str, default='cpu', help='Device to run the model')
    parser.add_argument('--BOX_TRESHOLD', type=float, default=0.05, help='Threshold for box detection')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host for the API')
    parser.add_argument('--port', type=int, default=8000, help='Port for the API')
    args = parser.parse_args()
    return args

args = parse_arguments()
config = vars(args)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logger = logging.getLogger("omniparser")
uploads_dir = os.path.join(root_dir, "uploads")
os.makedirs(uploads_dir, exist_ok=True)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    t0 = time.perf_counter()
    body = await request.body()
    logger.debug(
        ">> %s %s  qs=%s  hdr=%s  body=%s",
        request.method, request.url.path, request.url.query, dict(request.headers), body[:200]
    )
    response = await call_next(request)
    dt = (time.perf_counter() - t0) * 1000
    logger.debug(
        "<< %s %s  status=%d  %.1f ms",
        request.method, request.url.path, response.status_code, dt
    )
    return response

omniparser = Omniparser(config)

class ParseRequest(BaseModel):
    base64_image: str

@app.post("/parse")
@app.post("/parse/")
async def parse(parse_request: ParseRequest):
    print('start parsing...')
    start = time.time()
    try:
        image_bytes = base64.b64decode(parse_request.base64_image)
        img = Image.open(io.BytesIO(image_bytes))
        ext = (img.format or "PNG").lower()
        filename = f"{int(time.time()*1000)}_{uuid.uuid4().hex[:8]}.{ext}"
        save_path = os.path.join(uploads_dir, filename)
        img.save(save_path)
        logger.info("Saved input image to %s", save_path)
    except Exception as e:
        logger.warning("Failed to save input image: %s", e)
    dino_labled_img, parsed_content_list = omniparser.parse(parse_request.base64_image)
    latency = time.time() - start
    print('time:', latency)
    return {"som_image_base64": dino_labled_img, "parsed_content_list": parsed_content_list, 'latency': latency}

@app.get("/probe/")
async def root():
    return {"message": "Omniparser API ready"}

if __name__ == "__main__":

    # Enable full DEBUG logging
    logging.basicConfig(level=logging.DEBUG)
    
    uvicorn.run("omniparserserver:app", host=args.host, port=args.port, log_level="debug", reload=True)