import dataclasses
import logging
import sys

import uvicorn
from fastapi import FastAPI
from fastapi import UploadFile
from fastapi.responses import StreamingResponse

import core
import img_loader

app = FastAPI()


@dataclasses.dataclass
class ServerSettings:
    host: str = '0.0.0.0'
    port: int = 10000
    media_type: str = "image/png"

    img_parser_caching: bool = True
    img_size: int = 120
    img_cache_limit: int = 20

    loglevel: int = logging.DEBUG
    verbose: bool = True


@app.get("/")
def read_root() -> dict:
    return {'response': 200}


@app.post("/fetch_random_img")
def fetch_random_img() -> StreamingResponse:
    img = emoji_parser.fetch_random()
    io_buf = core.encode_img(img)
    return StreamingResponse(content=io_buf, media_type=settings.media_type)


@app.post("/warp_imgs/{method}")
async def warp_imgs(method: str, file1: UploadFile, file2: UploadFile) -> StreamingResponse:
    image_bytes1 = await file1.read()
    image_bytes2 = await file2.read()
    img1 = core.decode_img(image_bytes1)
    img2 = core.decode_img(image_bytes2)
    method = method if method in ('cps', 'cas_v1', 'cas_v2') else 'cas_v2'
    logging.debug('Received initial images')
    if method == 'cps':
        warped, _, _ = transformer.contour_points_sampling(primary_img=img1, secondary_img=img2)
    elif method == 'cas_v1':
        warped, _, _ = transformer.contour_areas_stratification(primary_img=img1, secondary_img=img2, convex_hull=False)
    else:
        warped, _, _ = transformer.contour_areas_stratification(primary_img=img1, secondary_img=img2, convex_hull=True)
    logging.debug('Received warped image')
    io_buf = core.encode_img(warped)
    logging.debug('Done encoding result')
    return StreamingResponse(content=io_buf, media_type=settings.media_type)


settings = ServerSettings()
transformer = core.ImgTransformer(loglevel=settings.loglevel)
emoji_parser = img_loader.EmojipediaParser(
    img_size=settings.img_size, caching=settings.img_parser_caching,
    cache_limit=settings.img_cache_limit, verbose=settings.verbose, loglevel=settings.loglevel)


def main() -> None:
    if settings.verbose:
        logging.basicConfig(level=settings.loglevel,
                            format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S')
    uvicorn.run("serve_warper:app",
                host=settings.host,
                port=settings.port,
                debug=settings.verbose)
    emoji_parser.destroy()


if __name__ == "__main__":
    sys.exit(main())
