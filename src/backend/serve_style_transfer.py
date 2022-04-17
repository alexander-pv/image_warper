import dataclasses
import logging
import sys

import uvicorn
from fastapi import FastAPI
from fastapi import UploadFile
from fastapi.responses import StreamingResponse

import core
import numpy as np
from style_transfer import StyleTransfer

app = FastAPI()


@dataclasses.dataclass
class ServerSettings:
    host: str = '0.0.0.0'
    port: int = 15000
    media_type: str = "image/jpg"
    devices: tuple = ('cuda', 'cpu')
    pooling: str = 'max'
    loglevel: int = logging.DEBUG
    verbose: bool = True


@app.get("/")
def read_root() -> dict:
    return {'response': 200}


@app.post("/style_img/{steps}")
async def style_img(steps: int, file_content: UploadFile, file_style: UploadFile) -> StreamingResponse:
    content_image_bytes = await file_content.read()
    style_image_bytes = await file_style.read()
    content_image = core.decode_img(content_image_bytes)
    style_image = core.decode_img(style_image_bytes)

    logging.debug('Received images for style transfer')
    st_transfer.stylize(content_image=content_image, style_images=[style_image], iterations=steps)
    output = np.array(st_transfer.get_image())

    logging.debug('Received style-transfer result')
    io_buf = core.encode_img(output, 'jpg')
    logging.debug('Done encoding output')
    return StreamingResponse(content=io_buf, media_type=settings.media_type)


settings = ServerSettings()
st_transfer = StyleTransfer(devices=settings.devices, pooling=settings.pooling)


def main() -> None:
    if settings.verbose:
        logging.basicConfig(level=settings.loglevel,
                            format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S')
    uvicorn.run("serve_style_transfer:app",
                host=settings.host,
                port=settings.port,
                debug=settings.verbose)


if __name__ == "__main__":
    sys.exit(main())
