import sys

import uvicorn
from fastapi import FastAPI
from fastapi import UploadFile
from fastapi.responses import StreamingResponse

import core
import img_loader

app = FastAPI()
transformer = core.ImgTransformer()
emoji_parser = img_loader.EmojipediaParser()


@app.get("/")
async def read_root() -> dict:
    return {'response': 200}


@app.post("/fetch_random_img")
async def fetch_random_img() -> StreamingResponse:
    img = emoji_parser.fetch_random()
    io_buf = core.encode_img(img)
    return StreamingResponse(content=io_buf, media_type="image/png")


@app.post("/warp_imgs/{method}")
async def warp_imgs(method: str, file1: UploadFile, file2: UploadFile) -> StreamingResponse or dict:
    image_bytes1 = await file1.read()
    image_bytes2 = await file2.read()
    img1 = core.decode_img(image_bytes1)
    img2 = core.decode_img(image_bytes2)
    method = method if method in ('cps', 'cas') else 'cas'
    if method == 'cps':
        warped = transformer.contour_points_sampling(img1, img2)
    elif method == 'cas':
        warped = transformer.contour_areas_stratification(img1, img2)
    io_buf = core.encode_img(warped)
    return StreamingResponse(content=io_buf, media_type="image/png")


def main() -> None:
    uvicorn.run("serve_warper:app",
                host='0.0.0.0',
                port=10000,
                debug=True)


if __name__ == "__main__":
    sys.exit(main())
