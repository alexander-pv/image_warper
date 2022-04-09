import os
import sys

import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

import core
import img_loader

app = FastAPI()
transformer = core.ImgTransformer()
emoji_parser = img_loader.EmojipediaParser()


@app.get("/")
async def read_root() -> dict:
    return {'response': 200}


@app.post("/warp_test/{method}")
async def warp_test(method: str) -> StreamingResponse:
    img_path = os.path.join(os.getcwd(), 'src', 'tests', 'pics')
    wrapped_buf = core.dry_run(img_path, method, True)
    return StreamingResponse(wrapped_buf, media_type="image/jpeg")


@app.post("/warp_random_emoji/{method}")
async def warp_random(method: str) -> StreamingResponse or dict:
    img0 = emoji_parser.fetch_random()
    img1 = emoji_parser.fetch_random()
    method = method if method in ('cps', 'cas') else 'cas'
    if method == 'cps':
        warped = transformer.contour_points_sampling(img0, img1)
    elif method == 'cas':
        warped = transformer.contour_areas_stratification(img0, img1)
    buf = core.decode_img(warped)
    return StreamingResponse(buf, media_type="image/jpeg")


def main() -> None:
    uvicorn.run("serve_warper:app",
                host='0.0.0.0',
                port=10000,
                debug=True)


if __name__ == "__main__":
    sys.exit(main())
