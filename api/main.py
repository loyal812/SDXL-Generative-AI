import os
import sys

# Get the current script's directory
current_script_directory = os.path.dirname(os.path.abspath(__file__))

# Get the project root path
project_root = os.path.abspath(os.path.join(current_script_directory, os.pardir))

sys.path.append(project_root)
sys.path.append(current_script_directory)

from fastapi import Depends, FastAPI, Response
from scripts.txt2img import txt2img
from models.txt2img_model import Txt2ImgRequest
from starlette.responses import RedirectResponse
from starlette.status import HTTP_201_CREATED

app = FastAPI(swagger_ui_parameters={"tryItOutEnabled": True})


@app.get("/")
async def root():
    return RedirectResponse(app.docs_url)


@app.post("/txt2img", status_code=HTTP_201_CREATED)
async def t2i(request_body: Txt2ImgRequest):
    result = txt2img(request_body)
    result.save("output.png")

    # Return the image file as the response content
    with open("output.png", "rb") as f:
        file_content = f.read()

    return Response(content=file_content, media_type="image/png")
