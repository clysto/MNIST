import torch
import numpy as np
import re
from typing import Optional
from fastapi import FastAPI, Request
from main.net import Net
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import base64
from io import BytesIO
from PIL import Image

model = Net()
model.load_state_dict(torch.load("data/model.pt"))
model.eval()

app = FastAPI()


app.mount("/static", StaticFiles(directory="app/static"), name="static")

templates = Jinja2Templates(directory="app/templates")


class ImageUrl(BaseModel):
    data_url: str


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/recognize")
async def recognize(
    image_url: Optional[ImageUrl] = None,
):
    base64_data = re.sub("^data:image/.+;base64,", "", image_url.data_url)
    binary_data = base64.b64decode(base64_data)
    file_jpgdata = BytesIO(binary_data)
    im = Image.open(file_jpgdata).convert("L")
    im = im.resize((28, 28))
    x = np.array(im)
    y = model(torch.tensor([x]).reshape((-1, 28 * 28)).float())
    r = (torch.argmax(y, dim=1)).item()
    return r
