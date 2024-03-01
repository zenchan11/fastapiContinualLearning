from fastapi import FastAPI, Request, Form, UploadFile
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pathlib import Path
from monuments.faster_models.fasterrcnn import fasterrcnn_resnet50_fpn, filter_pred, classes, CLASSES
import os
import shutil
import uvicorn
from datetime import datetime
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/media", StaticFiles(directory="media"), name="media")

templates = Jinja2Templates(directory="templates")


class ImageForm(BaseModel):
    image: UploadFile


@app.get("/")
async def monuments(request: Request):
    return templates.TemplateResponse("welcome.html", {"request": request})


@app.get("/index", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# @app.post("/upload")
# async def upload(request: Request, image: UploadFile = File(...)):
#     print('reached')
#     if image:
#         image_path = f"media/images/{image.filename}"
#         print(image_path)
#         with open(image_path, "wb") as image_file:
#             image_file.write(image.file.read())

#         return templates.TemplateResponse("upload.html", {"request": request, "form": image, "img_object": image_path})
    
#     return templates.TemplateResponse("image_form.html", {"request": request, "form": image})
@app.post("/upload")
async def upload(request: Request,file: UploadFile):
    upload_dir =  Path("media") / "images"
    # Create the upload directory if it doesn't exist
    upload_dir.mkdir(parents=True, exist_ok=True)

    dest = upload_dir / file.filename
    img_name = file.filename
    print(dest)
    # copy the file contents
    with open(dest, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return templates.TemplateResponse("upload.html", {"request":request, "img_name": img_name })

@app.post("/predict")
async def predict(request: Request, image_path: str = Form(...), model: str = "base_model"):
    filename = image_path
    absolute_path = os.path.join("media", "images", filename)
    print(absolute_path)
    if model == "base_model":
        image = Image.open(absolute_path).convert("RGB")

        # Define transformations
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        img_tensor = transform(image)
        img_tensor = img_tensor.unsqueeze(0)

        model_path = 'C:/Users/DELL/Downloads/Base Model/model2.pth'  # Adjust the path as necessary
        model = fasterrcnn_resnet50_fpn(num_classes=16)
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu')))
        model.to('cuda' if torch.cuda.is_available() else 'cpu')
        model.eval()

        with torch.no_grad():
            predictions = model(img_tensor)

        outputs = filter_pred(predictions)
        boxes = outputs[0]['boxes'].cpu().numpy()
        labels = outputs[0]['labels'].cpu().numpy()
        scores = outputs[0]['scores'].cpu().numpy()

        original_np = np.array(image)

        # Original Image
        fig, axs = plt.subplots(figsize=(10, 5))
        axs.imshow(original_np)  # Assuming original images are in CHW format
        axs.axis('off')
        axs.set_title('Original')

        # Add predicted bounding boxes to the predicted image
        for j, box in enumerate(boxes):
            rect = patches.Rectangle(
                (box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=2, edgecolor='r', facecolor='none'
            )
            axs.add_patch(rect)
            axs.text(
                box[0], box[1] - 5, f'{CLASSES[int(labels[j])]}' , color='r', fontsize=10,
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.2')
            )

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_filename = f"file_{timestamp}.png"
        image_path = os.path.join("static", image_filename)
        absolute_path = os.path.join("static", image_filename)
        plt.savefig(absolute_path)

        return templates.TemplateResponse("predict.html", {"request": request, "image_path": image_path})
    return {"detail": "Invalid model specified."}

if __name__ == '__main__':
    uvicorn.run("main:app", reload=True)
