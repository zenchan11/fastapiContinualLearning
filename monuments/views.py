from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
from django.template import loader
from .forms import ImageUploadForm
from django.conf import settings
from datetime import datetime
from monuments.faster_models.fasterrcnn import fasterrcnn_resnet50_fpn, filter_pred, classes, CLASSES
import numpy as np
import os
import torch
import torchvision.transforms as transforms
from torchvision.transforms import functional as F
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import matplotlib.patches as patches




def monuments(request):
    return render(request, 'index.html')

def index(request):
    # return render(request, 'loader_template.html')
    return render(request, 'welcome.html')

def upload(request):
    if request.method == 'POST':
        form = ImageUploadForm(request.POST,request.FILES)
        if form.is_valid():
            form.save()
            img_object = form.instance
            print(img_object.image.url)
            return render(request, 'upload.html',{'form': form,'img_object':img_object})
    else:
        form = ImageUploadForm()
        return render(request, 'image_form.html', {'form': form})
    
def predict(request):
    if request.method == 'POST':
        image_path = request.POST.get('image_path')
        filename = os.path.basename(image_path)
        absolute_path = os.path.join('media', 'images', filename)
        print(absolute_path)
        if request.method == 'GET':
            model = request.GET.get('model')
            request.session['model'] = model

        if filename != None :
            # if request.session.get('model') == 'base_model':
                image = Image.open(absolute_path).convert("RGB")

                # Define transformations
                transform = transforms.Compose([
                    transforms.ToTensor(),
                ])
                img_tensor = transform(image)
                img_tensor = img_tensor.unsqueeze(0)
                print('code reached here')

                model_path = 'C:/Users/DELL/Downloads/Base Model/model2.pth'  # Adjust the path as necessary
                model = fasterrcnn_resnet50_fpn(num_classes = 16)
                model.load_state_dict(torch.load(model_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu')))
                model.to('cuda' if torch.cuda.is_available() else 'cpu')
                model.eval()
                

                with torch.no_grad():
                    predictions = model(img_tensor)

                outputs=filter_pred(predictions)
                boxes = outputs[0]['boxes'].cpu().numpy()
                labels = outputs[0]['labels'].cpu().numpy()
                scores = outputs[0]['scores'].cpu().numpy() 

                print(boxes,labels,scores)
                original_np = np.array(image)

                image_bgr = original_np
                # Original Image
                fig, axs = plt.subplots(figsize=(10, 5))
                axs.imshow(image_bgr)  # Assuming original images are in CHW format
                axs.axis('off')
                axs.set_title('Original')


                # Add predicted bounding boxes to the predicted image
                for j,box in enumerate(boxes):
                    rect = patches.Rectangle(
                        (box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=2, edgecolor='r', facecolor='none'
                    )
                    axs.add_patch(rect)
                    axs.text(
                        box[0], box[1] - 5,f'{CLASSES[int(labels[j])]}' , color='r', fontsize=10,
                        bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.2')
                    )

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                image_filename = f"file_{timestamp}.png"
                # Define the path to save the image within the predicted_image folder
                image_path =  os.path.join(settings.STATIC_URL,image_filename)
                absolute_path = os.path.join(settings.BASE_DIR,'static',image_filename)
                # absolute_path = os.path.join( image_path)   
                plt.savefig(absolute_path)
                print(absolute_path)
                print('successfully reached here yeah boy')
                print(image_path)

                return render(request, 'predict.html',{'image_path':image_path})
            # elif request.session.get('model') ==  'meta_learning':
            #     return HttpResponse('reaching to this message is not possible', status=403)
            # elif request.session.get('model') ==  'mnad':
            #     return HttpResponse('reaching to this message is not possible', status=403)
            # return JsonResponse({'success': True, 'image_path': image_path})
    
    # return render(request,'predict.html')
    return HttpResponse('reaching to this message is not possible', status=403)

