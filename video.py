import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.models as models
from models import TransformNet, VGG, MetaNet
from utils import mean_std

# Load pre-trained models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vgg16 = models.vgg16(pretrained=True)
vgg16 = VGG(vgg16.features[:23]).to(device).eval()
transform_net = TransformNet(base=16).to(device)
metanet = MetaNet(transform_net.get_param_dict()).to(device)

# Load pre-trained weights
metanet.load_state_dict(torch.load('sourse/model/metanet_base16_style50_tv1e-06_tagnohvd_5.pth'))
transform_net.load_state_dict(torch.load('sourse/model/metanet_base16_style50_tv1e-06_tagnohvd_transform_net_5.pth'))

# Load style image
style_image_path = 'static/style/style3/style3.jpg'
style_image = cv2.imread(style_image_path)
style_image = cv2.cvtColor(style_image, cv2.COLOR_BGR2RGB)
style_image = torch.tensor(style_image, dtype=torch.float32).permute(2, 0, 1) / 255.0
style_image = style_image.unsqueeze(0).to(device)

# Set up video capture
cap = cv2.VideoCapture(0)  # 0 for webcam, you can also pass a file path for a video file

# Define video writer
output_path = "output_video.mp4"
fourcc = cv2.VideoWriter_fourcc(*'XVID')
fps = 60
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess frame for the model
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = frame.transpose((2, 0, 1))  # HWC to CHW
    frame = torch.tensor(frame, dtype=torch.float32) / 255.0
    frame = frame.unsqueeze(0).to(device)

    # Apply style transfer
    with torch.no_grad():
        style_features = vgg16(style_image)
        style_mean_std = mean_std(style_features)
        weights = metanet.forward(style_mean_std)
        transform_net.set_weights(weights, 0)
        transformed_frame = transform_net(frame)

    # Postprocess transformed frame
    transformed_frame = transformed_frame.clamp(0, 1).squeeze(0).cpu().numpy()
    transformed_frame = (transformed_frame.transpose((1, 2, 0)) * 255).astype(np.uint8)
    transformed_frame = cv2.cvtColor(transformed_frame, cv2.COLOR_RGB2BGR)

    # Display the transformed frame
    cv2.imshow('Transformed Video', transformed_frame)
    out.write(transformed_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
