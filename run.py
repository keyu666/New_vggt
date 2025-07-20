import os
import torch
import numpy as np
import plotly.graph_objs as go
import gradio as gr
from PIL import Image
import torchvision.transforms as T
import os, sys
import traceback


# Redirect Gradio temp uploads to a local writable directory
HERE = os.path.dirname(__file__)
GRADIO_TMP = os.path.join(HERE, "tmp_gradio")
os.makedirs(GRADIO_TMP, exist_ok=True)
os.environ["GRADIO_TEMP_DIR"] = GRADIO_TMP

# 把项目根目录下的 src 加到导入路径里
SRC = os.path.join(HERE, "src")
sys.path.insert(0, SRC)

# 之后再正常 import
from streamvggt.models.streamvggt import StreamVGGT
# --- Configuration ---
WEIGHTS_PATH = "/home/kf239/streamVGGT/ckpt/checkpoints.pth"  # 修改为实际权重路径
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 518
PATCH_SIZE = 14
EMBED_DIM = 1024

# --- Load model ---
model = StreamVGGT(img_size=IMG_SIZE, patch_size=PATCH_SIZE, embed_dim=EMBED_DIM).to(device)
ckpt = torch.load(WEIGHTS_PATH, map_location=device)
model.load_state_dict(ckpt)
model.eval()

# --- Preprocess transform ---
transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
])

def inference_cloud(image_files):
    """
    接收PIL图像列表，返回Plotly 3D散点图对象
    """
    print(f"[inference_cloud] Received files: {image_files}")
    if not image_files:
        print("[inference_cloud] No files uploaded.")
        return gr.Markdown("**Error during inference:** No files were uploaded.")
    try:
        frames = []
        for file in image_files:
            img = Image.open(file).convert("RGB")
            img_tensor = transform(img).unsqueeze(0).to(device)
            K = torch.eye(3, device=device)
            K[0,2] = IMG_SIZE/2
            K[1,2] = IMG_SIZE/2
            frames.append({"img": img_tensor, "camera_intrinsics": K.unsqueeze(0)})
        with torch.no_grad():
            output = model.inference(frames)
        pts = output.ress[-1]["pts3d_in_other_view"].cpu().numpy()[0]
        H, W, _ = pts.shape
        pts_flat = pts.reshape(-1, 3)
        scatter = go.Scatter3d(
            x=pts_flat[:,0], y=pts_flat[:,1], z=pts_flat[:,2],
            mode='markers', marker=dict(size=1, color=pts_flat[:,2], colorscale='Viridis', showscale=True)
        )
        layout = go.Layout(
            scene=dict(
                xaxis_title='X', yaxis_title='Y', zaxis_title='Z'
            ), margin=dict(l=0, r=0, b=0, t=0)
        )
        fig = go.Figure(data=[scatter], layout=layout)
        return fig
    except Exception as e:
        traceback.print_exc()
        return gr.Markdown(f"**Error during inference:**\n```\n{str(e)}\n```")

# --- Gradio interface ---
iface = gr.Interface(
    fn=inference_cloud,
    inputs=gr.Files(label="Upload Images", file_types=["image"], file_count="multiple"),
    outputs=gr.Plot(label="3D Point Cloud"),
    title="StreamVGGT Interactive 3D Reconstruction",
    description="上传一系列连续的 RGB 图像，实时生成并在浏览器中交互式查看点云。"
)

if __name__ == "__main__":
    iface.launch()  # 本地启动，默认 http://127.0.0.1:7860/
