from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware  # 关键导入
import os
import numpy as np
from PIL import Image
import torch
import cv2
from pydantic import BaseModel

from model import Model  # 假设 model.py 在同目录或已加入 sys.path

app = FastAPI()
model = Model()

# 允许的源列表（根据需求修改）
origins = [
    "*",    # 前端开发服务器
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,      # 允许的源
    allow_credentials=True,     # 是否支持凭据（如 Cookies）
    allow_methods=["*"],        # 允许的 HTTP 方法（如 GET, POST）
    allow_headers=["*"],        # 允许的请求头
)

SEGMENT_OUT_DIR = "api_out/segment"
os.makedirs(SEGMENT_OUT_DIR, exist_ok=True)

#region
#  {
#         "img_name": "291.png",
#         "img_path_relative": "segmentation/KidneyUS/imgs/291.png",
#         "img_size_bytes": 518259,
#         "img_dimensions": [
#             1024,
#             768
#         ],
#         "img_format": "png",
#         "task": "segmentation",
#         "dataset_name": "KidneyUS",
#         "organ": "Kidney",
#         "data_partition_group": "public_all",
#         "mask_name": "",
#         "mask_path_relative": "",
#         "seg_target_info":null
#         "class_label_index": null,
#         "class_label_name": null
#     },

#endregion

class SegmentRequest(BaseModel):
    img_name: str
    img_path_relative: str
    img_size_bytes: int
    img_dimensions: list
    img_format: str
    task: str
    dataset_name: str
    organ: str
    data_partition_group: str
    mask_name: str = ""
    mask_path_relative: str = ""
    seg_target_info: dict = None
    class_label_index: int = None
    class_label_name: str = None

@app.post("/segment/")
async def segment_image(req: SegmentRequest):
    # 1. 读取本地图片
    img_path = req.img_path_relative
    img = Image.open(img_path).convert('RGB')
    original_size = img.size

    # 2. 预处理
    img_np = np.array(img)
    sample = {'image': img_np / 255.0, 'label': np.zeros(img_np.shape[:2])}
    processed_sample = model.transform(sample)
    image_tensor = processed_sample['image'].to(model.device)

    # 3. 推理
    with torch.no_grad():
        outputs_tuple = model.network(image_tensor)
        seg_out = outputs_tuple[0]
        out_label_back_transform = torch.cat(
            [seg_out[:, 0:1], seg_out[:, 1:2+1-1]], axis=1)
        out = out_label_back_transform[:,0,:,:]>0.5
        prediction = out.squeeze(0).cpu().detach().numpy()
        binary_mask_224 = prediction.astype(np.uint8) * 255
        resized_mask = cv2.resize(
            binary_mask_224, 
            original_size, 
            interpolation=cv2.INTER_NEAREST
        )
        mask_img = Image.fromarray(resized_mask)

    # 4. 保存分割结果
    save_name = os.path.splitext(req.img_name)[0] + "_mask.png"
    save_path = os.path.abspath(os.path.join(SEGMENT_OUT_DIR, save_name))
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    mask_img.save(save_path)

    # 5. 构造返回json，和入参一致，补充mask路径
    result = req.dict()
    result["mask_name"] = save_name
    result["mask_path_relative"] = os.path.relpath(save_path, start=os.getcwd())
    result["mask_path_absolute"] = save_path
    return JSONResponse(result)
@app.post("/segment2/")
async def segment_image2(req: SegmentRequest):
    img_path = req.img_path_relative
    img = Image.open(img_path).convert('RGB')
    original_size = img.size
    img_np = np.array(img)
    sample = {'image': img_np / 255.0, 'label': np.zeros(img_np.shape[:2])}
    processed_sample = model.transform(sample)
    image_tensor = processed_sample['image'].to(model.device)

    with torch.no_grad():
        outputs_tuple = model.network(image_tensor)

    # 根据 task 字段返回不同内容
    if req.task == "segmentation":
        seg_out = outputs_tuple[0]
        out_label_back_transform = torch.cat(
            [seg_out[:, 0:1], seg_out[:, 1:2+1-1]], axis=1)
        out = out_label_back_transform[:,0,:,:]>0.5
        prediction = out.squeeze(0).cpu().detach().numpy()
        binary_mask_224 = prediction.astype(np.uint8) * 255
        resized_mask = cv2.resize(
            binary_mask_224, 
            original_size, 
            interpolation=cv2.INTER_NEAREST
        )
        mask_img = Image.fromarray(resized_mask)
        save_name = os.path.splitext(req.img_name)[0] + "_mask.png"
        save_path = os.path.abspath(os.path.join(SEGMENT_OUT_DIR, save_name))
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        mask_img.save(save_path)

        result = req.dict()
        result["mask_name"] = save_name
        result["mask_path_relative"] = os.path.relpath(save_path, start=os.getcwd())
        result["mask_path_absolute"] = save_path
        return JSONResponse(result)

    elif req.task == "classification":
        # 假设二分类
        logits = outputs_tuple[1]
        probabilities = torch.softmax(logits, dim=1).cpu().numpy().flatten()
        prediction = int(np.argmax(probabilities))
        result = req.dict()
        result["probability"] = probabilities.tolist()
        result["prediction"] = prediction
        return JSONResponse(result)

    else:
        return JSONResponse({"error": "不支持的task类型"}, status_code=400)

@app.get("/")
def read_root():
    return {"message": "普渡超声"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=16009)