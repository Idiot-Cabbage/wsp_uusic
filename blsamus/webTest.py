import requests

url = "http://localhost:26009/segment2/"

# 构造和接口一致的入参
data = {
    "img_name": "1.png",
    "img_path_relative": "data/segmentation/KidneyUS/imgs/1.png",
    "img_size_bytes": 518259,
    "img_dimensions": [1024, 768],
    "img_format": "png",
    "task": "segmentation",
    "dataset_name": "KidneyUS",
    "organ": "Kidney",
    "data_partition_group": "public_all",
    "mask_name": "",
    "mask_path_relative": "",
    "seg_target_info": {},           # 空字典
    "class_label_index": 0,          # int
    "class_label_name": ""           # str
}
# for i in range(0,200):
resp = requests.post(url, json=data)
print(resp.json())

datacls={
        "img_name": "normal_10.png",
        "img_path_relative": "data/classification/Appendix/0/normal_10.png",
        "img_size_bytes": 153383,
        "img_dimensions": [
            1024,
            789
        ],
        "img_format": "png",
        "task": "classification",
        "dataset_name": "Appendix",
        "organ": "Appendix",
        "data_partition_group": "private_train",
        "mask_name": '',
        "mask_path_relative": '',
        "seg_target_info": {},
        "class_label_index": 0,
        "class_label_name": "no appendicitis"
    }

# for i in range(0,200):
resp = requests.post(url, json=datacls)
print(resp.json())

