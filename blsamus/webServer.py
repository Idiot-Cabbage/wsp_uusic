from fastapi import FastAPI

app = FastAPI()


#region
#   {
#         "img_name": "cls_00000.jpg",
#         "img_path_relative": "classification/Appendix/imgs/cls_00000.jpg",
#         "img_size_bytes": 118306,
#         "img_dimensions": [
#             1008,
#             819
#         ],
#         "img_format": "jpg",
#         "task": "classification",
#         "dataset_name": "Appendix",
#         "organ": "Appendix",
#         "data_partition_group": "private_val",
#         "mask_name": null,
#         "mask_path_relative": null,
#         "seg_target_info": null,
#         "class_label_index": null,
#         "class_label_name": null
#     },

#endregion
@app.get("/")
def read_root():
    return {"message": "hello world"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=6006)