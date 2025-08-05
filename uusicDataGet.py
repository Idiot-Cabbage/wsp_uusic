# # import synapseclient
# # import ssl

# # # 禁用SSL验证（仅用于测试）
# # ssl._create_default_https_context = ssl._create_unverified_context

# # syn = synapseclient.Synapse()
# # syn.login(authToken="eyJ0eXAiOiJKV1QiLCJraWQiOiJXN05OOldMSlQ6SjVSSzpMN1RMOlQ3TDc6M1ZYNjpKRU9VOjY0NFI6VTNJWDo1S1oyOjdaQ0s6RlBUSCIsImFsZyI6IlJTMjU2In0.eyJhY2Nlc3MiOnsic2NvcGUiOlsidmlldyIsImRvd25sb2FkIl0sIm9pZGNfY2xhaW1zIjp7fX0sInRva2VuX3R5cGUiOiJQRVJTT05BTF9BQ0NFU1NfVE9LRU4iLCJpc3MiOiJodHRwczovL3JlcG8tcHJvZC5wcm9kLnNhZ2ViYXNlLm9yZy9hdXRoL3YxIiwiYXVkIjoiMCIsIm5iZiI6MTc1MjEzNTczMiwiaWF0IjoxNzUyMTM1NzMyLCJqdGkiOiIyMjY4MiIsInN1YiI6IjM1NDk0NzAifQ.BgVlwWspmAbJqQk8NdgWNfU6VJQIvYTL9zcQB0jkf8EwzB8EDEYGwoZViCHHlhKzG1rtwQjqRUvVifRnaPQ1JkggyvuySftHmEw_Ds-TGBbnhdUrQlL6w45AECZtSOE-kjQ8RBs2eR4-FgpLPG1ocPQghUamgI18xey0i-o95pp1342gMfoXeV_Vr9pCs0beLlptxaGEMebNTNEeOA0WihqnBu0r07-VwWx_xD1Lo4izdpuruxZBHvlmTgCssyk9ND74xLu4FpQ-eDCCzkSHCIZgqdJSy-2-5Cs31z0ASxppWmc8iTaX_rbsfj9qJKEGnuEZ_0J3fcoYqV0Rf0W_eQ")  # or syn.login('username', 'password')
# # entity = syn.get("syn68188515")


# import synapseclient
# import requests

# # 检查网络连接
# try:
#     response = requests.get("https://repo-prod.prod.sagebase.org/repo/v1/version")
#     print(f"Synapse 服务器状态: {response.status_code}")
# except Exception as e:
#     print(f"网络连接错误: {e}")

# syn = synapseclient.Synapse()
# syn.login(authToken="eyJ0eXAiOiJKV1QiLCJraWQiOiJXN05OOldMSlQ6SjVSSzpMN1RMOlQ3TDc6M1ZYNjpKRU9VOjY0NFI6VTNJWDo1S1oyOjdaQ0s6RlBUSCIsImFsZyI6IlJTMjU2In0")

import synapseclient
syn = synapseclient.Synapse()
print(f"缓存目录: {syn.cache.cache_root_dir}")
import ssl
import synapseclient
import urllib3
import os

# # 方法1: 禁用SSL验证
# ssl._create_default_https_context = ssl._create_unverified_context

# # 方法2: 禁用urllib3警告
# urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# # 方法3: 设置环境变量
# os.environ['PYTHONHTTPSVERIFY'] = '0'

def download_folder_contents(syn, folder_id, download_dir, folder_name):
    """下载文件夹中的所有文件"""
    folder_path = os.path.join(download_dir, folder_name)
    os.makedirs(folder_path, exist_ok=True)
    
    print(f"📁 正在下载文件夹: {folder_name}")
    
    try:
        children = list(syn.getChildren(folder_id))
        for i, child in enumerate(children, 1):
            child_name = child['name']
            child_id = child['id']
            child_type = child['type']
            
            if child_type == 'org.sagebionetworks.repo.model.FileEntity':
                try:
                    print(f"  📥 正在下载 ({i}/{len(children)}): {child_name}")
                    file_entity = syn.get(child_id, downloadLocation=folder_path)
                    print(f"  ✅ 下载完成: {file_entity.path}")
                    
                    # 显示文件大小
                    file_size = os.path.getsize(file_entity.path)
                    if file_size > 1024*1024:
                        print(f"  📊 文件大小: {file_size / (1024*1024):.2f} MB")
                    else:
                        print(f"  📊 文件大小: {file_size / 1024:.2f} KB")
                    print()
                    
                except Exception as e:
                    print(f"  ❌ 下载失败 {child_name}: {e}")
                    print()
            else:
                print(f"  ⚠️ 跳过非文件项: {child_name} (类型: {child_type})")
                
    except Exception as e:
        print(f"❌ 处理文件夹失败: {e}")

try:
    syn = synapseclient.Synapse()
    syn.login(authToken="eyJ0eXAiOiJKV1QiLCJraWQiOiJXN05OOldMSlQ6SjVSSzpMN1RMOlQ3TDc6M1ZYNjpKRU9VOjY0NFI6VTNJWDo1S1oyOjdaQ0s6RlBUSCIsImFsZyI6IlJTMjU2In0.eyJhY2Nlc3MiOnsic2NvcGUiOlsidmlldyIsImRvd25sb2FkIl0sIm9pZGNfY2xhaW1zIjp7fX0sInRva2VuX3R5cGUiOiJQRVJTT05BTF9BQ0NFU1NfVE9LRU4iLCJpc3MiOiJodHRwczovL3JlcG8tcHJvZC5wcm9kLnNhZ2ViYXNlLm9yZy9hdXRoL3YxIiwiYXVkIjoiMCIsIm5iZiI6MTc1MjEzNTczMiwiaWF0IjoxNzUyMTM1NzMyLCJqdGkiOiIyMjY4MiIsInN1YiI6IjM1NDk0NzAifQ.BgVlwWspmAbJqQk8NdgWNfU6VJQIvYTL9zcQB0jkf8EwzB8EDEYGwoZViCHHlhKzG1rtwQjqRUvVifRnaPQ1JkggyvuySftHmEw_Ds-TGBbnhdUrQlL6w45AECZtSOE-kjQ8RBs2eR4-FgpLPG1ocPQghUamgI18xey0i-o95pp1342gMfoXeV_Vr9pCs0beLlptxaGEMebNTNEeOA0WihqnBu0r07-VwWx_xD1Lo4izdpuruxZBHvlmTgCssyk9ND74xLu4FpQ-eDCCzkSHCIZgqdJSy-2-5Cs31z0ASxppWmc8iTaX_rbsfj9qJKEGnuEZ_0J3fcoYqV0Rf0W_eQ")
    entity = syn.get("syn68188515")
    # 显示详细信息
    print("✅ 下载成功!")
    print(f"📁 文件名: {entity.name}")
 # 安全地检查路径
    if hasattr(entity, 'path') and entity.path:
        print(f"📍 文件路径: {entity.path}")
        if os.path.exists(entity.path):
            print(f"📊 文件大小: {os.path.getsize(entity.path) / (1024*1024):.2f} MB")
            print(f"✅ 文件确实存在于: {entity.path}")
        else:
            print("❌ 文件路径存在但文件不存在")
    else:
        print("⚠️ entity.path 不可用")
        
    print(f"🗂️ 缓存目录: {syn.cache.cache_root_dir}")
    
    # 显示entity的所有属性
    print("\n📋 Entity信息:")
    print(f"ID: {entity.id}")
    print(f"类型: {type(entity)}")
    # 下载项目中的所有文件到指定目录
    download_dir = "./UUSIC2025_data"
    os.makedirs(download_dir, exist_ok=True)

    print(f"📥 开始下载项目中的所有文件到: {download_dir}")

    print(syn)
    try:
        # 首先获取失败项目的详细信息
        children = list(syn.getChildren(entity.id))
        for child in children:
            child_name = child['name']
            child_id = child['id']
            child_type = child['type']
            if child['type'] == 'org.sagebionetworks.repo.model.FileEntity':
            # 这是一个文件，直接下载
                try:
                    file_entity = syn.get(child_id, downloadLocation=download_dir)
                    print(f"✅ 文件下载完成: {file_entity.path}")
                    print(f"📊 文件大小: {os.path.getsize(file_entity.path) / (1024*1024):.2f} MB")
                    print()
                except Exception as e:
                    print(f"❌ 文件下载失败 {child_name}: {e}")
                    print()
                    
            elif child['name'] == 'dataset_json_fingerprints_v3':
                failed_id = child['id']
                print(f"🔍 正在诊断: {child['name']}")
                print(f"🆔 ID: {failed_id}")
                print(f"📋 类型: {child['type']}")
                
                # 获取实体信息但不下载文件
                failed_entity = syn.get(failed_id, downloadFile=False)
                print(f"📊 实体类型: {type(failed_entity)}")
                print(f"📄 实体属性: {dir(failed_entity)}")
                
                
                # 如果是文件夹，列出其内容
                if child['type'] == 'org.sagebionetworks.repo.model.Folder':                    
                    print("📁 这是一个文件夹，包含:")
                    sub_children = list(syn.getChildren(failed_id))
                    for sub_child in sub_children:
                        print(f"  - {sub_child['name']} (类型: {sub_child['type']})")
                    
                    download_folder_contents(syn, child_id, download_dir, child_name)
                break

    except Exception as e:
        print(f"❌ 诊断失败: {e}")  
    # for i, child in enumerate(children, 1):
    #     try:
    #         print(f"📥 正在下载 ({i}/{len(children)}): {child['name']}")
    #         file_entity = syn.get(child['id'], downloadLocation=download_dir)
    #         print(f"✅ 下载完成: {file_entity.path}")
    #         print(f"📊 文件大小: {os.path.getsize(file_entity.path) / (1024*1024):.2f} MB")
    #         print()
    #     except Exception as e:
    #         print(f"❌ 下载失败 {child['name']}: {e}")
    #         print()

    # print("🎉 所有文件下载完成!")
    
    # # 如果是Folder或Project，列出内容
    # if hasattr(entity, 'children'):
    #     print("📂 这是一个文件夹，包含:")
    #     for child in syn.getChildren(entity.id):
    #         print(f"  - {child['name']} (ID: {child['id']})")
            
except Exception as e:
    print(f"❌ 错误: {e}")
    print(f"错误类型: {type(e)}")

