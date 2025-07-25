# This file is used to configure the training parameters for each task
class Config_US30K:
    # This dataset contain all the collected ultrasound dataset
    data_path = "dataset/SAMUS/"  
    save_path = "./checkpoints/SAMUS/"
    result_path = "./result/SAMUS/"
    tensorboard_path = "./tensorboard/SAMUS/"
    load_path = save_path + "/xxx.pth"
    save_path_code = "_"

    workers = 1                         # number of data loading workers (default: 8)
    epochs = 200                        # number of total epochs to run (default: 400)
    batch_size = 8                     # batch size (default: 4)
    learning_rate = 5e-4                # iniial learning rate (default: 0.001)
    momentum = 0.9                      # momntum
    classes = 2                         # thenumber of classes (background + foreground)
    img_size = 256                      # theinput size of model
    train_split = "train"               # the file name of training set
    val_split = "val"                   # the file name of testing set
    test_split = "test"                 # the file name of testing set
    crop = None                         # the cropped image size
    eval_freq = 1                       # the frequency of evaluate the model
    save_freq = 2000                    # the frequency of saving the model
    device = "cuda"                     # training device, cpu or cuda
    cuda = "on"                         # switch on/off cuda option (default: off)
    gray = "yes"                        # the type of input image
    img_channel = 1                     # the channel of input image
    eval_mode = "mask_slice"                 # the mode when evaluate the model, slice level or patient level
    pre_trained = False
    mode = "train"
    visual = False
    modelname = "SAM"

# -------------------------------------------------------------------------------------------------
class Config_TN3K:
    data_path = "./dataset/SAMUS/"
    data_subpath = "./dataset/SAMUS/TN3K/"
    save_path = "./checkpoints/TN3K/"
    result_path = "./result/TN3K/"
    tensorboard_path = "./tensorboard/TN3K/"
    load_path = save_path + "/SAMUS_01111456_352_0.8360216348510668.pth" # SAMUS_12291902_47_0.8595338325001363.pth   SAMUS_01081717_8_0.42477967570244357.pth
    load_classifier_path = save_path + "/Resnet18_01032018_59_0.7805483249970666.pth"
    save_path_code = "_"

    workers = 0                  # number of data loading workers (default: 8)
    epochs = 400                 # number of total epochs to run (default: 400)
    batch_size = 8               # batch size (default: 4)
    learning_rate = 1e-4         # initial learning rate (default: 0.001)
    momentum = 0.9               # momentum
    classes = 2                  # the number of classes (background + foreground)
    img_size = 256               # the input size of model
    train_split = "train"  # the file name of training set
    val_split = "val"     # the file name of testing set
    test_split = "test"     # the file name of testing set
    crop = None                  # the cropped image size
    eval_freq = 1                # the frequency of evaluate the model
    save_freq = 2000               # the frequency of saving the model
    device = "cuda"              # training device, cpu or cuda
    cuda = "on"                  # switch on/off cuda option (default: off)
    gray = "yes"                 # the type of input image
    img_channel = 1              # the channel of input image
    eval_mode = "mask_slice"        # the mode when evaluate the model, slice level or patient level
    pre_trained = False
    mode = "train"
    visual = False
    modelname = "SAMUS"

    # 分类参数
    segmentation_data_path = ""

    classifier_epochs= 200
    classifier_batch_size = 128
    classifier_learning_rate = 1e-4  # initial learning rate (default: 0.001)
    classifier_momentum = 0.9  # momentum
    classifier_classes = 2  # the number of classes (normal + abnormal)
    classifier_name = "Resnet18"  # (Vit,Resnet18)
    classifier_size = 256 # (Vit:224,Resnet18 256)

class Config_KTD:
    data_path = "./dataset/SAMUS/KTD/"
    data_subpath = "./dataset/SAMUS/KTD/"
    save_path = "./checkpoints/KTD/"
    result_path = "./result/KTD/"
    tensorboard_path = "./tensorboard/KTD/"
    load_path = save_path + "/SAMUS_07121649_13_0.8649140261499766.pth" # SAMUS_12291902_47_0.8595338325001363.pth   SAMUS_01081717_8_0.42477967570244357.pth
    load_classifier_path = save_path + "/Resnet18_01032018_59_0.7805483249970666.pth"
    save_path_code = "_"

    workers = 0                  # number of data loading workers (default: 8)
    epochs = 100                 # number of total epochs to run (default: 400)
    batch_size = 4               # batch size (default: 4)
    learning_rate = 1e-4         # initial learning rate (default: 0.001)
    momentum = 0.9               # momentum
    classes = 2                  # the number of classes (background + foreground)
    img_size = 256               # the input size of model
    train_split = "train"  # the file name of training set
    val_split = "val"     # the file name of testing set
    test_split = "test"     # the file name of testing set
    crop = None                  # the cropped image size
    eval_freq = 1                # the frequency of evaluate the model
    save_freq = 2000               # the frequency of saving the model
    device = "cuda"              # training device, cpu or cuda
    cuda = "on"                  # switch on/off cuda option (default: off)
    gray = "yes"                 # the type of input image
    img_channel = 1              # the channel of input image
    eval_mode = "mask_slice"        # the mode when evaluate the model, slice level or patient level
    pre_trained = False
    mode = "train"
    visual = False
    modelname = "SAMUS"

    # 分类参数
    segmentation_data_path = ""

    classifier_epochs= 200
    classifier_batch_size = 128
    classifier_learning_rate = 1e-4  # initial learning rate (default: 0.001)
    classifier_momentum = 0.9  # momentum
    classifier_classes = 2  # the number of classes (normal + abnormal)
    classifier_name = "Resnet18"  # (Vit,Resnet18)
    classifier_size = 256 # (Vit:224,Resnet18 256)
class Config_BUSI:
    # This dataset is for breast cancer segmentation
    data_path = "../../dataset/SAMUS/"
    data_subpath = "../../dataset/SAMUS/Breast-BUSI/"   
    save_path = "./checkpoints/BUSI/"
    result_path = "./result/BUSI/"
    tensorboard_path = "./tensorboard/BUSI/"
    load_path = save_path + "/xxx.pth"
    save_path_code = "_"

    workers = 1                         # number of data loading workers (default: 8)
    epochs = 400                        # number of total epochs to run (default: 400)
    batch_size = 8                     # batch size (default: 4)
    learning_rate = 1e-4                # iniial learning rate (default: 0.001)
    momentum = 0.9                      # momntum
    classes = 2                         # thenumber of classes (background + foreground)
    img_size = 256                      # theinput size of model
    train_split = "train-Breast-BUSI"   # the file name of training set
    val_split = "val-Breast-BUSI"       # the file name of testing set
    test_split = "test-Breast-BUSI"     # the file name of testing set
    crop = None                         # the cropped image size
    eval_freq = 1                       # the frequency of evaluate the model
    save_freq = 2000                    # the frequency of saving the model
    device = "cuda"                     # training device, cpu or cuda
    cuda = "on"                         # switch on/off cuda option (default: off)
    gray = "yes"                        # the type of input image
    img_channel = 1                     # the channel of input image
    eval_mode = "mask_slice"                 # the mode when evaluate the model, slice level or patient level
    pre_trained = False
    mode = "train"
    visual = False
    modelname = "SAM"

class Config_CAMUS:
    # This dataset is for breast cancer segmentation
    data_path = "../../dataset/SAMUS/"  # 
    data_subpath = "../../dataset/SAMUS/Echocardiography-CAMUS/" 
    save_path = "./checkpoints/CAMUS/"
    result_path = "./result/CAMUS/"
    tensorboard_path = "./tensorboard/CAMUS/"
    load_path = save_path + "/xxx.pth"
    save_path_code = "_"

    workers = 1                         # number of data loading workers (default: 8)
    epochs = 400                        # number of total epochs to run (default: 400)
    batch_size = 8                     # batch size (default: 4)
    learning_rate = 1e-4                # iniial learning rate (default: 0.001)
    momentum = 0.9                      # momntum
    classes = 4                        # thenumber of classes (background + foreground)
    img_size = 256                      # theinput size of model
    train_split = "train-EchocardiographyLA-CAMUS"   # the file name of training set
    val_split = "val-EchocardiographyLA-CAMUS"       # the file name of testing set
    test_split = "test-Echocardiography-CAMUS"     # the file name of testing set # HMCQU
    crop = None                         # the cropped image size
    eval_freq = 1                       # the frequency of evaluate the model
    save_freq = 2000                    # the frequency of saving the model
    device = "cuda"                     # training device, cpu or cuda
    cuda = "on"                         # switch on/off cuda option (default: off)
    gray = "yes"                        # the type of input image
    img_channel = 1                     # the channel of input image
    eval_mode = "camusmulti"                 # the mode when evaluate the model, slice level or patient level
    pre_trained = False
    mode = "train"
    visual = False
    modelname = "SAM"

class Config_OurDataset:
    # This dataset contain our the collected Kidney ultrasound dataset
    data_path = "../../dataset/SAMUS/"
    save_path = "./checkpoints/SAMUS/"
    result_path = "./result/SAMUS/"
    tensorboard_path = "./tensorboard/SAMUS/"
    load_path = save_path + "/xxx.pth"
    save_path_code = "_"

    workers = 1  # number of data loading workers (default: 8)
    epochs = 200  # number of total epochs to run (default: 400)
    batch_size = 8  # batch size (default: 4)
    learning_rate = 5e-4  # iniial learning rate (default: 0.001)
    momentum = 0.9  # momntum
    classes = 2  # thenumber of classes (background + foreground)
    img_size = 256  # theinput size of model
    train_split = "train"  # the file name of training set
    val_split = "val"  # the file name of testing set
    test_split = "test"  # the file name of testing set
    crop = None  # the cropped image size
    eval_freq = 1  # the frequency of evaluate the model
    save_freq = 2000  # the frequency of saving the model
    device = "cuda"  # training device, cpu or cuda
    cuda = "on"  # switch on/off cuda option (default: off)
    gray = "yes"  # the type of input image
    img_channel = 1  # the channel of input image
    eval_mode = "mask_slice"  # the mode when evaluate the model, slice level or patient level
    pre_trained = False
    mode = "train"
    visual = False
    modelname = "SAM"
class Config_UUSIC:
    datatype="private_Thyroid/"
    # datatype="Fetal_HC/"
    data_path = "blsamus/data/segmentation/"+ datatype
    data_subpath = "blsamus/data/segmentation/"+ datatype
    save_path = "./checkpoints/UUSIC/"+datatype
    result_path = "./result/UUSIC/"+datatype
    tensorboard_path = "./tensorboard/UUSIC/"+datatype
    load_path = save_path + "SAMUS_07160118_56_0.6883226564873587.pth" # SAMUS_12291902_47_0.8595338325001363.pth   SAMUS_01081717_8_0.42477967570244357.pth
    load_classifier_path = save_path + "/Resnet18_01032018_59_0.7805483249970666.pth"
    save_path_code = "_"

    workers = 0                  # number of data loading workers (default: 8)
    epochs = 100                 # number of total epochs to run (default: 400)
    batch_size = 1               # batch size (default: 4)
    learning_rate = 1e-4         # initial learning rate (default: 0.001)
    momentum = 0.9               # momentum
    classes = 2                  # the number of classes (background + foreground)
    img_size = 224               # the input size of model
    train_split = "train"  # the file name of training set
    val_split = "val"     # the file name of testing set
    test_split = "test"     # the file name of testing set
    crop = None                  # the cropped image size
    eval_freq = 1                # the frequency of evaluate the model
    save_freq = 2000               # the frequency of saving the model
    device = "cuda"              # training device, cpu or cuda
    cuda = "on"                  # switch on/off cuda option (default: off)
    gray = "yes"                 # the type of input image
    img_channel = 1              # the channel of input image
    eval_mode = "mask_slice"        # the mode when evaluate the model, slice level or patient level
    pre_trained = False
    mode = "train"
    visual = False
    modelname = "SAMUS"

    # 分类参数
    segmentation_data_path = ""

    classifier_epochs= 200
    classifier_batch_size = 128
    classifier_learning_rate = 1e-4  # initial learning rate (default: 0.001)
    classifier_momentum = 0.9  # momentum
    classifier_classes = 2  # the number of classes (normal + abnormal)
    classifier_name = "Resnet18"  # (Vit,Resnet18)
    classifier_size = 256 # (Vit:224,Resnet18 256)    
# ==================================================================================================
def get_config(task="US30K"):
    if task == "US30K":
        return Config_US30K()
    elif task == "TN3K":
        return Config_TN3K()
    elif task == 'KTD':
        return  Config_KTD()
    elif task == "BUSI":
        return Config_BUSI()
    elif task == "CAMUS":
        return Config_CAMUS()
    elif task == "OurDataset":
        return Config_OurDataset()
    elif task == "UUSIC":
        return Config_UUSIC()
    else:
        assert("We do not have the related dataset, please choose another task.")