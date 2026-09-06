from collections import defaultdict
from itertools import product

###### Config
##Version
FULL_DATASET_IDENTIFIER = "L"
BOX_LARGE_DATASET_IDENTIFIER = "B"
FULL_LARGE_DATASET_IDENTIFIER = "F"
##Size
SMALL_IDENTIFIER = "s"
MEDIUM_IDENTIFIER = "m"
LARGE_IDENTIFIER = "l"

###### A2D2 Dataset
A2D2_DATASET_NAME = "A2D2"
A2D2_SIZE_IDENTIFIER = {"normal": "", "large": LARGE_IDENTIFIER, "medium": MEDIUM_IDENTIFIER, "small": SMALL_IDENTIFIER}
A2D2_DATASET_VERSION_IDENTIFIER = ["", FULL_DATASET_IDENTIFIER,BOX_LARGE_DATASET_IDENTIFIER,FULL_LARGE_DATASET_IDENTIFIER]
A2D2_PREPROCESSING_IDENTIFIER = {"none": "", "norm": "N", "local": "U", "equalized": "E", "preNorm": "P"}
A2D2_AUGMENTATION_IDENTIFIER = ["", "R", "R2"]


def create_dataset(name,size,ds_version,preprocessing,augmentation):
    ## name - 0/1 - 0/1 - 0/1 -0/2
    combinations=list(product([name],size.values(),ds_version,preprocessing.values(),augmentation))
    datasets=[]
    for comb in combinations:
        datasets.append("".join(comb))
    dataset_sizes = defaultdict(list)
    dataset_preprocessing = defaultdict(list)
    for k,v in size.items():
        for data in datasets:
            if v in data:
                dataset_sizes[k].append(data)
    for k, v in preprocessing.items():
        for data in datasets:
            if v in data:
                dataset_preprocessing[k].append(data)
    return datasets, dataset_sizes, dataset_preprocessing,


A2D2_DATASETS, A2D2_size, A2D2_pre = create_dataset(A2D2_DATASET_NAME, A2D2_SIZE_IDENTIFIER, A2D2_DATASET_VERSION_IDENTIFIER, A2D2_PREPROCESSING_IDENTIFIER, A2D2_AUGMENTATION_IDENTIFIER)
A2D2_SMALL, A2D2_MEDIUM, A2D2_LARGE = A2D2_size["small"], A2D2_size["medium"], A2D2_size["large"]
A2D2_NORM, A2D2_LOCAL_NORM_UNI, A2D2_EQUALIZED, A2D2_PRE_NORM = A2D2_pre["norm"], A2D2_pre["local"], A2D2_pre["equalized"], A2D2_pre["preNorm"]
# A2D2_EQUALIZED=["A2D2E", "A2D2ER", "A2D2ER2", 'A2D2lE', 'A2D2mE', 'A2D2lLE', 'A2D2mLE']
# A2D2_LOCAL_NORM_UNI=["A2D2U", "A2D2UR", "A2D2UR2", 'A2D2lU', 'A2D2mU', 'A2D2lLU', 'A2D2mLU']


########### GTA-V Dataset

GTAV_DATASET_NAME = "GTAVS"
GTAV_SIZE_IDENTIFIER = {"normal": "", "medium": "m", "small": "s"}
GTAV_DATASET_VERSION_IDENTIFIER = [""]
GTAV_PREPROCESSING_IDENTIFIER = {"none": "", "norm": "N", "local": "U", "equalized": "E"}
GTAV_AUGMENTATION_IDENTIFIER = ["", "R","2","3"]
GTAV_DATASETS, GTAV_size, GTAV_pre = create_dataset(GTAV_DATASET_NAME, GTAV_SIZE_IDENTIFIER, GTAV_DATASET_VERSION_IDENTIFIER, GTAV_PREPROCESSING_IDENTIFIER, GTAV_AUGMENTATION_IDENTIFIER)
GTAV_SMALL, GTAV_MEDIUM, GTAV_LARGE = GTAV_size["small"], GTAV_size["medium"], GTAV_size["large"]
GTAV_NORM, GTAV_LOCAL_NORM_UNI, GTAV_EQUALIZED = GTAV_pre["norm"], GTAV_pre["local"], GTAV_pre["equalized"]


########### 50 salads Dataset

SALAD_DATASET_NAME = "salad"
SALAD_SIZE_IDENTIFIER = {"normal": "", "medium": "m", "small": "s"}
SALAD_DATASET_VERSION_IDENTIFIER = [""]
SALAD_PREPROCESSING_IDENTIFIER = {"none": "", "norm": "N", "local": "U", "equalized": "E"}
SALAD_AUGMENTATION_IDENTIFIER = ["", "R","2"]
SALAD_DATASETS, SALAD_size, SALAD_pre = create_dataset(SALAD_DATASET_NAME, SALAD_SIZE_IDENTIFIER, SALAD_DATASET_VERSION_IDENTIFIER, SALAD_PREPROCESSING_IDENTIFIER, SALAD_AUGMENTATION_IDENTIFIER)
SALAD_SMALL, SALAD_MEDIUM, SALAD_LARGE = SALAD_size["small"], SALAD_size["medium"], SALAD_size["large"]
SALAD_NORM, SALAD_LOCAL_NORM_UNI, SALAD_EQUALIZED = SALAD_pre["norm"], SALAD_pre["local"], SALAD_pre["equalized"]

########### GTEA Dataset

GTEA_DATASET_NAME = "GTEA"
GTEA_SIZE_IDENTIFIER = {"normal": "", "medium": "m", "small": "s"}
GTEA_DATASET_VERSION_IDENTIFIER = [""]
GTEA_PREPROCESSING_IDENTIFIER = {"none": "", "norm": "N", "local": "U", "equalized": "E"}
GTEA_AUGMENTATION_IDENTIFIER = ["", "R", "2"]
GTEA_DATASETS, GTEA_size, GTEA_pre = create_dataset(GTEA_DATASET_NAME, GTEA_SIZE_IDENTIFIER, GTEA_DATASET_VERSION_IDENTIFIER, GTEA_PREPROCESSING_IDENTIFIER, GTEA_AUGMENTATION_IDENTIFIER)
GTEA_SMALL, GTEA_MEDIUM, GTEA_LARGE = GTEA_size["small"], GTEA_size["medium"], GTEA_size["large"]
GTEA_NORM, GTEA_LOCAL_NORM_UNI, GTEA_EQUALIZED = GTEA_pre["norm"], GTEA_pre["local"], GTEA_pre["equalized"]


################ Nuscenes

NS_DATASET_NAME = "nuscenes" #s inside
NS_SIZE_IDENTIFIER = {"normal": ""}
NS_DATASET_VERSION_IDENTIFIER = [""]
NS_PREPROCESSING_IDENTIFIER = {"none": "", "norm": "N", "local": "U", "equalized": "E"}
NS_AUGMENTATION_IDENTIFIER = ["", "R"]
NS_DATASETS, NS_size, NS_pre = create_dataset(NS_DATASET_NAME, NS_SIZE_IDENTIFIER, NS_DATASET_VERSION_IDENTIFIER, NS_PREPROCESSING_IDENTIFIER, NS_AUGMENTATION_IDENTIFIER)
NS_SMALL, NS_MEDIUM, NS_LARGE = NS_size["small"], NS_size["medium"], NS_size["large"]
NS_NORM, NS_LOCAL_NORM_UNI, NS_EQUALIZED = NS_pre["norm"], NS_pre["local"], NS_pre["equalized"]


################ Cityscapes

CS_DATASET_NAME = "cityscapes" #s inside
CS_SIZE_IDENTIFIER = {"normal": ""}
CS_DATASET_VERSION_IDENTIFIER = [""]
CS_PREPROCESSING_IDENTIFIER = {"none": "", "norm": "N", "local": "U", "equalized": "E", "preNorm": "P"}
CS_AUGMENTATION_IDENTIFIER = ["", "R", "2", "R2"]
CS_DATASETS, CS_size, CS_pre = create_dataset(CS_DATASET_NAME, CS_SIZE_IDENTIFIER, CS_DATASET_VERSION_IDENTIFIER, CS_PREPROCESSING_IDENTIFIER, CS_AUGMENTATION_IDENTIFIER)
CS_SMALL, CS_MEDIUM, CS_LARGE = CS_size["small"], CS_size["medium"], CS_size["large"]
CITYSCAPES_NORM, CITYSCAPES_LOCAL_NORM_UNI, CITYSCAPES_EQUALIZED, CITYSCAPES_PRE_NORM = CS_pre["norm"], CS_pre["local"], CS_pre["equalized"], CS_pre["PreNorm"]


### Collection


DATASETS_CLS=A2D2_DATASETS+GTAV_DATASETS+NS_DATASETS+CS_DATASETS+SALAD_DATASETS+GTEA_DATASETS
NORMED_DATASETS_CLS=A2D2_NORM+GTAV_NORM+NS_NORM+CITYSCAPES_NORM


CITYSCAPES_DATASET_NAME = "cityscapes"
CITYSCAPES_SIZE_IDENTIFIER = {"normal": "", "small": "m", "medium": "l"}
CITYSCAPES_DATASET_VERSION_IDENTIFIER = [""]
CITYSCAPES_PREPROCESSING_IDENTIFIER = {"none": "", "norm": "N", "local": "U", "equalized": "E", "preNorm": "P"}
CITYSCAPES_AUGMENTATION_IDENTIFIER = ["","R","r","R2","r2","2"]

CITYSCAPES_DATASETS, CITYSCAPES_size, CITYSCAPES_pre = create_dataset(CITYSCAPES_DATASET_NAME,CITYSCAPES_SIZE_IDENTIFIER,CITYSCAPES_DATASET_VERSION_IDENTIFIER,CITYSCAPES_PREPROCESSING_IDENTIFIER,CITYSCAPES_AUGMENTATION_IDENTIFIER)
CITYSCAPES_SMALL, CITYSCAPES_MEDIUM, CITYSCAPES_LARGE = CITYSCAPES_size["small"], CITYSCAPES_size["medium"], CITYSCAPES_size["large"]
CITYSCAPES_PLAIN, CITYSCAPES_NORM, CITYSCAPES_LOCAL_NORM_UNI, CITYSCAPES_EQUALIZED, CITYSCAPES_PRE_NORM = CITYSCAPES_pre["none"], CITYSCAPES_pre["norm"], CITYSCAPES_pre["local"], CITYSCAPES_pre["equalized"], CITYSCAPES_pre["preNorm"]

DATASETS_SEM_SEG=CITYSCAPES_DATASETS



LYFT_DATASET_NAME = "lyft"
LYFT_SIZE_IDENTIFIER = {"normal": ""}
LYFT_DATASET_VERSION_IDENTIFIER = ["","SSD"]
LYFT_PREPROCESSING_IDENTIFIER = {"none": "", "norm": "N", "local": "U", "equalized": "E", "preNorm": "P"}
LYFT_AUGMENTATION_IDENTIFIER = ["","R"]

LYFT_DATASETS, LYFT_size, LYFT_pre = create_dataset(LYFT_DATASET_NAME,LYFT_SIZE_IDENTIFIER,LYFT_DATASET_VERSION_IDENTIFIER,LYFT_PREPROCESSING_IDENTIFIER,LYFT_AUGMENTATION_IDENTIFIER)
LYFT_SMALL, LYFT_MEDIUM, LYFT_LARGE = LYFT_size["small"], LYFT_size["medium"], LYFT_size["large"]
LYFT_PLAIN, LYFT_NORM, LYFT_LOCAL_NORM_UNI, LYFT_EQUALIZED, LYFT_PRE_NORM = LYFT_pre["none"], LYFT_pre["norm"], LYFT_pre["local"], LYFT_pre["equalized"], LYFT_pre["preNorm"]



GTA_V_STREETS_DATASETS=["GTAVS", "GTAVSN", "GTAVSm", "GTAVSmN","GTAVSmU","GTAVSU","GTAVSmE","GTAVSE"]

A2D2_DATASETS_CLS=A2D2_LOCAL_NORM_UNI+A2D2_EQUALIZED+A2D2_PRE_NORM
LYFT_DATASETS= LYFT_NORM+ LYFT_LOCAL_NORM_UNI+ LYFT_EQUALIZED+ LYFT_PRE_NORM+LYFT_PLAIN

DATASETS_CLS
DATASETS_SEM_SEG
DETECTION2d_DATASETS=LYFT_DATASETS
ALL_DATASETS = ["svhn","svhn-ll",'cifar10',"cifar10-ll",'cifar10-test',"cifar10-ms","cifar100-ll","cifar100-ba","cifar100-ta","cifar100-ub-ta","cifar100-ls","cifar100-ms","cifar100", 'A2D2', 'A2D2B', 'A2D2TE',"A2D2LN","A2D2LE", 'cifar100', "cityscapesN2","cityscapesN", "cityscapesNm", "cityscapesPR", 'nuscenes', 'nuscenesM', "nuscenesN","cityscapes","cityscapesR",
                "nuscenesNR", 'A2D2M', 'A2D2N', 'A2D2l', 'A2D2m','A2D2lL', 'A2D2mL','A2D2lLN', 'A2D2mLN', 'A2D2lN', 'A2D2mN', 'A2D2mNR', "A2D2E", "A2D2R", "A2D2ER", "A2D2NR", "A2D2R2",
                "A2D2ER2", "A2D2NR2", "cityscapesNm2","GTAVSmE2","GTAVSmE3",
                "A2D2NR3","A2D2BNR","Kitti","TinyImageNet","TinyImageNet-ta","TinyImageNet-be", "NuScenes"]+A2D2_DATASETS_CLS+DETECTION2d_DATASETS+DATASETS_CLS+DATASETS_SEM_SEG
ALL_DATASETS +=GTA_V_STREETS_DATASETS