from pathlib import Path 

# Name of output files: Use together for all environments: Outdoor, I, BCEF,...
FEATURE_NAME="feats-superpoint-n4096-r1024.h5"
MATCH_NAME="feats-superpoint-n4096-r1024_matches-superglue_pairs-netvlad.h5"
RETRIEVAL_NAME="global-feats-netvlad.h5"
MODEL_POINT_NAME="sfm_superpoint+superglue"

# Roots outputs of each models
OUTDOOR_ROOT="Hierarchical-Localization-Core/outputs/Outdoor"
BCEF_2_ROOT="Hierarchical-Localization-Core/outputs/BCEF/Floor2/outputs"
I_ROOT="Hierarchical-Localization-Core/outputs/I"
GLOBAL_ROOT="Hierarchical-Localization-Core/outputs/global-features"

confs_path={
    "O":{
        "outputs_root":Path(OUTDOOR_ROOT),
        "feature_path":Path("/".join([OUTDOOR_ROOT,FEATURE_NAME])),
        "match_path":Path("/".join([OUTDOOR_ROOT,MATCH_NAME])),
        "retrieval_path":Path("/".join([OUTDOOR_ROOT,RETRIEVAL_NAME])),
        "model_point":Path("/".join([OUTDOOR_ROOT,MODEL_POINT_NAME]))
    },
    "BCEF2":{
        "outputs_root":Path(BCEF_2_ROOT),
        "feature_path":Path("/".join([BCEF_2_ROOT,FEATURE_NAME])),
        "match_path":Path("/".join([BCEF_2_ROOT,MATCH_NAME])),
        "retrieval_path":Path("/".join([BCEF_2_ROOT,RETRIEVAL_NAME])),
        "model_point":Path("/".join([BCEF_2_ROOT,MODEL_POINT_NAME]))
    },
    "I_G":{
        "outputs_root":Path("/".join([I_ROOT, "G - j4t"])),
        "feature_path":Path("/".join([I_ROOT, "G - j4t", FEATURE_NAME])),
        "match_path":Path("/".join([I_ROOT, "G - j4t", MATCH_NAME])),
        "retrieval_path":Path("/".join([I_ROOT, "G - j4t", RETRIEVAL_NAME])),
        "model_point":Path("/".join([I_ROOT, "G - j4t", MODEL_POINT_NAME]))
    },
    "I2":{
        "outputs_root":Path("/".join([I_ROOT, "Floor2/2"])),
        "feature_path":Path("/".join([I_ROOT, "Floor2/2", FEATURE_NAME])),
        "match_path":Path("/".join([I_ROOT, "Floor2/2", MATCH_NAME])),
        "retrieval_path":Path("/".join([I_ROOT, "Floor2/2", RETRIEVAL_NAME])),
        "model_point":Path("/".join([I_ROOT, "Floor2/2", MODEL_POINT_NAME]))
    },
    "I3":{
        "outputs_root":Path("/".join([I_ROOT, "Floor3/3"])),
        "feature_path":Path("/".join([I_ROOT, "Floor3/3", FEATURE_NAME])),
        "match_path":Path("/".join([I_ROOT, "Floor3/3", MATCH_NAME])),
        "retrieval_path":Path("/".join([I_ROOT, "Floor3/3", RETRIEVAL_NAME])),
        "model_point":Path("/".join([I_ROOT, "Floor3/3", MODEL_POINT_NAME]))
    },
    "I4":{
        "outputs_root":Path("/".join([I_ROOT, "Floor4/4"])),
        "feature_path":Path("/".join([I_ROOT, "Floor4/4", FEATURE_NAME])),
        "match_path":Path("/".join([I_ROOT, "Floor4/4", MATCH_NAME])),
        "retrieval_path":Path("/".join([I_ROOT, "Floor4/4", RETRIEVAL_NAME])),
        "model_point":Path("/".join([I_ROOT, "Floor4/4", MODEL_POINT_NAME]))
    },
    "I5":{
        "outputs_root":Path("/".join([I_ROOT, "Floor5/5"])),
        "feature_path":Path("/".join([I_ROOT, "Floor5/5", FEATURE_NAME])),
        "match_path":Path("/".join([I_ROOT, "Floor5/5", MATCH_NAME])),
        "retrieval_path":Path("/".join([I_ROOT, "Floor5/5", RETRIEVAL_NAME])),
        "model_point":Path("/".join([I_ROOT, "Floor5/5", MODEL_POINT_NAME]))
    },
    "I6":{
        "outputs_root":Path("/".join([I_ROOT, "Floor6/6"])),
        "feature_path":Path("/".join([I_ROOT, "Floor6/6", FEATURE_NAME])),
        "match_path":Path("/".join([I_ROOT, "Floor6/6", MATCH_NAME])),
        "retrieval_path":Path("/".join([I_ROOT, "Floor6/6", RETRIEVAL_NAME])),
        "model_point":Path("/".join([I_ROOT, "Floor6/6", MODEL_POINT_NAME]))
    },
    "I7":{
        "outputs_root":Path("/".join([I_ROOT, "Floor7/7"])),
        "feature_path":Path("/".join([I_ROOT, "Floor7/7", FEATURE_NAME])),
        "match_path":Path("/".join([I_ROOT, "Floor7/7", MATCH_NAME])),
        "retrieval_path":Path("/".join([I_ROOT, "Floor7/7", RETRIEVAL_NAME])),
        "model_point":Path("/".join([I_ROOT, "Floor7/7", MODEL_POINT_NAME]))
    },
    "I8":{
        "outputs_root":Path("/".join([I_ROOT, "Floor8/8"])),
        "feature_path":Path("/".join([I_ROOT, "Floor8/8", FEATURE_NAME])),
        "match_path":Path("/".join([I_ROOT, "Floor8/8", MATCH_NAME])),
        "retrieval_path":Path("/".join([I_ROOT, "Floor8/8", RETRIEVAL_NAME])),
        "model_point":Path("/".join([I_ROOT, "Floor8/8", MODEL_POINT_NAME]))
    },
    "I9":{
        "outputs_root":Path("/".join([I_ROOT, "Floor9/9"])),
        "feature_path":Path("/".join([I_ROOT, "Floor9/9", FEATURE_NAME])),
        "match_path":Path("/".join([I_ROOT, "Floor9/9", MATCH_NAME])),
        "retrieval_path":Path("/".join([I_ROOT, "Floor9/9", RETRIEVAL_NAME])),
        "model_point":Path("/".join([I_ROOT, "Floor9/9", MODEL_POINT_NAME]))
    },
    "I10":{
        "outputs_root":Path("/".join([I_ROOT, "Floor10"])),
        "feature_path":Path("/".join([I_ROOT, "Floor10", FEATURE_NAME])),
        "match_path":Path("/".join([I_ROOT, "Floor10", MATCH_NAME])),
        "retrieval_path":Path("/".join([I_ROOT, "Floor10", RETRIEVAL_NAME])),
        "model_point":Path("/".join([I_ROOT, "Floor10", MODEL_POINT_NAME]))
    },
    "I11":{
        "outputs_root":Path("/".join([I_ROOT, "Floor11/11"])),
        "feature_path":Path("/".join([I_ROOT, "Floor11/11", FEATURE_NAME])),
        "match_path":Path("/".join([I_ROOT, "Floor11/11", MATCH_NAME])),
        "retrieval_path":Path("/".join([I_ROOT, "Floor11/11", RETRIEVAL_NAME])),
        "model_point":Path("/".join([I_ROOT, "Floor11/11", MODEL_POINT_NAME]))
    },
    "global":{
        "outputs_root":Path(GLOBAL_ROOT),
        "retrieval_path":Path("/".join([GLOBAL_ROOT, RETRIEVAL_NAME])),
    }
}

confs_labels={
    "O":"O",
    "BCEF2":"BCEF2",
    "I_G":"I_G",
    "I2":"I2",
    "I3":"I3",
    "I4":"I4",
    "I5":"I5",
    "I6":"I6",
    "I7":"I7",
    "I8":"I8",
    "I9":"I9",
    "I10":"I10",
    "I11":"I11"
}