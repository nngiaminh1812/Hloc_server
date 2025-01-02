from pathlib import Path 

# Name of output files:Use together for all environments: Oudoor, I, BCEF,...
FEATURE_NAME="feats-superpoint-n4096-r1024.h5"
MATCH_NAME="feats-superpoint-n4096-r1024_matches-superglue_pairs-netvlad.h5"
RETRIEVAL_NAME="global-feats-netvlad.h5"
MODEL_POINT_NAME="sfm_superpoint+superglue"

# Roots outputs of each models
OUTDOOR_ROOT="Hierarchical-Localization-Core/outputs/Outdoor"
BCEF_2_ROOT="Hierarchical-Localization-Core/outputs/BCEF-2"


confs_path={
    "Outdoor":{
        "outputs_root":Path(OUTDOOR_ROOT),
        "feature_path":Path("/".join([OUTDOOR_ROOT,FEATURE_NAME])),
        "match_path":Path("/".join([OUTDOOR_ROOT,MATCH_NAME])),
        "retrieval_path":Path("/".join([OUTDOOR_ROOT,RETRIEVAL_NAME])),
        "model_point":Path("/".join([OUTDOOR_ROOT,MODEL_POINT_NAME]))
    },
    "BCEF_2":{
        "outputs_root":Path(BCEF_2_ROOT),
        "feature_path":Path("/".join([BCEF_2_ROOT,FEATURE_NAME])),
        "match_path":Path("/".join([BCEF_2_ROOT,MATCH_NAME])),
        "retrieval_path":Path("/".join([BCEF_2_ROOT,RETRIEVAL_NAME])),
        "model_point":Path("/".join([BCEF_2_ROOT,MODEL_POINT_NAME]))
    }
}

confs_labels={
    "0":"Outdoor",
    "2":"BCEF_2"
}