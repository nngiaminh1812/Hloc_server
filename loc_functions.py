import os
import sys
from flask import jsonify
import pycolmap
from collections import Counter
from config import confs_path, confs_labels

src_path = os.path.abspath("Hierarchical-Localization-Core/")
if src_path not in sys.path:
    sys.path.append(src_path)
from hloc import (
    extract_features_query_local,
    extract_features_query_global,
    match_features_query,
    pairs_from_retrieval
)
from hloc.localize_sfm import QueryLocalizer, pose_from_cluster
from hloc import extractors, logger
from hloc.utils.base_model import dynamic_load
import torch

# Config for extract and match query
retrieval_conf_loc = extract_features_query_global.confs["netvlad"]
feature_conf_loc = extract_features_query_local.confs["superpoint_aachen"]
matcher_conf_loc = match_features_query.confs["NN-superpoint"]
NUM_PAIRS = 30

# Query on global all envs to get the most common pref env
def query_global(conf_path, images_query, image_name, loc_pairs=None):
    try: 
        extract_features_query_global.main(
            retrieval_conf_loc, 
            images_query, 
            export_dir=conf_path['outputs_root'], 
            image_list=[image_name],
            overwrite=True
        )
        print("[INFO] Generating pairs from retrieval")
        pairs_result=pairs_from_retrieval.main(
            conf_path['retrieval_path'], 
            num_matched=NUM_PAIRS, 
            query_list=[image_name],
            return_rs=True
        )

        return pairs_result
    
    except Exception as e:
        print(f"[ERROR]: {str(e)}")
        return jsonify({"error": str(e)}), 500    

# Process for specific env after get the most common prefix env
def process_query(conf_path, images_query, image_name, loc_pairs=None):

    try: 
        print("[INFO] Extracting features")
        extract_features_query_local.main(
            feature_conf_loc, 
            images_query, 
            image_list=[image_name], 
            feature_path=conf_path['feature_path'],
            overwrite=True
        )
        extract_features_query_global.main(
            retrieval_conf_loc, 
            images_query, 
            export_dir=conf_path['outputs_root'], 
            image_list=[image_name],
            overwrite=True
        )
        print("[INFO] Generating pairs from retrieval")
        pairs_result=pairs_from_retrieval.main(
            conf_path['retrieval_path'], 
            num_matched=NUM_PAIRS, 
            query_list=[image_name],
            return_rs=True
        )

        print("[INFO] Matching features")
        match_features_query.main(
            matcher_conf_loc, 
            loc_pairs, 
            features=conf_path['feature_path'], 
            matches=conf_path['match_path'], 
            overwrite=True,
            read_from_file=False,
            pairs_rs=pairs_result
        )
        return pairs_result
    
    except Exception as e:
        print(f"[ERROR]: {str(e)}")
        return jsonify({"error": str(e)}), 500

def localize(points_model, conf_path, loc_pairs, image_path, image_name, pairs_rs: str = None, read_from_file=True):
    references_registered = []
    try:
        if read_from_file:
            with open(loc_pairs) as f:
                lines = f.readlines()
                for line in lines:
                    ref_name = line.split(" ")[1].strip()
                    references_registered.append(ref_name)
        else:
            lines = pairs_rs.split("\n")
            for line in lines:
                ref_name = line.split(" ")[1].strip()
                references_registered.append(ref_name)

        print("[INFO] Inferring camera from image")
        camera = pycolmap.infer_camera_from_image(image_path)
        ref_ids = []
        for r in references_registered:
            image = points_model.find_image_with_name(r)
            if image is not None:
                ref_ids.append(image.image_id)

        print(f"[INFO] Reference IDs: {ref_ids}")
        conf = {
            "estimation": {"ransac": {"max_error": 12}},
            "refinement": {"refine_focal_length": True, "refine_extra_params": True},
        }
        localizer = QueryLocalizer(points_model, conf)
        print("[INFO] Running pose from cluster")
        ret, log = pose_from_cluster(localizer, image_name, camera, ref_ids, conf_path['feature_path'], conf_path['match_path'])

        print("[INFO] Returning pose")
        pose = log['PnP_ret']['cam_from_world']
        rotation = [pose.rotation]
        translation = [pose.translation]
        return str(rotation), str(translation)
    except Exception as e:
        print(f"[ERROR]: {str(e)}")
        raise