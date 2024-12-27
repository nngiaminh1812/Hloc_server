from pathlib import Path
import pycolmap
from hloc import (
    extract_features,
    match_features,
    pairs_from_retrieval,
)
from hloc.localize_sfm import QueryLocalizer, pose_from_cluster
import argparse
# import tqdm, tqdm.notebook
# tqdm.tqdm = tqdm.notebook.tqdm  # notebook-friendly progress bars


# Change the paths to the dataset and the outputs for your own setup
images = Path("datasets/BCDEF_2/db/")
outputs = Path("/run/media/nngiaminh1812/Data/unity_server/result/BCEF_Floor2/")
sfm_pairs = outputs / "pairs-netvlad.txt"
sfm_dir = outputs / "sfm_superpoint+superglue"
loc_pairs = outputs / 'pairs-loc.txt'
feature_path = outputs / 'feats-superpoint-n4096-r1024.h5'
match_path = outputs / 'feats-superpoint-n4096-r1024_matches-superglue_pairs-netvlad.h5'
retrieval_path=outputs/'global-feats-netvlad.h5'


retrieval_conf_loc = extract_features.confs["netvlad"]
feature_conf_loc = extract_features.confs["superpoint_aachen"]
matcher_conf_loc = match_features.confs["NN-superpoint"]



def localize_indoor(query_parent_path:Path,query_name:str):
    
    # Load model 
    model = pycolmap.Reconstruction(sfm_dir)
    
    # Extract local features 
    extract_features.main(
        feature_conf_loc,
        query_parent_path, 
        image_list=[query_name],
        feature_path=feature_path,
        overwrite=True
    )
    
    # Extract global features
    extract_features.main(
        retrieval_conf_loc, 
        query_parent_path, 
        export_dir=outputs,
        image_list=[query_name],
        overwrite=True
    )
    
    # Generate pairs for matching
    pairs_from_retrieval.main(
        retrieval_path, 
        loc_pairs, 
        num_matched=10, 
        query_list=[query_name]
    )

    # Match the features
    match_features.main(
        matcher_conf_loc,
        loc_pairs, 
        features=feature_path, 
        matches=match_path, 
        overwrite=True
    )
    
    # Get images reference in database for localization
    with open(loc_pairs) as f:
        lines=f.readlines()
        references_registered=[line.split()[1].strip() for line in lines]
    
    # Get id of images in database in model 
    ref_ids=[model.find_image_with_name(image).image_id for image in references_registered \
             if model.find_image_with_name(image) is not None]
    
    
    conf = {
    "estimation": {"ransac": {"max_error": 12}},
        "refinement": {"refine_focal_length": True, "refine_extra_params": True},
    }
    localizer = QueryLocalizer(model, conf)
    camera = pycolmap.infer_camera_from_image(query_parent_path/query_name)
    ret, log = pose_from_cluster(localizer, query_name, camera, ref_ids, feature_path, match_path)
    
    return log['PnP_ret']['cam_from_world']

# def main():
#     parser = argparse.ArgumentParser(description="Example of optional arguments")
#     parser.add_argument("--folder_query", type=str, help="Your folder contains all query images.",required=True) 
#     parser.add_argument("--query_name", type=str, help="Query name you want to localize.",required=True,default='1.jpg')
#     args = parser.parse_args()
#     rs=localize_indoor(Path(args.folder_query),args.query_name)
#     return rs
    
# if __name__ == "__main__":
#     print(main())