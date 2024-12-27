import os 
import sys
from flask import Flask, request, jsonify

src_path = os.path.abspath("/run/media/nngiaminh1812/Data/unity_server/Hierarchical-Localization")
if src_path not in sys.path:
    sys.path.append(src_path)

from hloc import (
    extract_features,
    match_features,
    reconstruction,
    visualization,
    pairs_from_retrieval,
    pairs_from_exhaustive,
)

from hloc.bcef_2_localize import localize_indoor
from hloc.visualization import plot_images, read_image
from hloc.localize_sfm import QueryLocalizer, pose_from_cluster
import pycolmap
from pathlib import Path

sfm_dir_outdoor = Path('/run/media/nngiaminh1812/Data/unity_server/result/outdoor/sfm_superpoint+superglue')
model_outdoor = pycolmap.Reconstruction(sfm_dir_outdoor)

outputs_bcef_2 = Path("/run/media/nngiaminh1812/Data/unity_server/result/BCEF_Floor2/")
sfm_dir_bcef_2 = outputs_bcef_2 / "sfm_superpoint+superglue"
model_bcef_2 = pycolmap.Reconstruction(sfm_dir_bcef_2)

retrieval_conf_loc = extract_features.confs["netvlad"]
feature_conf_loc = extract_features.confs["superpoint_aachen"]
matcher_conf_loc = match_features.confs["NN-superpoint"]

app = Flask(__name__)

@app.route('/localize_outdoor', methods=['POST'])
def localize_endpoint():

    loc_pairs = Path('/run/media/nngiaminh1812/Data/unity_server/result/outdoor/pairs-loc.txt')
    feature_path = Path('/run/media/nngiaminh1812/Data/unity_server/result/outdoor/feats-superpoint-n4096-r1024.h5')
    match_path = Path('/run/media/nngiaminh1812/Data/unity_server/result/outdoor/feats-superpoint-n4096-r1024_matches-superglue_pairs-netvlad.h5')
    retrieval_path = Path('/run/media/nngiaminh1812/Data/unity_server/result/outdoor/global-feats-netvlad.h5')


    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if file:
        print(f"Received image {file.filename}")
        try:
            images_query = Path('/run/media/nngiaminh1812/Data/unity_server/query/outdoor')
            outputs = Path('/run/media/nngiaminh1812/Data/unity_server/result/outdoor')    


            image_name = file.filename
            image_path = Path(f'/run/media/nngiaminh1812/Data/unity_server/query/outdoor/{image_name}')
            print(f"Saving image to {image_path}")
            file.save(f'/run/media/nngiaminh1812/Data/unity_server/query/outdoor/{image_name}')
            
            # print("Plotting image")
            # plot_images([read_image(image_path)], dpi=75)
            print("Extracting features")
            extract_features.main(feature_conf_loc, images_query, image_list=[image_name], feature_path=feature_path, overwrite=True)
            extract_features.main(retrieval_conf_loc, images_query, export_dir=outputs, image_list=[image_name])
            print("Generating pairs from retrieval")
            pairs_from_retrieval.main(
                retrieval_path, loc_pairs, num_matched=10, query_list=[image_name]
            )

            print("Matching features")
            match_features.main(matcher_conf_loc, loc_pairs, features=feature_path, matches=match_path, overwrite=True)

            with open(loc_pairs) as f:
                lines = f.readlines()
            references_registered = []
            for line in lines:
                ref_name = line.split(" ")[1].strip()
                references_registered.append(ref_name)

            print("Inferring camera from image")
            camera = pycolmap.infer_camera_from_image(image_path)
            ref_ids = []
            for r in references_registered:
                image = model_outdoor.find_image_with_name(r)
                if image is not None:
                    ref_ids.append(image.image_id)

            print(f"Reference IDs: {ref_ids}")
            conf = {
                "estimation": {"ransac": {"max_error": 12}},
                "refinement": {"refine_focal_length": True, "refine_extra_params": True},
            }
            localizer = QueryLocalizer(model_outdoor, conf)
            print("Running pose from cluster")
            ret, log = pose_from_cluster(localizer, image_name, camera, ref_ids, feature_path, match_path)

            print("Returning pose")
            pose = log['PnP_ret']['cam_from_world']
            
            rotation = [pose.rotation]
            translation = [pose.translation]
            return jsonify({
                "rotation": str(rotation),
                "translation": str(translation)
        })
        except Exception as e:
            print(f"Error: {str(e)}")
            return jsonify({"error": str(e)}), 500

@app.route('/', methods=['GET'])
def index():
    return "Chu mi ngaaaaaa"

@app.route('/localize_bcef_2', methods=['POST'])
def bcef_floor2(): 
    sfm_pairs = outputs_bcef_2 / "pairs-netvlad.txt"
    loc_pairs = outputs_bcef_2 / 'pairs-loc.txt'
    feature_path = outputs_bcef_2 / 'feats-superpoint-n4096-r1024.h5'
    match_path = outputs_bcef_2 / 'feats-superpoint-n4096-r1024_matches-superglue_pairs-netvlad.h5'
    retrieval_path=outputs_bcef_2/'global-feats-netvlad.h5'

    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if file:
        print(f"Received image {file.filename}")
        try:
            images_query = Path('/run/media/nngiaminh1812/Data/unity_server/query/BCEF_2')  

            image_name = file.filename
            image_path = Path(f'/run/media/nngiaminh1812/Data/unity_server/query/BCEF_2/{image_name}')
            print(f"Saving image to {image_path}")
            file.save(f'/run/media/nngiaminh1812/Data/unity_server/query/BCEF_2/{image_name}')
            
            pose = localize_indoor(images_query, image_name)
            rotation = [pose.rotation]
            translation = [pose.translation]
            return jsonify({
                        "rotation": str(rotation),
                        "translation": str(translation)})
        
        except Exception as e:
            print(f"Error: {str(e)}")
            return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)