from flask import Flask, request, jsonify
import pycolmap
from pathlib import Path
import config
import loc_functions,keypoints_functions
import re
import os
import sys
from collections import Counter
src_path = os.path.abspath("Hierarchical-Localization-Core/")
if src_path not in sys.path:
    sys.path.append(src_path)


# Init app
app = Flask(__name__)


#===================================================KEYPOINTS_CONFIG=========================================================

conf_local_feature=keypoints_functions.get_conf_local_feature()
model_local=keypoints_functions.load_model(conf_local_feature)
TMP_DIR_FRAME = "/tmp/frame"   #Using for store frame 
THRESHOLD_KEYPOINTS=500        #Threshold for select frame have enough keypoints to localize

#===================================================LOCALIZE_CONFIG=========================================================
# Define folder path contains query images of users
TMP_DIR_QUERY = "/tmp/query"

# Temp
loc_pairs = Path('Hierarchical-Localization-Core/outputs/Outdoor/pairs-loc.txt')

# # Load point cloud models
# outdoor_model = pycolmap.Reconstruction(config.confs_path['O']['model_point'])
# bcef_2_model = pycolmap.Reconstruction(config.confs_path['BCEF2']['model_point'])
# # Load models for each floor of building I
# i_models = {
#     f"I{i}": pycolmap.Reconstruction(config.confs_path[f"I{i}"]['model_point'])
#     for i in range(2, 12)
# }


# # Combine all models
# points_model = {'O': outdoor_model, 'BCEF2': bcef_2_model}
# points_model.update(i_models)


bcef_2_model = pycolmap.Reconstruction(config.confs_path['BCEF2']['model_point'])
points_model = { 'BCEF2': bcef_2_model}


@app.route('/localize', methods=['POST'])
def localize_endpoint():
    os.makedirs(TMP_DIR_QUERY,exist_ok=True)
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    image_name = file.filename
    if file:
        print(f"[INFO] Received image {image_name}")
        image_path = Path(TMP_DIR_QUERY) / image_name
        file.save(image_path)

        try:
            # Use global model for initial retrieval
            global_model = config.confs_path['global']
            pairs_rs = loc_functions.query_global(global_model, Path(TMP_DIR_QUERY), image_name)
            # Determine the most common environment
            lines = [line for line in pairs_rs.split("\n") if line.strip()]
            env_prefixes = [line.split(" ")[1].split("_")[0] for line in lines]
            most_common_env = Counter(env_prefixes).most_common(1)[0][0]
            print(f"[INFO] Using model {most_common_env}")
            # Use the specific environment model for localization
            HLOC_model = config.confs_path[config.confs_labels[most_common_env]]
            
            pairs_rs_env=loc_functions.process_query(HLOC_model,Path(TMP_DIR_QUERY),image_name)
            # Get data 
            rotation,translation=loc_functions.localize(points_model[most_common_env],HLOC_model,loc_pairs,image_path,image_name,pairs_rs_env,read_from_file=False)

            # Delete query image store in tmp folder 
            os.remove(image_path)

            # Delete query key in feature_path,match_path,retrieval_path 
            feature_path,match_path,retrieval_path=HLOC_model["feature_path"],HLOC_model["match_path"],HLOC_model["retrieval_path"]
            loc_functions.delete_query_h5(feature_path,match_path,retrieval_path,image_name)


            return jsonify({
                "rotation": rotation,
                "translation": translation
            })
        
        except Exception as e:
            print(f"[ERROR]: {str(e)}")
            return jsonify({"error": str(e)}), 500    


@app.route('/keypoints',methods=['POST'])
def keypoints_endpoints():
    os.makedirs(TMP_DIR_FRAME,exist_ok=True)
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    image_name = file.filename
    if file:
        print(f"[INFO] Received image {image_name}")
        image_path=os.path.join(TMP_DIR_FRAME,image_name)
        file.save(image_path)
        print(f"[INFO] Saved image to tmp folder with path {image_path}")
    try:

        results=keypoints_functions.get_keypoints_utils(TMP_DIR_FRAME,image_name,conf_local_feature,model_local)
        keypoints=results['keypoints']

        # Handle the frame is selected by number of keyoints
        # Code here and send to client
        
        num_keypoints=len(keypoints)
        can_query=True
        if num_keypoints<THRESHOLD_KEYPOINTS:
            can_query=False

        os.remove(image_path)
        return jsonify({
            "key_points": keypoints.astype(float).tolist(),
            "can_query":can_query
        })
    except Exception as e:
            print(f"[ERROR]: {str(e)}")
            return jsonify({"error": str(e)}), 500   
@app.route('/', methods=['GET'])
def index():
    return "Chu mi ngaaaaaa"

if __name__ == '__main__':
    app.run(debug=True)