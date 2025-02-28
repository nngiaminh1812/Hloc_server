from flask import Flask, request, jsonify
import pycolmap
from pathlib import Path
import config
import loc_functions
import re
import os
import sys
from collections import Counter

# Add sys
src_path = os.path.abspath("Hierarchical-Localization-Core/")
if src_path not in sys.path:
    sys.path.append(src_path)

# Init app
app = Flask(__name__)

# Define folder path contains query images of users
images_query = Path('query')

# Temp
loc_pairs = Path('Hierarchical-Localization-Core/outputs/Outdoor/pairs-loc.txt')

# Load point cloud models
outdoor_model = pycolmap.Reconstruction(config.confs_path['O']['model_point'])
bcef_2_model = pycolmap.Reconstruction(config.confs_path['BCEF2']['model_point'])

# Load models for each floor of building I
i_models = {
    f"I{i}": pycolmap.Reconstruction(config.confs_path[f"I{i}"]['model_point'])
    for i in range(2, 12)
}
i_models["I_G"] = pycolmap.Reconstruction(config.confs_path['I_G']['model_point'])

# Combine all models
points_model = {'O': outdoor_model, 'BCEF2': bcef_2_model}
points_model.update(i_models)

@app.route('/localize', methods=['POST'])
def localize_endpoint():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    image_name = file.filename
    if file:
        print(f"[INFO] Received image {image_name}")
        image_path = images_query / image_name
        file.save(image_path)

        try:
            # Use global model for initial retrieval
            global_model = config.confs_path['global']
            pairs_rs = loc_functions.query_global(global_model, images_query, image_name)
            print(pairs_rs)
            # Determine the most common environment
            lines = [line for line in pairs_rs.split("\n") if line.strip()]
            env_prefixes = [line.split(" ")[1].split("_")[0] for line in lines]
            most_common_env = Counter(env_prefixes).most_common(1)[0][0]
            print(f"[INFO] Using model {most_common_env}")
            # Use the specific environment model for localization
            HLOC_model = config.confs_path[config.confs_labels[most_common_env]]
            
            pairs_rs_env=loc_functions.process_query(HLOC_model,images_query,image_name,loc_pairs)
        
            # Get data 
            rotation,translation=loc_functions.localize(points_model[most_common_env],HLOC_model,loc_pairs,image_path,image_name,pairs_rs_env,read_from_file=False)
            return jsonify({
                "rotation": rotation,
                "translation": translation
            })
        
        except Exception as e:
            print(f"[ERROR]: {str(e)}")
            return jsonify({"error": str(e)}), 500    

@app.route('/', methods=['GET'])
def index():
    return "Chu mi ngaaaaaa"

if __name__ == '__main__':
    app.run(debug=True)