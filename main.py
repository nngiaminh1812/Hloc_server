from flask import Flask, request, jsonify
import pycolmap
from pathlib import Path
import config
import loc_functions
import re
import os 
import sys

# Add sys 
src_path = os.path.abspath("Hierarchical-Localization-Core/")
if src_path not in sys.path:
    sys.path.append(src_path)
# from hloc import (
#     extract_features_query,
# )

# Init app
app = Flask(__name__)


# Define folder path contains query images of users
images_query=Path(f'query')

# Temp
loc_pairs=Path('Hierarchical-Localization-Core/outputs/Outdoor/pairs-loc.txt')

# Load point cloud models 
outdoor_model = pycolmap.Reconstruction(config.confs_path['Outdoor']['model_point'])
bcef_2_model = pycolmap.Reconstruction(config.confs_path['BCEF_2']['model_point'])

points_model = {'0':outdoor_model, '2':bcef_2_model}
# Load built model extract feature and match feature

@app.route('/localize', methods=['POST'])
def localize_endpoint():
    
    if 'file' not in request.files or 'label' not in request.form:
        return jsonify({"error": "No file or label of model part"}), 400
    
    file = request.files['file']
    label=request.form.get('label')

    if file.filename == '' or label =='':
        return jsonify({"error": "No selected file or label of model"}), 400
    
    image_name = file.filename
    if file:
        print(f"[INFO] Received image {image_name}")
        image_path =images_query/image_name

        print(f"[INFO] Saving image to {image_path}")
        file.save(image_path)

        print(f"[INFO] Received label {str(label)}")

        # Verify label 
        label=str(label)
        pattern_label=r'^[0-9]\d*$'
        if not re.match(pattern_label,label) or label not in config.confs_labels.keys():
            return jsonify({"error": "Invalid label of model"}), 400
        
        # Define model HLOC is used
        HLOC_model=config.confs_path[config.confs_labels[label]]

        # Extract global, local feature and match feature.
        pairs_rs=loc_functions.process_query(HLOC_model,images_query,image_name,loc_pairs)
        
        # Get data 
        rotation,translation=loc_functions.localize(points_model[label],HLOC_model,loc_pairs,image_path,image_name,pairs_rs,read_from_file=False)
        return jsonify({
            "rotation": rotation,
            "translation": translation
        })
@app.route('/', methods=['GET'])
def index():
    return "Chu mi ngaaaaaa"


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)