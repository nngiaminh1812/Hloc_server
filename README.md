# Utilizing Augmented Reality combined with Indoor Positioning for Navigation Systems
## Project Summary
This project develops an indoor navigation system for the University of Science (HCMUS) using Augmented Reality (AR) combined with Visual Positioning System (VPS).

Unlike GPS, which performs poorly indoors, the system uses visual data and Hierarchical Localization (HLoc) to determine accurate user positions. AR arrows are displayed on the phone screen to guide users in real time.

The solution helps students and visitors easily find classrooms or offices and can be extended to other environments such as hospitals or malls.

## System Structure

![(Poster)](https://github.com/nngiaminh1812/Hloc_server/blob/main/img/Final_Poster_-resized-1.png)

**1. Positioning Module:** Handles indoor localization using VPS combined with Hierarchical Localization (HLoc).
- Offline Phase: Collects images of the campus (buildings I, B, C) using a Canon EOS 700D camera, builds a 3D point cloud with COLMAP, and extracts global/local features via SuperPoint for storage in a database.
- Online Phase: User captures an image via app camera; sent to a Flask (Python) server for processing. HLoc pipeline: (1) Retrieves reference images, (2) Matches features using SuperGlue or LightGlue, (3) Estimates pose with PyColmap, returning 3D coordinates (x, y, z) and rotation as JSON. Deployed with Docker and Gunicorn for efficiency.
  
**2. Intermediate Linking Module:** Bridges the positioning and navigation modules by transforming coordinates.

**3. Navigation Module:** Provides AR-based guidance using Unity engine with AR Foundation.
- Maps transformed user position onto a scaled 3D campus model.
- Uses NavMesh (Unity AI Navigation) to model walkable areas (e.g., corridors, stairs) as a graph with cost-weighted nodes (higher costs for stairs to prioritize optimal paths).
- Computes shortest path with A* algorithm.
- Displays real-time AR overlays (3D arrows or virtual "Pet AR" guides), minimap for overview, and interactive info (e.g., room descriptions).
- Integrates device sensors (gyroscope, accelerometer) for continuous position/heading updates during movement.

# Hierarchical Localization Server
## 1. Directory structure 
    HLoc_server/
    ├── main.py
    ├── config.py
    ├── loc_funtions.py
    ├── hloc/
    ├── query/
    └── Hierarchical-Localization-Core/
        ├── hloc/
        ├── outputs/
        ├── pairs/
        ├── query/
        ├── thirty_party/
        ├── outputs/
## 2. How to use API
## 2.1 Create HLOC server
Navigate to root `Hloc_server`, run this code to run HLOC server on localhost:
```
gunicorn --workers 2 --threads 2 --bind 0.0.0.0:5000 wsgi:app
```
## 2.2 Using on other server
In another server, use this code get **translation** and **rotation** results with `bash` cmd:
```
 curl.exe -X POST -F "file=@path\of\your\image" -F "label=x"  http://localhost:5000/localize
```
- `x`: is label of model was built. To get label information, refer to the `config.py` file.

### Docker CMD:
Build image:
```
docker build -t final-flask-app .
```
List docker images:
```
docker images
```
Run docker image:
```
docker run -p 3185:3185 flask-app
```
Delete all images, containers, cache, ..
```
docker system prune -a
```
Rename docker image:
```
docker tag final-flask-app:latest puffycheeks/flask-app:latest
```
Push image to docker hub:
```
docker push username/name_img:tag

```
