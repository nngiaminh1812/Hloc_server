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