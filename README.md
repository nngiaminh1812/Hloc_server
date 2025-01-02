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
python3 main.py
```
## 2.2 Using on other server
In another server, use this code get **translation** and **rotation** results with `bash` cmd:
```
 curl.exe -X POST -F "file=@path\of\your\image" -F "label=x"  http://localhost:5000/localize
```
- `x`: is label of model was built. To get label information, refer to the `config.py` file.

