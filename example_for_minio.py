import numpy as np
from PIL import Image

from src.minio_client import MinIOClient, MINIO_ACCESS_KEY, MINIO_ENDPOINT, MINIO_SECRET, MINIO_BUCKET


minio = MinIOClient(MINIO_ENDPOINT, MINIO_ACCESS_KEY, MINIO_SECRET)

script_type = "download" # [download, upload]


if script_type == "download":
    # image_name = "DJI_20211023131738_0029_Z.jpg"
    image_name = "DJI_20240604132000_0021_V.JPG"
    # image_name = "fhhi_test.png"


    img = minio.download_image(MINIO_BUCKET, image_name)
    print(np.asarray(Image.fromarray(img)).shape)
    img = Image.fromarray(img)
    img.show()

elif script_type == "upload":
    image_name = "fhhi_test.png"
    file_name = "data/person_car_detection_data/148_test_images/images/val/0bedd9b6-image_252.jpg" 

    minio.upload_file(MINIO_BUCKET, image_name, file_name)

