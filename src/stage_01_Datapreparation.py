import argparse
import os
import shutil
from tqdm import tqdm
import logging
from src.utils.common import read_yaml, create_directories
import random
import cv2
from PIL import Image


STAGE = "Data Preparation" ## <<< change stage name 

logging.basicConfig(
    filename=os.path.join("logs", 'running_logs.log'), 
    level=logging.INFO, 
    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
    filemode="a"
    )


def main(config_path, params_path):
    try:
        ## read config files
        config = read_yaml(config_path)
        params = read_yaml(params_path)
        
        #Load first n number of images (to train on a subset of all images)
        #For demo purposes, let us use 5000 images
        logging.info(params)
        logging.info(params['train']['no_training'])
        n=params['train']['no_training']

        #local drive location
        data_dir = config['local_data_dirs']['data_dir']
        raw_dir = config['local_data_dirs']['raw_data']
        raw_data = os.path.join(data_dir,raw_dir) # raw data
        
        process_data = config['local_data_dirs']['process_data']
        process_dir = os.path.join(data_dir,process_data) # processed data dir

        hr_img_size = params['train']['hr_img_size']
        hr_img_size=tuple(map(int,hr_img_size.split(",")))
        lr_img_size = params['train']['lr_img_size']
        lr_img_size=tuple(map(int,lr_img_size.split(",")))
    except Exception as e:
        logging.exception(e)
        raise e    
    #high resolution and low resolution directory
    hr = process_dir+"/hr_images/"
    lr = process_dir+"/lr_images/"
    create_directories([hr,lr])

    for img in os.listdir(raw_data):
        try:
            file_name = raw_data+'\\'+img
            logging.info(f"file name : {file_name}")
            im=Image.open(file_name)
            im.verify()
            logging.info(f"file name : {file_name} is verified")
            # if file_name.lower().endwith(('.png', '.jpg', '.jpeg')):
            
            img_array = cv2.imread(file_name)
            logging.info(f"resize for {file_name} is started ")
            
            img_array = cv2.resize((img_array),(hr_img_size))
            lr_img_array = cv2.resize(img_array,lr_img_size)

            logging.info(f"resize for {file_name} is completed")

            cv2.imwrite(process_dir+"/hr_images/"+img,img_array)
            cv2.imwrite(process_dir+"/lr_images/"+img,lr_img_array)
        except Exception as e:
            logging.exception(f"expection occur while reading {file_name} \n expection is {e}")
            
if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c", default="configs/config.yaml")
    args.add_argument("--params", "-p", default="params.yaml")
    parsed_args = args.parse_args()

    try:
        logging.info("\n********************")
        logging.info(f">>>>> stage {STAGE} started <<<<<")
        main(config_path=parsed_args.config,params_path=parsed_args.params)
        logging.info(f">>>>> stage {STAGE} completed!<<<<<\n")
    except Exception as e:
        logging.exception(e)
        raise e