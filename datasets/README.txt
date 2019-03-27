The image clustering datasets are accessed through the standard torchvision interface.

For segmentation, we provide our own classes to access COCO-Stuff, COCO-Stuff-3, Potsdam, Potsdam-3. These are in code/datasets/segmentation.
The scripts already do all the necessary interfacing with these classes.
Just like image clustering, however, you will need to make sure you have the raw dataset sources stored and accessible.
This file describes how to set up the segmentation datasets.

COCO-Stuff and COCO-Stuff-3
1. Run setup_cocostuff164k.sh (in the background):
nohup ./setup_cocostuff164k.sh my_CocoStuff164k_directory > download_log.out &
2. Download the curated image ids from https://www.robots.ox.ac.uk/~xuji/datasets/COCOStuff164kCurated.tar.gz
3. Untar the file, move "curated" directory to my_CocoStuff164k_directory/curated. There should now be "annotations", "curated" and "images" within my_CocoStuff164k_directory.
4. To run scripts, make sure --dataset_root is set to (absolute path of) my_CocoStuff164k_directory, and --fine_to_coarse_dict is set to (absolute path of) code/datasets/segmentation/util/out/fine_to_coarse_fict.pickle

Potsdam and Potsdam-3
Either:
1. Download the files directly: wget https://www.robots.ox.ac.uk/~xuji/datasets/Potsdam.tar.gz
2. Untar the directory. Set its path as --dataset_root for running scripts.

Or:
1. Download the raw 6000x6000 pixel satellite images from ISPRS: http://www2.isprs.org/commissions/comm3/wg4/data-request-form2.html
2. Decompress the downloaded file, and run the script code/datasets/segmentation/util/potsdam_prepare.py with SOURCE_IMGS_DIR, SOURCE_GT_DIR, OUT_DIR changed as necessary.
3. Set path of OUT_DIR as --dataset_root for running scripts.