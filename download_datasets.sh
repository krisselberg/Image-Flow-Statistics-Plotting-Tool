mkdir data

mkdir data/Middlebury
cd data/Middlebury/
wget https://vision.middlebury.edu/flow/data/comp/zip/other-color-allframes.zip
unzip other-color-allframes.zip
wget https://vision.middlebury.edu/flow/data/comp/zip/other-gt-flow.zip
unzip other-gt-flow.zip
rm *.zip
cd ../..

mkdir data/Sintel
cd data/Sintel
wget http://files.is.tue.mpg.de/sintel/MPI-Sintel-training_images.zip
unzip MPI-Sintel-training_images.zip
wget http://files.is.tue.mpg.de/sintel/MPI-Sintel-training_extras.zip
unzip MPI-Sintel-training_extras.zip
rm *.zip
cd ../..