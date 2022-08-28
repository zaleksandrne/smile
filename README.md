# SMILE



## build
docker build . -t smile 

## run
docker run -v ~/dev/smile/data:/code/data -e INPUT_FOLDER='data/input_data' -e OUTPUT_FOLDER='data/output_data' -it smile
