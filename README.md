# SMILE



## build
docker build . -t smile 

## run
docker run -v ~/dev/smile/output_data:/code/output_data -e FOLDER='output_data' -it smile 