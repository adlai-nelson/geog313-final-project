# Final project for Advanced Geospatial with Python

Adlai Nelson



## Instructions for Use

Pull the data from my Github Repository

`git clone https://github.com/adlai-nelson/geog313-final-project`

navigate to folder where files were downloaded to

`cd geog313-final-project`

Create an image using the included dockerfile

`docker build -t forestmonitoring_analysis  .`

Run the image using 

`docker run -v $(pwd):/home/gisuser -it -p 8888:8888 -p 8787:8787 forestmonitoring_analysis`

Paste the jupyterlab link into your local web browser

Open analysis.ipynb to run the analysis
