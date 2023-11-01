#FROM nvcr.io/nvidia/rapidsai/base:23.08-cuda11.2-py3.10
FROM nvcr.io/nvidia/tensorflow:23.09-tf2-py3
LABEL Author, Kevin Brady Jr

# Set the working directory
WORKDIR /

#COPY . /app

#---------------- Prepare the environment
COPY requirements.txt .
RUN pip install -r requirements.txt
RUN apt-get update
RUN apt-get install -y graphviz

# Run app.py when the container launches
CMD ["python3",  "app/main.py"]
