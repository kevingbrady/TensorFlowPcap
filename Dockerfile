FROM tensorflow/tensorflow:latest-gpu
RUN pip install tensorflow_decision_forests
RUN pip install --extra-index-url https://pypi.nvidia.com cudf_cu11
RUN pip install pandas
RUN pip install pydot
RUN pip install pydotplus
RUN pip install silence_tensorflow
RUN apt-get update
RUN apt-get -y install python3 python3-pip
RUN apt-get -y install graphviz


# Set the working directory
WORKDIR ${pwd}

#COPY . /app

# Run app.py when the container launches
CMD ["python3",  "app/main.py"]