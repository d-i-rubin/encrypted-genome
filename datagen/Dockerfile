FROM seal-python

# define the folder where our src should exist/ be deposited
ARG SRC=/datagen
RUN mkdir -p ${SRC}

# copy into container requirements and install them before rest of code
COPY ./requirements.txt ${SRC}/.
RUN pip3 install -r ${SRC}/requirements.txt

COPY ./generate_model_data.py ${SRC}/.

CMD ["python3", "/datagen/generate_model_data.py", "--data_file", "/Challenge/Challenge.fa", "--out_dir", "/data"]
