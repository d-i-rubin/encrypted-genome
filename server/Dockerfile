FROM seal-python

# define the folder where our src should exist/ be deposited
ARG SRC=/server
RUN mkdir -p ${SRC}

# copy into container requirements and install them before rest of code
COPY ./requirements.txt ${SRC}/.
RUN pip3 install -r ${SRC}/requirements.txt

COPY ./server.py ${SRC}/.

CMD ["python3", "/server/server.py", "--model_coef_file", "/data/private/logmodel_coef.dump"]
