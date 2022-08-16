FROM seal-python

# define the folder where our src should exist/ be deposited
ARG SRC=/ipython-notebook
RUN mkdir -p ${SRC}

# copy into container requirements and install them before rest of code
COPY ./requirements.txt ${SRC}/.
RUN pip3 install -r ${SRC}/requirements.txt

RUN mkdir /runtime && chmod 777 /runtime
RUN mkdir /notebooks && chmod 777 /notebooks
CMD ["jupyter-notebook", "--ip=0.0.0.0", "--notebook-dir=/notebooks"]
