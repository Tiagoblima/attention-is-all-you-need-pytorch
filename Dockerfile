# set base image (host OS)
FROM python:3.8

# set the working directory in the container
WORKDIR ./

# copy the dependencies file to the working directory
COPY requirements.txt .

# install dependencies
RUN pip install -r requirements.txt

# copy the content of the local src directory to the working directory
COPY attention-is-all-you-need-pytorch .

# command to run on container start
RUN config.sh
RUN run_exp.sh
RUN run_exp2.sh
RUN run_exp3.sh
