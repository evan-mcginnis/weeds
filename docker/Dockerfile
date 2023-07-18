FROM python:3.8-slim-buster

# Upgrade pip
RUN /usr/local/bin/python -m pip install --upgrade pip

# Install the libaries
RUN pip install numpy
RUN pip install pandas
RUN pip install scikit_image
RUN pip install scikit_learn
RUN pip install matplotlib
RUN pip install pyyaml
RUN pip install more-itertools

RUN mkdir -p /app
WORKDIR /app
COPY ./lib /app
COPY ./jetson /app

ENV PYTHONPATH "${PYTHONPATH}:/app/lib"
#CMD ["ls",  "-l",  "/mnt"]
#CMD ["python3", "/app/Selection.py",  "-df /mnt/data/normalized.csv", "--selection ALL", "-f 10", "-o"]
CMD python3 /app/Selection.py -df /mnt/data/normalized.csv --selection ALL -f 10 -o
