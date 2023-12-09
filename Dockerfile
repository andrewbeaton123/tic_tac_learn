FROM python:3.8

WORKDIR /usr/src/app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD [ "python", "./learning/mc_2.py" , "-e", "PYTHONUNBUFFERED=1"]