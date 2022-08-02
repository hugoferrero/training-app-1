FROM gcr.io/deeplearning-platform-release/tf2-gpu.2-8
WORKDIR /root

# Copies the requirements.txt into the container to reduce network calls.
COPY requirements.txt .
# Installs additional packages.
RUN pip install -U -r requirements.txt

# Copies the trainer code to the docker image.
COPY . /trainer

# Sets the container working directory.
WORKDIR /trainer

ENTRYPOINT ["python", "-m", "trainer.train"]