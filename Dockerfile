FROM python:3.10

# Install dependencies
ADD requirements.txt /tmp-pip/requirements.txt
RUN pip install -r /tmp-pip/requirements.txt
RUN rm -rf /tmp-pip

# To use this container to execute, mount your local directory to the container as a volume.
# Then run files using `python run_xyz.py` in the mounted directory.
# Or if using VS Code, simply use the provided .devcontainer configuration.
