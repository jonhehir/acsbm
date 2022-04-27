FROM python:3.10

# Add src to PYTHONPATH
ENV PYTHONPATH "${PYTHONPATH}:/workspaces/acsbm/src"

# Install dependencies
ADD requirements.txt /tmp-pip/requirements.txt
RUN pip install -r /tmp-pip/requirements.txt
RUN rm -rf /tmp-pip

# To use this container to execute, mount your local directory to the container as /workspaces/acsbm.
# Or if using VS Code, simply use the provided .devcontainer configuration.
