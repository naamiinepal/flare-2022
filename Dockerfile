FROM python:3.8-slim

# Create directories in container
CMD mkdir -p /workspace/inputs /workspace/outputs
WORKDIR /workspace

# Install requirements
COPY docker_requirements.txt requirements.txt
RUN pip install --no-cache-dir -qr requirements.txt

# Copy Codes
COPY __init__.py predict_c2f_scratch.py saver.py predict.sh ./
COPY ./custom_transforms /workspace/custom_transforms

# Copy checkpoints
COPY coarse_boundingmask_semi.ts fine_flare_model_final.ts ./
