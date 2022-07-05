FROM python:3.8-slim

# Create directories in container
CMD mkdir -p /workspace/inputs /workspace/outputs
WORKDIR /workspace

# Install requirements
COPY docker_requirements.txt requirements.txt
RUN pip install --no-cache-dir -qr requirements.txt

# Copy Codes
COPY predict_c2f_scratch.py saver.py predict.sh custom_transforms.py ./

# Copy checkpoints
COPY coarse_flare_model.ts fine_flare_model.ts ./
