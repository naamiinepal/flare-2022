FROM python:3.8.10-slim

# Create directories in container
CMD mkdir -p /workspace/inputs /workspace/outputs
WORKDIR /workspace

# Install requirements
COPY requirements.txt ./
RUN pip install --no-cache-dir -qr requirements.txt

# Copy Codes
COPY predict_scratch.py saver.py predict.sh ./

# Copy checkpoints
COPY abdomen_checkpoint.pt ./