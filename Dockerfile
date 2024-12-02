# Use a base Python image
FROM python:3.10-slim

# Set the working directory
WORKDIR /benchmark_atlas/.

# Copy the Python script
COPY benchmark_atlas/test.py .

# Set environment variables (optional, can also be set dynamically)
ENV UPLOAD_ID=""
ENV PATH=""
ENV FILETYPE=""

# RUN pip install --no-cache-dir -r requirements.txt

# Run the Python script when the container starts
CMD ["python", "test.py"]


