# Use official Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements first (for caching)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy entire project
COPY . .

# Expose Streamlit default port
EXPOSE 8080

# Run Streamlit app
CMD streamlit run fwa/app.py --server.port=8080 --server.address=0.0.0.0
