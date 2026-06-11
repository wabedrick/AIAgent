# Use an official lightweight Python 3.11 image
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Copy just the requirements first to leverage Docker caching
COPY requirements.txt .

# Install dependencies cleanly without caching junk
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r requirements.txt

# Copy the rest of your agent code into the container
COPY . .

# Expose the port FastAPI will run on
EXPOSE 10000

# Start the Uvicorn server matching Render's expected port
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000"]