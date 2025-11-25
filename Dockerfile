FROM python:3.10-slim

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py index.html ./

# Expose the port the app runs on
EXPOSE 7860

# Command to run the application with gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "app:application"]