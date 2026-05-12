# 1. Use a slightly larger image to ensure no missing C-libraries
FROM python:3.11-slim

# 2. Set the work directory
WORKDIR /app

# 3. Install dependencies first (for better caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy only the folders we need
COPY artifacts/ ./artifacts/
COPY src/ ./src/

# 5. Expose port 5000
EXPOSE 5000

# 6. Run the app directly
CMD ["python", "src/deployment/app.py"]