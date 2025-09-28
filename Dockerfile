
FROM python:3.9.6-slim-bullseye

WORKDIR /docker

# Install dependencies
COPY requirements.txt ./
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN pip install supervisor

# Copy project files
COPY . .

# Expose both Flask (5001) and Streamlit (8501)
EXPOSE 5001 8501

# Copy Supervisor config
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Run Supervisor (manages Flask + Streamlit)
CMD ["supervisord", "-n"]
