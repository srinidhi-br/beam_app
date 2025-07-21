# Use a slim Python image
FROM python:3.10-slim

# Install system tools & build‑time deps for OpenSees
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential \
      wget \
      cmake \
      python3-dev \
      git \
    && rm -rf /var/lib/apt/lists/*

# Fetch and build the OpenSees core library
RUN git clone https://github.com/zhuminjie/OpenSees.git /opt/OpenSees \
 && mkdir /opt/OpenSees/build \
 && cd /opt/OpenSees/build \
 && cmake .. \
 && make -j4 install

# Set working dir
WORKDIR /app

# Copy requirements and install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your app code
COPY beam_app4_with_units_and_I.py .

# Expose Streamlit port
EXPOSE 8501

# Launch Streamlit
CMD ["streamlit", "run", "beam_app4_with_units_and_I.py", "--server.address=0.0.0.0"]
