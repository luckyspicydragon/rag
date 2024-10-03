# nvidia pytorch container
FROM nvcr.io/nvidia/pytorch:24.06-py3

# Upgrade pip
RUN python -m pip install --upgrade pip 

# Install git
RUN apt-get update && apt-get install -y git

COPY . /app/
WORKDIR /app/

RUN pip install \
pypdf \
langchain \
chromadb \
python-dotenv \
langchain-community \
langchain-core \
openai \
tiktoken