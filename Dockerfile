FROM ubuntu:22.04

# 安装Python 3.10
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3.10-venv \
    && rm -rf /var/lib/apt/lists/*

# 创建符号链接确保python指向python3.10
RUN ln -sf /usr/bin/python3.10 /usr/bin/python

## DO NOT EDIT these 3 lines.
RUN mkdir /challenge
COPY ./ /challenge
WORKDIR /challenge

## 安装依赖
RUN pip install -r requirements.txt