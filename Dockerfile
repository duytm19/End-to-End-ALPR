# Bước 1: Chọn Image Nền
# Sử dụng image chính thức của NVIDIA với CUDA 12.1 và cuDNN 8. Đây là nền tảng vững chắc.
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# Thiết lập biến môi trường để tránh các câu hỏi tương tác khi cài đặt
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Ho_Chi_Minh

# Bước 2: Cài đặt các gói hệ thống và Python
# Cài đặt Python, pip và các thư viện cần thiết cho OpenCV.
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*
 
# Bước 3: Thiết lập thư mục làm việc
WORKDIR /app

# Bước 4: Cài đặt các thư viện Python
# Copy file requirements trước để tận dụng Docker cache.
# Lớp này chỉ build lại nếu file requirements.txt thay đổi.
COPY requirements.txt .
RUN pip3 install --no-cache-dir --timeout=600 -r requirements.txt

# Bước 5: Sao chép mã nguồn và các model vào container
# Copy mã nguồn trong thư mục src
COPY ./src/ /app/src/
COPY ./models/ /app/models/

# Copy toàn bộ models
COPY ./models/ /app/models/

# Bước 6: Định nghĩa lệnh chạy mặc định
# Khi container khởi động, nó sẽ chạy lệnh này.
# Chúng ta cung cấp một ví dụ, nhưng bạn sẽ ghi đè nó khi chạy.
ENTRYPOINT ["python3", "src/main.py"]
CMD ["--help"]