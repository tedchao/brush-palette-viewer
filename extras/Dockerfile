# Simple docker file to run Brush. Note that as Brush has very few dependencies, you should be able to run
# it on any system already.
#
# This image is for easier server deployements. Nb: This dockerfile explicatly installs nvidia drivers,
# I haven't tested this on non nvidia GPUs. Alternatively, you should be able
# to run with host drivers.
FROM rust:1.86 as builder
ADD https://github.com/ArthurBrussee/brush/ .
RUN git clone https://github.com/ArthurBrussee/brush
RUN cargo build --release --manifest-path ./brush/Cargo.toml

FROM ubuntu:22.04

# Install dependencies.
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libvulkan1 \
    vulkan-tools \
    mesa-vulkan-drivers \
    nvidia-driver-570 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir /workdir
COPY --from=builder ./brush/target/release/brush_app /workdir
WORKDIR /workdir

# Below command creates home dir for 1000 UID user if it is not present.
RUN if ! id 1000; then useradd -m -u 1000 clouduser; fi
RUN chown -R 1000:root /workdir && chmod -R 775 /workdir

# Launch training.
ENTRYPOINT ["./brush_app"]
