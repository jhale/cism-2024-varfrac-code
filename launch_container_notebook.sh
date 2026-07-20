#!/bin/bash
podman run -p 8888:8888 -ti -v $(pwd):/shared -w /shared dolfinx/lab:v0.11.0
