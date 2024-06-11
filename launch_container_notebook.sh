#!/bin/bash
docker run -p 8888:8888 -ti -v $(pwd):/shared -w /shared dolfinx/lab:v0.8.0
