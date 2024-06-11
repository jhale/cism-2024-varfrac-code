#!/bin/bash
docker run -ti -v $(pwd):/shared -w /shared dolfinx/lab:v0.8.0
