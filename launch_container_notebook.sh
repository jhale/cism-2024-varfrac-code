#!/bin/bash
docker run -p 8888:8888 -ti -v $(pwd):/shared -w /shared dolfinx/lab:nightly
