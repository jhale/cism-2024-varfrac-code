#!/bin/bash
docker run -ti -v $(pwd):/shared -w /shared dolfinx/dolfinx:v0.8.0
