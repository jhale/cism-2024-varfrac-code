#!/bin/bash
docker run -ti -v $(pwd):/shared -w /shared --entrypoint /bin/bash dolfinx/dolfinx:v0.8.0
