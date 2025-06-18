#!/bin/bash
docker run --rm -e PYVISTA_OFF_SCREEN=true -ti -v $(pwd):/shared -w /shared --entrypoint /bin/bash dolfinx/lab:nightly -l
