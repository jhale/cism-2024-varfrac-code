#!/bin/bash
podman run --rm -e XDG_RUNTIME_DIR=/tmp -e PYVISTA_OFF_SCREEN=true -ti -v $(pwd):/shared -w /shared --entrypoint /bin/bash dolfinx/lab:v0.11.0 -l
