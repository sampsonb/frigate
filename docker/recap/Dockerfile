# Recap overlay for stock Frigate 0.17.2.
#
# Starts from ghcr.io/blakeblackshear/frigate:0.17.2 and replaces only the
# Python modules this feature changes, plus a rebuilt web UI. OpenCV, ffmpeg,
# FastAPI, Peewee, and NumPy are already in that image, so this file installs
# no system packages and no Python packages.
#
# The web stage matches docker/main/Dockerfile so asset paths stay the same.
# CI (docker/recap/check_overlay.py) fails if a runtime file changed on this
# branch is missing from the COPY list below.

FROM node:20 AS web

WORKDIR /work
COPY web/package.json web/package-lock.json ./
RUN npm install

COPY web/ ./
RUN npm run build \
    && mv dist/BASE_PATH/monacoeditorwork/* dist/assets/ \
    && rm -rf dist/BASE_PATH

# Index digest for tag 0.17.2 (image 0.17.2-3d4dd3a). Docker selects the
# platform manifest from this index, so an amd64 host gets the amd64 image.
FROM ghcr.io/blakeblackshear/frigate:0.17.2@sha256:d4351369984d4a9e2a49ac59736f6490856a7ea11f7790040746d21496967010

COPY frigate/api/auth.py /opt/frigate/frigate/api/auth.py
COPY frigate/api/defs/tags.py /opt/frigate/frigate/api/defs/tags.py
COPY frigate/api/fastapi_app.py /opt/frigate/frigate/api/fastapi_app.py
COPY frigate/api/recap.py /opt/frigate/frigate/api/recap.py
COPY frigate/app.py /opt/frigate/frigate/app.py
COPY frigate/config/camera/camera.py /opt/frigate/frigate/config/camera/camera.py
COPY frigate/config/config.py /opt/frigate/frigate/config/config.py
COPY frigate/config/recap.py /opt/frigate/frigate/config/recap.py
COPY frigate/const.py /opt/frigate/frigate/const.py
COPY frigate/recap/ /opt/frigate/frigate/recap/
COPY --from=web /work/dist/ /opt/frigate/web/
