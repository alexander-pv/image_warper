
#### Transform image shape transformation into a given contour

Running with Docker
```bash
# Build image
$ docker build -t image_warper:latest .

# Run
$ docker network create --subnet=165.52.0.0/29 demo_subnet
$ docker run --net demo_subnet \
  --ip 165.52.0.2 -p 8501:8501 --rm -t -d \
  --name demo image_warper:latest

# Terminal test with output display
$ xhost +local:docker
$ docker run --rm -it --name image_warper image_warper:latest bash
$ export | grep -i display ; declare -x DISPLAY=":0" ; export DISPLAY=:0
$ cd ./src/backend; python main.py -cas -cps -show
```
