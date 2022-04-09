

#### Running with Docker
```bash
# Build image
$ docker build -t image_warper:latest .

# Test run with output display
$ xhost +local:docker
$ docker run --rm -it --name image_warper image_warper:latest bash
$ export | grep -i display ; declare -x DISPLAY=":0" ; export DISPLAY=:0
$ cd backend; python main.py -cas -cps -show

```