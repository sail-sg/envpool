COMMIT_HASH=$(git log -1 --format=%h)
docker build -t $1:${COMMIT_HASH} -f Dockerfile .
if [ $2 ]
then
    docker run --mount type=bind,source=/,target=/host -it $1:${COMMIT_HASH} bash
fi
echo successfully build docker tag: $1:${COMMIT_HASH}
