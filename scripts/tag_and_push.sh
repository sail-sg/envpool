COMMIT_HASH=$(git log -1 --format=%h)
docker tag $1:${COMMIT_HASH} harbor.seacloud.garenanow.com/sail/$1:${COMMIT_HASH} \
    && docker push harbor.seacloud.garenanow.com/sail/$1:${COMMIT_HASH}
