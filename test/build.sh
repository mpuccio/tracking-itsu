set -ex
g++ --version
cmake --version
mkdir -p ${TRACKINGITSU_BUILD_DIR} 
cd ${TRACKINGITSU_BUILD_DIR}
cmake -DTRACKINGITSU_TARGET_DEVICE=${TRACKINGITSU_TARGET_DEVICE} ${TRACKINGITSU_SRC_DIR}
make

if [[ $TRACKINGITSU_TARGET_DEVICE == "CPU" ]]; then
    wget -q http://personalpages.to.infn.it/~puccio/data.tgz
    tar -xvzf data.tgz
    wget -q personalpages.to.infn.it/~puccio/labels.tgz
    tar -xvzf labels.tgz
    ./tracking-itsu-main data.txt labels.txt
fi
