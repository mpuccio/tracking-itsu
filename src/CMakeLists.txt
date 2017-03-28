set(MODULE src)

set(SRCS
  CACell.cxx
  CACluster.cxx
  CAEvent.cxx
  CAEventLoader.cxx
  CALayer.cxx
  CAIndexTable.cxx
  CARoad.cxx
  CATracker.cxx
  CATrackerContext.cxx
  CATracklet.cxx
  CAMathUtils.cxx
)

include_directories(${TRACKING-ITSU_SOURCE_DIR}/include)

add_library(${MODULE} ${SRCS})