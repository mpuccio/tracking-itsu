set(MODULE src)

set(SRCS
  CACell.cxx
  CACluster.cxx
  CAEvent.cxx
  CAIOUtils.cxx
  CALabel.cxx
  CALayer.cxx
  CAIndexTableUtils.cxx
  CARoad.cxx
  CATracker.cxx
  CATrackingUtils.cxx
  CATracklet.cxx
  CAMathUtils.cxx
)

include_directories(${TRACKING-ITSU_SOURCE_DIR}/include)

add_library(${MODULE} ${SRCS})