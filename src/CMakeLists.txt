set(MODULE src)

set(SRCS
  CACell.cxx
  CACluster.cxx
  CAEvent.cxx
  CAEventLoader.cxx
  CALayer.cxx
  CALookupTable.cxx
  CARoad.cxx
  CATracker.cxx
  CATracklet.cxx
  CAMathUtils.cxx
)

include_directories(${TRACKING-ITSU_SOURCE_DIR}/include)

add_library(${MODULE} ${SRCS})