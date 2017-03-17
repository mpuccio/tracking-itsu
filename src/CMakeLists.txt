set(MODULE src)

set(SRCS
  CACell.cxx
  CAEvent.cxx
  CAEventLoader.cxx
  CAHit.cxx
  CARoad.cxx
  CATracker.cxx
  CAUtils.cxx
)

include_directories(${TRACKING-ITSU_SOURCE_DIR}/include)

add_library(${MODULE} ${SRCS})