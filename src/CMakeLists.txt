set(MODULE src)

set(SRCS
  CACell.cpp
  CAEvent.cpp
  CARoad.cpp
  CATracker.cpp
  CAUtils.cpp
)

include_directories(${TRACKING-ITSU_SOURCE_DIR}/include)

add_library(${MODULE} ${SRCS})