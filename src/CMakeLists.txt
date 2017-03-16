set(MODULE src)

set(SRCS
  CACell.cpp
  CARoad.cpp
  CATracker.cpp
)

include_directories(${TRACKING-ITSU_SOURCE_DIR}/include)

add_library(${MODULE} ${SRCS})