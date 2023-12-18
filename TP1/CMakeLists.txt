cmake_minimum_required(VERSION 3.0.0)
project( TP1 )


find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable( tp1 tp1.cpp )
target_link_libraries( tp1 ${OpenCV_LIBS} )
