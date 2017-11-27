
PROJECT(echocv CXX)
CMAKE_MINIMUM_REQUIRED(VERSION 3.1 FATAL_ERROR)
set(PROJECT_VERSION 0.1.0)
include(GNUInstallDirs)

option(BUILD_APPS "Build echocv apps" ON)
option(BUILD_PYTHON "Build python bindings" ON)
option(BUILD_TESTS "Build test programs" OFF)
option(BUILD_HIGHGUI "Build with more GUI debug support" OFF)

set(ECHOCV_PYTHON_VERSION "2" CACHE STRING "Python version to use for compiling python bindings")

SET(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

SET(EXECUTABLE_OUTPUT_PATH ${CMAKE_CURRENT_BINARY_DIR})
SET(LIBRARY_OUTPUT_PATH ${CMAKE_CURRENT_BINARY_DIR})
SET(CMAKE_CXX_STANDARD 11)

IF (BUILD_HIGHGUI)
FIND_PACKAGE(OpenCV REQUIRED core imgproc video calib3d highgui)
add_definitions(-DENABLE_HIGHGUI)
ELSE()
FIND_PACKAGE(OpenCV REQUIRED core imgproc video calib3d)
ENDIF()
FIND_PACKAGE(echolib REQUIRED)

INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/src/ ${CMAKE_CURRENT_SOURCE_DIR}/include/ ${echolib_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIR})

ADD_LIBRARY(echocv SHARED src/messages.cpp)
TARGET_LINK_LIBRARIES(echocv m ${OpenCV_LIBS} ${echolib_LIBRARIES})
install(TARGETS echocv EXPORT echocv_targets DESTINATION ${CMAKE_INSTALL_LIBDIR})
INSTALL(FILES include/echolib/opencv.h DESTINATION include/echolib)

set_target_properties(echocv PROPERTIES
    VERSION 0.0.0
    SOVERSION 0
)

# CMake config file
include(CMakePackageConfigHelpers)

set(LIB_INSTALL_DIR ${CMAKE_INSTALL_LIBDIR})
set(INCLUDE_INSTALL_DIR ${CMAKE_INSTALL_INCLUDEDIR})

configure_package_config_file(echocv-config.cmake.in
    ${PROJECT_BINARY_DIR}/echocv-config.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/echocv
    PATH_VARS LIB_INSTALL_DIR INCLUDE_INSTALL_DIR)

write_basic_package_version_file(
    ${PROJECT_BINARY_DIR}/echocv-config-version.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/echocv-config.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/echocv-config-version.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/echocv)

install(
    EXPORT echocv_targets
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/echocv
    FILE echocv-targets.cmake
)

IF (BUILD_PYTHON)

    find_path(PYBIND11_INCLUDE_DIR pybind11/pybind11.h HINTS /usr/include DOC "PyBind11 include directory")
    find_package(PythonInterp ${ECHO_PYTHON_VERSION} REQUIRED)
    find_package(PythonLibs ${ECHO_PYTHON_VERSION} REQUIRED)

    add_library(pyechocv SHARED  src/conversion.cpp src/python.cpp)
    target_link_libraries(pyechocv echocv ${PYTHON_LIBRARY} ${echolib_LIBRARIES} ${OpenCV_LIBS})
    set_target_properties(pyechocv PROPERTIES PREFIX "")
    target_include_directories(pyechocv PRIVATE ${PYTHON_INCLUDE_DIRS} ${PYBIND11_INCLUDE_DIR})

    # Determine installation path
    execute_process(COMMAND
        ${PYTHON_EXECUTABLE} -c "from distutils.sysconfig import get_python_lib; print(get_python_lib(prefix='${CMAKE_INSTALL_PREFIX}'))"
        OUTPUT_VARIABLE PYTHON_SITE_PACKAGES OUTPUT_STRIP_TRAILING_WHITESPACE)

    install(TARGETS pyechocv DESTINATION ${PYTHON_SITE_PACKAGES}/echocv)
    install(DIRECTORY python/echocv DESTINATION ${PYTHON_SITE_PACKAGES})

ENDIF ()

IF (BUILD_APPS)

ADD_EXECUTABLE(echocv_camera src/cameraserver.cpp)
TARGET_LINK_LIBRARIES(echocv_camera ${echolib_LIBRARIES} echocv m ${OpenCV_LIBS})

ADD_EXECUTABLE(echocv_image src/imageserver.cpp)
TARGET_LINK_LIBRARIES(echocv_image ${echolib_LIBRARIES} echocv m ${OpenCV_LIBS})

ADD_EXECUTABLE(echocv_video src/videoserver.cpp)
TARGET_LINK_LIBRARIES(echocv_video ${echolib_LIBRARIES} echocv m ${OpenCV_LIBS})

ADD_EXECUTABLE(echocv_artrack src/artrack.cpp)
TARGET_LINK_LIBRARIES(echocv_artrack ${echolib_LIBRARIES} echocv m ${OpenCV_LIBS} stdc++fs)

install(TARGETS echocv_camera echocv_image echocv_video echocv_artrack DESTINATION ${CMAKE_INSTALL_BINDIR})

ENDIF ()

IF (BUILD_TESTS)

ADD_EXECUTABLE(test_images tests/images.cpp)
TARGET_LINK_LIBRARIES(test_images ${echolib_LIBRARIES} echocv m ${OpenCV_LIBS})

ENDIF ()
