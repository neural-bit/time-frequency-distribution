cmake_minimum_required(VERSION 3.18)

project(cuda_tfd VERSION 0.1 LANGUAGES CXX CUDA)

set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(QT NAMES Qt6 Qt5 REQUIRED COMPONENTS Widgets)
find_package(Qt${QT_VERSION_MAJOR} REQUIRED COMPONENTS Widgets)

set(PROJECT_SOURCES
    main.cpp
    mainwindow.cpp
    mainwindow.h
    mainwindow.ui
    cwt.cuh
    cwt.cu
)

if(${QT_VERSION_MAJOR} GREATER_EQUAL 6)
    qt_add_executable(cuda_tfd
        MANUAL_FINALIZATION
        ${PROJECT_SOURCES}
    )
else()
    add_executable(cuda_tfd ${PROJECT_SOURCES})
endif()

target_link_libraries(cuda_tfd PRIVATE Qt${QT_VERSION_MAJOR}::Widgets)

# Enable CUDA features
set_target_properties(cuda_tfd PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

include(GNUInstallDirs)
install(TARGETS cuda_tfd
    BUNDLE DESTINATION .
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

if(QT_VERSION_MAJOR EQUAL 6)
    qt_finalize_executable(cuda_tfd)
endif()
