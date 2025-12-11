cmake_minimum_required(VERSION 3.18)
project(pm_core LANGUAGES CXX)

set(PYBIND11_FINDPYTHON ON CACHE BOOL "Use modern FindPython" FORCE)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(pm_core
    src/module.cpp
    src/catalog.cpp
)

target_include_directories(pm_core PRIVATE include)
target_compile_features(pm_core PRIVATE cxx_std_17)

# >>> THIS PART MAKES THE .so GO INTO THE WHEEL <<<
install(TARGETS pm_core LIBRARY DESTINATION .)

