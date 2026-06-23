if (BUILD_AMD)
    if (APPLE)
        # Apple ships only <OpenCL/*.h>; the Khronos <CL/opencl.hpp> the code
        # includes comes from the vcpkg `opencl` port headers. Apple's framework
        # IS the OpenCL platform, but it is frozen at 1.2 and exports no 2.0+
        # symbols (clCreateCommandQueueWithProperties), and the Khronos ICD
        # loader finds no vendor on macOS. So: take the headers from vcpkg, link
        # the framework, and compile the C++ wrapper for 1.2.
        find_package(OpenCL CONFIG REQUIRED)
        set(OpenCL_INCLUDE_DIRS "")
        foreach (LM_CL_TARGET OpenCL::Headers OpenCL::HeadersCpp)
            if (TARGET ${LM_CL_TARGET})
                get_target_property(LM_CL_INC ${LM_CL_TARGET} INTERFACE_INCLUDE_DIRECTORIES)
                list(APPEND OpenCL_INCLUDE_DIRS ${LM_CL_INC})
            endif()
        endforeach()
        # Link the framework explicitly via the linker flag: a bare
        # find_library(OpenCL) would match vcpkg's own libOpenCL.a (the
        # vendorless ICD loader) first, since the vcpkg toolchain prepends its
        # lib dir to the search path.
        set(OpenCL_LIBRARIES "-framework OpenCL")
        add_compile_definitions(
            CL_HPP_ENABLE_EXCEPTIONS=true
            CL_HPP_TARGET_OPENCL_VERSION=120
            CL_HPP_MINIMUM_OPENCL_VERSION=120
            CL_TARGET_OPENCL_VERSION=120
            CL_USE_DEPRECATED_OPENCL_1_2_APIS
            AMD_ENABLE
        )
    else()
        find_package(OpenCL COMPONENTS OpenCL)
        add_compile_definitions(
            CL_HPP_ENABLE_EXCEPTIONS=true     # Enable C++ exceptions in OpenCL C++ wrapper
            CL_HPP_TARGET_OPENCL_VERSION=300  # Target OpenCL 3.0
            CL_HPP_MINIMUM_OPENCL_VERSION=200 # Minimum OpenCL 2.0 required
            AMD_ENABLE                        # Enable AMD-specific features
        )
    endif()
endif()
