#pragma once

#include <string>


namespace common
{
    namespace opencl
    {
        // Apple's OpenCL platform is named "Apple" (no "AMD" vendor string) and
        // does not implement cl_amd_device_attribute_query. On macOS accept any
        // platform and query the generic device name; elsewhere keep the existing
        // AMD-vendor filter and the AMD board-name extension.
        inline bool isUsablePlatform(std::string const& platformName)
        {
#if defined(__APPLE__)
            (void)platformName;
            return true;
#else
            return std::string::npos != platformName.find("AMD");
#endif
        }
    }
}

#if defined(__APPLE__)
    #define LM_OPENCL_BOARD_NAME CL_DEVICE_NAME
#else
    #define LM_OPENCL_BOARD_NAME CL_DEVICE_BOARD_NAME_AMD
#endif
